"""
Audio I/O Manager
=================
Uses ``sounddevice`` for non-blocking audio capture and playback.
Supports two operation modes:

* **Direct passthrough** (``direct_passthrough=True``):
  Single full-duplex ``sd.Stream`` — copies ``indata`` directly to ``outdata``
  for minimum-latency mic → headphones loopback.

* **Filter active** (``direct_passthrough=False``):
  Separate input / output streams with inference queues for the processing
  pipeline.

Both modes feed waveform queues so the GUI visualisation stays animated.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import queue
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd

import config as cfg
from audio.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)

DEVICE_PREFS_PATH = Path("device_prefs.json")


# ---------------------------------------------------------------------------
#  Device helpers
# ---------------------------------------------------------------------------

def list_audio_devices() -> List[Dict[str, Any]]:
    """Return a list of all available audio devices with metadata."""
    devices = []
    for idx, d in enumerate(sd.query_devices()):
        devices.append({
            "index": idx,
            "name": d["name"],
            "max_input_channels": d["max_input_channels"],
            "max_output_channels": d["max_output_channels"],
            "default_samplerate": d["default_samplerate"],
        })
    return devices


def select_devices(
    input_idx: Optional[int] = None,
    output_idx: Optional[int] = None,
) -> None:
    """Update the default devices and persist the choice."""
    if input_idx is not None or output_idx is not None:
        current = list(sd.default.device)
        if input_idx is not None:
            current[0] = input_idx
        if output_idx is not None:
            current[1] = output_idx
        sd.default.device = tuple(current)

    # Persist
    try:
        prefs = {"input": input_idx, "output": output_idx}
        DEVICE_PREFS_PATH.write_text(json.dumps(prefs, indent=2))
    except Exception as exc:
        logger.warning("Could not save device prefs: %s", exc)


def _load_device_prefs() -> Tuple[Optional[int], Optional[int]]:
    """Load persisted device preferences, if any."""
    try:
        if DEVICE_PREFS_PATH.exists():
            prefs = json.loads(DEVICE_PREFS_PATH.read_text())
            return prefs.get("input"), prefs.get("output")
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
#  AudioIOManager
# ---------------------------------------------------------------------------

class AudioIOManager:
    """Manages real-time audio capture and playback via sounddevice.

    Parameters
    ----------
    ring_buffer : RingBuffer
        Shared ring buffer for the inference pipeline.
    inference_queue : multiprocessing.Queue
        Raw frames → inference process.
    output_queue : multiprocessing.Queue
        Clean frames ← inference process.
    input_waveform_queue : multiprocessing.Queue
        Raw PCM → GUI waveform display.
    output_waveform_queue : multiprocessing.Queue
        Clean PCM → GUI waveform display.
    direct_passthrough : bool
        If True, skip inference entirely (mic → headphones raw).
    """

    def __init__(
        self,
        ring_buffer: RingBuffer,
        inference_queue: Optional[multiprocessing.Queue] = None,
        output_queue: Optional[multiprocessing.Queue] = None,
        input_waveform_queue: Optional[multiprocessing.Queue] = None,
        output_waveform_queue: Optional[multiprocessing.Queue] = None,
        direct_passthrough: bool = False,
    ) -> None:
        self._ring_buffer = ring_buffer
        self._inference_q = inference_queue
        self._output_q = output_queue
        self._input_wf_q = input_waveform_queue
        self._output_wf_q = output_waveform_queue
        self._passthrough = direct_passthrough

        self._stream: Optional[sd.Stream] = None
        self._input_stream: Optional[sd.InputStream] = None
        self._output_stream: Optional[sd.OutputStream] = None
        self._running = False
        self._retry_count = 0
        self._max_retries = 3

        # Apply persisted device prefs
        saved_in, saved_out = _load_device_prefs()
        input_dev = cfg.INPUT_DEVICE_INDEX or saved_in
        output_dev = cfg.OUTPUT_DEVICE_INDEX or saved_out
        if input_dev is not None or output_dev is not None:
            select_devices(input_dev, output_dev)

        # Audio settings
        sd.default.latency = ("low", "low")
        sd.default.dtype = ("float32", "float32")

    # ── Callbacks ────────────────────────────────────────────────────────

    def _passthrough_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Full-duplex passthrough: ``outdata[:] = indata[:]``."""
        if status:
            logger.warning("Passthrough stream status: %s", status)

        # Zero-copy passthrough
        outdata[:] = indata

        # Feed waveform queues
        frame_copy = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self._push_waveform(self._input_wf_q, frame_copy)
        self._push_waveform(self._output_wf_q, frame_copy)

    def _input_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Input-only callback: write to ring buffer + inference queue."""
        if status:
            logger.warning("Input stream status: %s", status)

        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.ravel().copy()

        # Write to ring buffer
        self._ring_buffer.write(mono)

        # Push to inference queue
        if self._inference_q is not None:
            try:
                self._inference_q.put_nowait(mono)
            except queue.Full:
                pass  # Drop frame rather than block

        # Feed input waveform
        self._push_waveform(self._input_wf_q, mono)

    def _output_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Output-only callback: read from output queue."""
        if status:
            logger.warning("Output stream status: %s", status)

        if self._output_q is not None:
            try:
                clean_frame = self._output_q.get_nowait()
                if clean_frame.ndim == 1:
                    clean_frame = clean_frame[:, np.newaxis]
                n = min(len(clean_frame), len(outdata))
                outdata[:n] = clean_frame[:n]
                outdata[n:] = 0.0
                self._push_waveform(self._output_wf_q, clean_frame[:, 0].copy())
            except queue.Empty:
                outdata[:] = 0.0  # Silence if nothing available
        else:
            outdata[:] = 0.0

    # ── Waveform helper ──────────────────────────────────────────────────

    @staticmethod
    def _push_waveform(q: Optional[multiprocessing.Queue], data: np.ndarray) -> None:
        """Non-blocking push to a waveform queue — silently drops if full."""
        if q is None:
            return
        try:
            q.put_nowait(data)
        except (queue.Full, Exception):
            pass

    # ── Stream lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Open and start audio stream(s)."""
        if self._running:
            return

        try:
            if self._passthrough:
                self._stream = sd.Stream(
                    samplerate=cfg.SAMPLE_RATE,
                    blocksize=cfg.PASSTHROUGH_BLOCK_SIZE,
                    channels=cfg.CHANNELS,
                    dtype="float32",
                    callback=self._passthrough_callback,
                    latency="low",
                )
                self._stream.start()
                logger.info(
                    "Passthrough stream started  (sr=%d, bs=%d)",
                    cfg.SAMPLE_RATE,
                    cfg.PASSTHROUGH_BLOCK_SIZE,
                )
            else:
                self._input_stream = sd.InputStream(
                    samplerate=cfg.SAMPLE_RATE,
                    blocksize=cfg.BLOCK_SIZE,
                    channels=cfg.CHANNELS,
                    dtype="float32",
                    callback=self._input_callback,
                    latency="low",
                )
                self._output_stream = sd.OutputStream(
                    samplerate=cfg.SAMPLE_RATE,
                    blocksize=cfg.BLOCK_SIZE,
                    channels=cfg.CHANNELS,
                    dtype="float32",
                    callback=self._output_callback,
                    latency="low",
                )
                self._input_stream.start()
                self._output_stream.start()
                logger.info(
                    "Dual-stream started  (sr=%d, bs=%d)",
                    cfg.SAMPLE_RATE,
                    cfg.BLOCK_SIZE,
                )

            self._running = True
            self._retry_count = 0
        except sd.PortAudioError as exc:
            logger.error("PortAudio error opening streams: %s", exc)
            self._retry_count += 1
            if self._retry_count <= self._max_retries:
                logger.info("Retrying in 500 ms (attempt %d/%d)…",
                            self._retry_count, self._max_retries)
                time.sleep(0.5)
                self.start()
            else:
                logger.error("Max retries exceeded — audio unavailable.")

    def stop(self) -> None:
        """Stop and close all audio streams."""
        self._running = False
        for stream in (self._stream, self._input_stream, self._output_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        self._stream = None
        self._input_stream = None
        self._output_stream = None
        logger.info("Audio streams stopped.")

    def set_passthrough(self, enabled: bool) -> None:
        """Switch between passthrough and filter mode at runtime."""
        if enabled == self._passthrough:
            return
        self.stop()
        self._passthrough = enabled
        self.start()

    def set_input_device(self, device_index: int) -> None:
        """Change the input device at runtime."""
        self.stop()
        select_devices(input_idx=device_index)
        self.start()

    def set_output_device(self, device_index: int) -> None:
        """Change the output device at runtime."""
        self.stop()
        select_devices(output_idx=device_index)
        self.start()

    def get_active_device_names(self) -> Tuple[str, str]:
        """Return (input_name, output_name) for the currently active devices."""
        try:
            devs = sd.query_devices()
            default = sd.default.device
            in_name = devs[default[0]]["name"] if default[0] is not None else "(System Default)"
            out_name = devs[default[1]]["name"] if default[1] is not None else "(System Default)"
            return in_name, out_name
        except Exception:
            return "(Unknown)", "(Unknown)"

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

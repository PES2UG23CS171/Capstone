"""
Process Manager
===============
Separates the GUI process from the audio-processing process using Python
``multiprocessing`` to prevent UI interactions from causing audio dropouts.

Classes:
* ``AudioProcess``  — runs in a separate OS process: audio I/O + inference
* ``ProcessManager`` — owned by the GUI, spawns and manages AudioProcess
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import time
import traceback
from typing import Optional, Tuple

import numpy as np

import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Audio Process (child)
# ---------------------------------------------------------------------------

class AudioProcess(multiprocessing.Process):
    """Runs in a separate OS process: audio I/O + inference pipeline.

    Communication with the GUI happens through three queue pairs:
    * ``command_queue``       — commands from GUI (start, stop, set params)
    * ``stats_queue``         — metrics back to GUI every 100 ms
    * ``input_waveform_queue`` / ``output_waveform_queue`` — PCM for waveforms
    """

    def __init__(
        self,
        command_queue: multiprocessing.Queue,
        stats_queue: multiprocessing.Queue,
        input_waveform_queue: multiprocessing.Queue,
        output_waveform_queue: multiprocessing.Queue,
        audio_config: Optional[dict] = None,
    ) -> None:
        super().__init__(daemon=True, name="AudioProcess")
        self._cmd_q = command_queue
        self._stats_q = stats_queue
        self._input_wf_q = input_waveform_queue
        self._output_wf_q = output_waveform_queue
        self._audio_config = audio_config or {}

    def run(self) -> None:
        """Entry point — runs in the child process."""
        logging.basicConfig(
            level=logging.INFO,
            format="[AUDIO %(levelname)s] %(message)s",
        )
        log = logging.getLogger("audio_process")

        try:
            from audio.ring_buffer import RingBuffer
            from audio.audio_io import AudioIOManager

            # Initialise components
            ring_buf = RingBuffer()
            passthrough = self._audio_config.get("passthrough", cfg.DIRECT_PASSTHROUGH)

            # Create inference queue (for filter mode)
            inference_q: multiprocessing.Queue = multiprocessing.Queue(maxsize=100)
            output_q: multiprocessing.Queue = multiprocessing.Queue(maxsize=100)

            audio_mgr = AudioIOManager(
                ring_buffer=ring_buf,
                inference_queue=inference_q,
                output_queue=output_q,
                input_waveform_queue=self._input_wf_q,
                output_waveform_queue=self._output_wf_q,
                direct_passthrough=passthrough,
            )

            # Try to load ONNX model for filter mode
            onnx_runner = None
            pipeline = None
            if not passthrough:
                try:
                    from inference.onnx_runner import ONNXInferenceRunner
                    from inference.pipeline import RealTimePipeline
                    from pathlib import Path

                    if Path(cfg.ONNX_MODEL_PATH).exists():
                        onnx_runner = ONNXInferenceRunner(cfg.ONNX_MODEL_PATH)
                        onnx_runner.warmup()
                        pipeline = RealTimePipeline(onnx_runner, ring_buf)
                        log.info("ONNX model loaded — filter active.")
                    else:
                        log.warning("ONNX model not found — running in passthrough.")
                        passthrough = True
                        audio_mgr.set_passthrough(True)
                except ImportError as exc:
                    log.warning("Inference dependencies unavailable: %s", exc)
                    passthrough = True
                    audio_mgr.set_passthrough(True)

            audio_mgr.start()
            log.info("Audio engine started (passthrough=%s)", passthrough)

            # ── Event loop ───────────────────────────────────────────────
            last_stats_time = time.monotonic()
            running = True

            while running:
                # Process commands
                try:
                    cmd = self._cmd_q.get(timeout=0.05)
                except queue.Empty:
                    cmd = None

                if cmd is not None:
                    action = cmd.get("action")
                    log.debug("Command: %s", action)

                    if action == "stop":
                        running = False
                    elif action == "set_passthrough":
                        passthrough = cmd.get("value", False)
                        audio_mgr.set_passthrough(passthrough)
                    elif action == "set_suppression_level":
                        if pipeline:
                            pipeline.suppression_level = float(cmd.get("value", 1.0))
                    elif action == "bypass":
                        if pipeline:
                            pipeline.bypass_mode = bool(cmd.get("value", False))
                    elif action == "set_input_device":
                        audio_mgr.set_input_device(int(cmd["value"]))
                    elif action == "set_output_device":
                        audio_mgr.set_output_device(int(cmd["value"]))

                # Process inference (filter mode)
                if not passthrough and pipeline is not None:
                    try:
                        raw_frame = inference_q.get_nowait()
                        clean_frame = pipeline.process_chunk(raw_frame)
                        try:
                            output_q.put_nowait(clean_frame)
                        except queue.Full:
                            pass
                    except queue.Empty:
                        pass

                # Send stats periodically
                now = time.monotonic()
                if now - last_stats_time >= 0.1:
                    last_stats_time = now
                    stats = {
                        "running": True,
                        "passthrough": passthrough,
                    }
                    if pipeline:
                        stats.update(pipeline.get_stats())
                    try:
                        self._stats_q.put_nowait(stats)
                    except queue.Full:
                        pass

            # Cleanup
            audio_mgr.stop()
            log.info("Audio engine stopped.")

        except Exception:
            log.error("Audio process crashed:\n%s", traceback.format_exc())


# ---------------------------------------------------------------------------
#  Process Manager (GUI side)
# ---------------------------------------------------------------------------

class ProcessManager:
    """Owned by the GUI. Spawns and manages the AudioProcess.

    Provides a clean API for the GUI to control the audio engine
    without directly touching multiprocessing primitives.
    """

    def __init__(self) -> None:
        self.command_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=256)
        self.stats_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=256)
        self.input_waveform_queue: multiprocessing.Queue = multiprocessing.Queue(
            maxsize=cfg.WAVEFORM_QUEUE_MAXSIZE
        )
        self.output_waveform_queue: multiprocessing.Queue = multiprocessing.Queue(
            maxsize=cfg.WAVEFORM_QUEUE_MAXSIZE
        )
        self._process: Optional[AudioProcess] = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start_audio_engine(self, passthrough: bool = False) -> None:
        """Spawn the audio processing child process."""
        if self._process is not None and self._process.is_alive():
            logger.warning("Audio engine already running.")
            return

        self._process = AudioProcess(
            command_queue=self.command_queue,
            stats_queue=self.stats_queue,
            input_waveform_queue=self.input_waveform_queue,
            output_waveform_queue=self.output_waveform_queue,
            audio_config={"passthrough": passthrough},
        )
        self._process.start()
        logger.info("Audio engine started (pid=%s)", self._process.pid)

    def stop_audio_engine(self) -> None:
        """Gracefully stop the audio engine."""
        if self._process is None or not self._process.is_alive():
            return

        self._send_command({"action": "stop"})
        self._process.join(timeout=3.0)

        if self._process.is_alive():
            logger.warning("Audio engine did not stop — terminating.")
            self._process.terminate()

        self._process = None
        logger.info("Audio engine stopped.")

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    # ── Commands ─────────────────────────────────────────────────────────

    def set_suppression_level(self, level: float) -> None:
        self._send_command({"action": "set_suppression_level", "value": level})

    def set_bypass(self, enabled: bool) -> None:
        self._send_command({"action": "bypass", "value": enabled})

    def set_passthrough(self, enabled: bool) -> None:
        self._send_command({"action": "set_passthrough", "value": enabled})

    def set_input_device(self, device_index: int) -> None:
        self._send_command({"action": "set_input_device", "value": device_index})

    def set_output_device(self, device_index: int) -> None:
        self._send_command({"action": "set_output_device", "value": device_index})

    # ── Stats & waveforms ────────────────────────────────────────────────

    def get_stats(self) -> Optional[dict]:
        """Non-blocking — returns the latest stats or None."""
        latest = None
        while True:
            try:
                latest = self.stats_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def drain_waveform_queues(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Drain ALL pending frames from both waveform queues.

        Returns
        -------
        (input_pcm, output_pcm) : tuple
            Concatenated audio frames, or ``(None, None)`` if empty.
        """
        input_frames = self._drain_queue(self.input_waveform_queue)
        output_frames = self._drain_queue(self.output_waveform_queue)

        input_pcm = np.concatenate(input_frames) if input_frames else None
        output_pcm = np.concatenate(output_frames) if output_frames else None

        return input_pcm, output_pcm

    # ── Helpers ──────────────────────────────────────────────────────────

    def _send_command(self, cmd: dict) -> None:
        try:
            self.command_queue.put_nowait(cmd)
        except queue.Full:
            logger.warning("Command queue full — dropping: %s", cmd.get("action"))

    @staticmethod
    def _drain_queue(q: multiprocessing.Queue) -> list:
        frames = []
        while True:
            try:
                frames.append(q.get_nowait())
            except queue.Empty:
                break
        return frames


if __name__ == "__main__":
    print("Starting audio engine in passthrough mode…")
    mgr = ProcessManager()
    mgr.start_audio_engine(passthrough=True)

    try:
        while True:
            stats = mgr.get_stats()
            if stats:
                print(f"  Stats: {stats}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    mgr.stop_audio_engine()
    print("Done.")

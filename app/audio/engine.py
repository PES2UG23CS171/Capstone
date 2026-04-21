"""
Audio-processing engine — runs in a dedicated child process.

Architecture
~~~~~~~~~~~~
::

    ┌─ GUI Process ─────────────────┐        ┌─ Audio-Engine Process ──────────┐
    │  PyQt6  +  pystray            │  cmd_q  │  sounddevice.Stream             │
    │                               │ ──────► │  + StubDenoiser (→ ONNXDenoiser)│
    │  Control-window sliders/      │  evt_q  │                                 │
    │  toggles, level meters        │ ◄────── │  Non-blocking audio callbacks   │
    └───────────────────────────────┘        └─────────────────────────────────┘

The engine is spawned via ``multiprocessing.Process(target=run_engine)``.
It owns the ``sounddevice.Stream`` and the inference model.  The GUI
communicates exclusively through two ``multiprocessing.Queue`` objects:

*  **cmd_q** (GUI → Engine): ``Command`` messages.
*  **evt_q** (Engine → GUI): ``Event`` messages (status, device list, errors).
"""

from __future__ import annotations

import logging
import math
import queue
import sys
import time
import traceback
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

from app.audio.devices import query_devices
from app.config import AppConfig
from app.inference.stub import PyTorchDenoiser
from app.ipc.messages import (
    CmdType,
    Command,
    Event,
    EvtType,
    StatusPayload,
)

# Import RealTimeFilter from the PoC module at project root
try:
    # Ensure project root is on path so poc_realtime_transient can be imported
    _project_root = str(Path(__file__).resolve().parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from poc_realtime_transient import RealTimeFilter
    _HAS_REALTIME_FILTER = True
except ImportError:
    _HAS_REALTIME_FILTER = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------


def _db(linear: float) -> float:
    """Convert linear amplitude to dBFS, clamped to -120."""
    return max(20.0 * math.log10(max(linear, 1e-12)), -120.0)


# ---------------------------------------------------------------------------
#  Engine loop (runs inside the child process)
# ---------------------------------------------------------------------------


def run_engine(cmd_q: Queue, evt_q: Queue, cfg: AppConfig) -> None:  # noqa: C901
    """Entry point for the audio-engine child process.

    Parameters
    ----------
    cmd_q : Queue
        Commands from the GUI.
    evt_q : Queue
        Events back to the GUI.
    cfg : AppConfig
        Initial configuration snapshot.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="[ENGINE %(levelname)s] %(message)s",
    )
    log = logging.getLogger("audio.engine")

    # ── Mutable state (modified by GUI commands) ─────────────────────────
    enabled: bool = cfg.suppression_enabled
    strength: float = cfg.suppression_strength
    gain_db: float = cfg.output_gain_db
    gain_lin: float = 10.0 ** (gain_db / 20.0)
    passthrough: bool = False       # direct mic→headphones with zero processing

    input_dev: Optional[int] = cfg.input_device
    output_dev: Optional[int] = cfg.output_device

    xruns: int = 0
    last_status_time: float = 0.0
    current_rtf: float = 0.0

    # Peak levels (written by callback, read by status tick)
    peak_in: float = 0.0
    peak_out: float = 0.0

    # ── Denoiser / RealTimeFilter ────────────────────────────────────────
    rt_filter: Optional["RealTimeFilter"] = None
    if _HAS_REALTIME_FILTER:
        rt_filter = RealTimeFilter(sr=cfg.sample_rate, chunk=128, use_noise_est=True)
        # Retune transient detector for live mic (defaults are for offline demo)
        td = rt_filter.transient
        td.threshold = np.float32(10.0 ** (20.0 / 10.0))       # 20 dB — only real impulses
        td.suppress_gain = np.float32(10.0 ** (-15.0 / 20.0))  # -15 dB — gentle attenuation
        td.hold_samples = int(25.0 * cfg.sample_rate / 1000.0)  # 25 ms hold
        td.release_samples = int(30.0 * cfg.sample_rate / 1000.0)  # 30 ms ramp-up
        td.alpha_release = np.float32(1.0 - np.exp(-1.0 / (0.050 * cfg.sample_rate)))  # faster slow env
        # Retune noise estimator — gentle (max -8 dB cut, preserves voice)
        ne = rt_filter.noise_est
        ne.beta = np.float32(1.5)    # gentle subtraction (was 8.0)
        ne.g_min = np.float32(0.4)   # never cut more than -8 dB (was 0.01 = -40 dB)
        log.info("RealTimeFilter loaded — transient suppression active (live-tuned).")
    else:
        log.warning("RealTimeFilter not available — falling back to StubDenoiser.")

    denoiser = PyTorchDenoiser(
        model_path=cfg.model_path,
        sample_rate=cfg.sample_rate,
        block_size=cfg.block_size,
    )
    denoiser.strength = strength

    # ── sounddevice callback ─────────────────────────────────────────────

    def _audio_callback(
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: sd.CallbackFlags,
        status: sd.CallbackFlags,
    ) -> None:
        nonlocal peak_in, peak_out, xruns, current_rtf

        if status:
            xruns += 1
            log.warning("stream status: %s", status)

        # Measure input level
        peak_in = float(np.max(np.abs(indata)))

        # Direct passthrough: zero processing for minimum latency
        if passthrough:
            out_ch = outdata.shape[1]
            if indata.shape[1] < out_ch:
                outdata[:] = np.tile(indata, (1, out_ch))[:, :out_ch]
            else:
                outdata[:] = indata[:, :out_ch]
            peak_out = peak_in
            current_rtf = 0.0
            return

        if enabled:
            processed = denoiser.process(indata[:, :cfg.channels].copy())
        else:
            processed = indata[:, :cfg.channels].copy()

        # Apply output gain
        processed *= gain_lin

        # Clip to [-1, 1]
        np.clip(processed, -1.0, 1.0, out=processed)

        # Measure output level
        peak_out = float(np.max(np.abs(processed)))

        # Write to output (handle channel mismatch gracefully)
        out_ch = outdata.shape[1]
        if processed.ndim == 1:
            processed = processed[:, np.newaxis]
        if processed.shape[1] < out_ch:
            # Mono→stereo: duplicate
            processed = np.tile(processed, (1, out_ch))
        outdata[:] = processed[:, :out_ch]

    # ── Build and start the stream ───────────────────────────────────────

    stream: Optional[sd.Stream] = None

    def _open_stream() -> sd.Stream:
        s = sd.Stream(
            samplerate=cfg.sample_rate,
            blocksize=cfg.block_size,
            device=(input_dev, output_dev),
            channels=cfg.channels,
            dtype=cfg.dtype,
            callback=_audio_callback,
            latency="high",
        )
        s.start()
        log.info(
            "Stream opened  in=%s  out=%s  sr=%d  bs=%d",
            input_dev,
            output_dev,
            cfg.sample_rate,
            cfg.block_size,
        )
        return s

    try:
        stream = _open_stream()
    except Exception:
        log.error("Failed to open initial stream:\n%s", traceback.format_exc())
        evt_q.put(Event(EvtType.ERROR, "Failed to open audio stream."))

    # ── Main command loop ────────────────────────────────────────────────

    running = True
    while running:
        # --- process pending commands (non-blocking) ---------------------
        try:
            cmd: Command = cmd_q.get(timeout=cfg.status_interval)
        except queue.Empty:
            cmd = None  # type: ignore[assignment]

        if cmd is not None:
            try:
                if cmd.kind == CmdType.SHUTDOWN:
                    log.info("Shutdown command received.")
                    running = False

                elif cmd.kind == CmdType.SET_ENABLED:
                    enabled = bool(cmd.value)
                    log.debug("Suppression enabled = %s", enabled)

                elif cmd.kind == CmdType.SET_STRENGTH:
                    strength = float(np.clip(cmd.value, 0.0, 1.0))
                    denoiser.strength = strength
                    log.debug("Suppression strength = %.2f", strength)

                elif cmd.kind == CmdType.SET_GAIN:
                    gain_db = float(cmd.value)
                    gain_lin = 10.0 ** (gain_db / 20.0)
                    log.debug("Output gain = %.1f dB", gain_db)

                elif cmd.kind == CmdType.SET_INPUT_DEVICE:
                    input_dev = cmd.value
                    log.info("Switching input device → %s", input_dev)
                    if stream is not None:
                        stream.stop()
                        stream.close()
                    stream = _open_stream()

                elif cmd.kind == CmdType.SET_OUTPUT_DEVICE:
                    output_dev = cmd.value
                    log.info("Switching output device → %s", output_dev)
                    if stream is not None:
                        stream.stop()
                        stream.close()
                    stream = _open_stream()

                elif cmd.kind == CmdType.SET_PASSTHROUGH:
                    passthrough = bool(cmd.value)
                    log.info("Passthrough mode = %s", passthrough)

                elif cmd.kind == CmdType.GET_DEVICES:
                    devices = query_devices()
                    evt_q.put(Event(EvtType.DEVICE_LIST, devices))

            except Exception:
                msg = traceback.format_exc()
                log.error("Error handling command %s:\n%s", cmd.kind, msg)
                evt_q.put(Event(EvtType.ERROR, msg))

        # --- periodic status update --------------------------------------
        now = time.monotonic()
        if now - last_status_time >= cfg.status_interval:
            last_status_time = now
            evt_q.put(
                Event(
                    EvtType.STATUS,
                    StatusPayload(
                        running=True,
                        input_level_db=_db(peak_in),
                        output_level_db=_db(peak_out),
                        xruns=xruns,
                        rtf=current_rtf,
                    ),
                )
            )

    # ── Cleanup ──────────────────────────────────────────────────────────
    if stream is not None:
        stream.stop()
        stream.close()
        log.info("Stream closed.")

    evt_q.put(Event(EvtType.ENGINE_STOPPED))
    log.info("Engine process exiting.")


# ---------------------------------------------------------------------------
#  Convenience wrapper for spawning the engine from the GUI process
# ---------------------------------------------------------------------------


class AudioEngineHandle:
    """Thin handle held by the GUI process to manage the child process."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.cmd_q: Queue = Queue(maxsize=256)
        self.evt_q: Queue = Queue(maxsize=256)
        self._process: Optional[Process] = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._process = Process(
            target=run_engine,
            args=(self.cmd_q, self.evt_q, self.cfg),
            daemon=True,
            name="AudioEngine",
        )
        self._process.start()
        logger.info("Audio-engine process started (pid=%s).", self._process.pid)

    def stop(self, timeout: float = 3.0) -> None:
        if self._process is None or not self._process.is_alive():
            return
        self.send(Command(CmdType.SHUTDOWN))
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            logger.warning("Engine did not stop in time — terminating.")
            self._process.terminate()
        self._process = None

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    # ── IPC helpers ──────────────────────────────────────────────────────

    def send(self, cmd: Command) -> None:
        try:
            self.cmd_q.put_nowait(cmd)
        except queue.Full:
            logger.warning("Command queue full — dropping %s", cmd.kind)

    def poll_events(self) -> list[Event]:
        """Drain the event queue (non-blocking). Returns a list."""
        events: list[Event] = []
        while True:
            try:
                events.append(self.evt_q.get_nowait())
            except queue.Empty:
                break
        return events

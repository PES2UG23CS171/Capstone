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
import threading
import time
import traceback
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

from app.audio.devices import query_devices
from app.config import AppConfig
from app.inference.stub import StubDenoiser
from app.ipc.messages import (
    CmdType,
    Command,
    Event,
    EvtType,
    StatusPayload,
)

# Import the voiceiso StreamingPipeline (DeepFilterNet3 + full chain).
try:
    # Ensure project root is on path so the voiceiso package is importable.
    _project_root = str(Path(__file__).resolve().parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from voiceiso.config import PipelineConfig
    from voiceiso.pipeline import StreamingPipeline
    _HAS_PIPELINE = True
except Exception:  # noqa: BLE001 — fall back gracefully if deps missing
    _HAS_PIPELINE = False

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
    # Clamp the initial gain the same way SET_GAIN commands are clamped —
    # a persisted/hand-edited config must not bypass the +6 dB feedback margin.
    gain_db: float = float(np.clip(cfg.output_gain_db, -60.0, 6.0))
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

    # ── voiceiso StreamingPipeline (DeepFilterNet3 + VAD + controller + …) ─
    pipeline: Optional["StreamingPipeline"] = None
    if _HAS_PIPELINE:
        try:
            pcfg = PipelineConfig(sample_rate=cfg.sample_rate, channels=cfg.channels)
            pipeline = StreamingPipeline(pcfg, enh_threads=4)
            log.info("voiceiso pipeline loaded — backends: %s", pipeline.backend_summary)
        except Exception:
            log.error("Failed to init voiceiso pipeline:\n%s", traceback.format_exc())
            pipeline = None
    if pipeline is None:
        log.warning("voiceiso pipeline unavailable — falling back to StubDenoiser passthrough.")
        # Make the degraded (no-denoising) state VISIBLE to the GUI, not silent.
        evt_q.put(Event(
            EvtType.ERROR,
            "voiceiso pipeline failed to load — running in PASSTHROUGH (no "
            "denoising). Check DeepFilterNet / onnxruntime install.",
        ))
    elif pipeline.backend_summary.get("enhancement") != "deepfilternet3":
        evt_q.put(Event(
            EvtType.ERROR,
            "DeepFilterNet3 not loaded — enhancement is PASSTHROUGH. "
            "Install deepfilternet (+ Rust toolchain).",
        ))

    # ── Warm-up (C2): prime every model BEFORE the stream starts. ────────
    # Without this, first-call costs stall the DSP worker 1–2 s at stream
    # start: the callback plays zeros, the input queue fills, and when the
    # worker catches up the output queue parks at full depth — permanently
    # ratcheting mouth-to-ear latency by up to 400 ms.  There is no drain
    # path for that backlog other than the callback-side drain below.
    if pipeline is not None:
        try:
            _t0 = time.perf_counter()
            pipeline.warmup()
            log.info("pipeline warm-up complete in %.2f s", time.perf_counter() - _t0)
        except Exception:
            log.error("pipeline warm-up failed:\n%s", traceback.format_exc())

    denoiser = StubDenoiser(
        model_path=cfg.model_path,
        sample_rate=cfg.sample_rate,
        block_size=cfg.block_size,
    )
    denoiser.strength = strength

    # ── DSP worker thread (H2) ───────────────────────────────────────────
    # DeepFilterNet3 + the full chain must NOT run on the real-time audio
    # callback: at 20 ms blocks a single over-budget block causes an xrun.
    # The callback only enqueues input and dequeues the most-recent output
    # (drop-oldest, bounded latency); this thread does the heavy work.
    _proc_in: queue.Queue = queue.Queue(maxsize=4)
    _proc_out: queue.Queue = queue.Queue(maxsize=4)
    _worker_alive = threading.Event()
    _worker_alive.set()

    def _drop_oldest_put(q: queue.Queue, item) -> None:
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass

    def _flush(q: queue.Queue) -> None:
        """Drain a queue completely (mode/device transitions).

        Without this, toggling passthrough or switching devices replays up to
        400 ms of stale audio still parked in the queues.
        """
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    worker_errors = 0
    worker_err_last_log = 0.0

    def _worker_loop() -> None:
        nonlocal worker_errors, worker_err_last_log
        while _worker_alive.is_set():
            try:
                block = _proc_in.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                t0 = time.perf_counter()
                ctx = pipeline.process_block(block)
                wet = ctx.audio[: len(block)]
                s = strength            # current value (benign read race)
                if s < 1.0:
                    wet = (s * wet + (1.0 - s) * block[: len(wet)]).astype(np.float32)
                budget = len(block) / cfg.sample_rate
                rtf = (time.perf_counter() - t0) / budget if budget > 0 else 0.0
                _drop_oldest_put(_proc_out, (wet, rtf))
            except Exception:
                # Containment: emit the DRY block (passthrough beats silence)
                # so the demo keeps producing audio; tell the GUI once and
                # rate-limit the traceback (it previously spammed every 100 ms).
                worker_errors += 1
                if worker_errors == 1:
                    try:
                        evt_q.put_nowait(Event(
                            EvtType.ERROR,
                            "Pipeline error — passing mic audio through unprocessed "
                            "for affected blocks (see engine log).",
                        ))
                    except queue.Full:
                        pass          # never park the DSP worker on the GUI queue
                now = time.monotonic()
                if now - worker_err_last_log >= 5.0:
                    worker_err_last_log = now
                    log.error("DSP worker process_block failed (%d so far):\n%s",
                              worker_errors, traceback.format_exc())
                _drop_oldest_put(_proc_out, (block.astype(np.float32, copy=False), 0.0))

    _worker: Optional[threading.Thread] = None
    if pipeline is not None:
        _worker = threading.Thread(target=_worker_loop, daemon=True, name="voiceiso-dsp")
        _worker.start()

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
            # Count only — logging (blocking I/O) inside the RT callback
            # worsens the very xrun storm that triggered it; the status loop
            # reports the counter.
            xruns += 1

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

        if enabled and pipeline is not None:
            # Hand the block to the DSP worker thread; emit the most-recent
            # finished block.  The callback never runs DFN3 itself, so a slow
            # block can't xrun the audio device — it only adds bounded latency.
            mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            _drop_oldest_put(_proc_in, mono)
            # Backlog policy: CAP the output backlog at 2 blocks, then take the
            # oldest remaining.  This keeps ONE spare block as a jitter cushion
            # (a single >100 ms worker spike plays the spare instead of
            # silence) while bounding any post-stall latency ratchet at
            # +100 ms.  Red-team simulation showed pure drain-to-freshest has
            # zero cushion (silence gap on every spike — ~1.1 s/min under the
            # measured thermal profile) and the old unbounded backlog parked
            # +400 ms forever; cap-at-2/take-one beats both for a live demo.
            try:
                while _proc_out.qsize() > 2:
                    _proc_out.get_nowait()           # drop stalest beyond the cap
                y, rtf = _proc_out.get_nowait()
                current_rtf = rtf
            except queue.Empty:
                y = np.zeros(frames, dtype=np.float32)   # warmup / brief underrun
            if len(y) < frames:
                y = np.concatenate([y, np.zeros(frames - len(y), dtype=np.float32)])
            processed = y[:frames]
        elif enabled:
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
        # Up to 2 output channels (clamped to device capability): mono into a
        # stereo device (speakers, BlackHole virtual mic) would otherwise
        # leave one channel silent; the callback duplicates mono into both.
        out_info = sd.query_devices(output_dev, "output")
        out_ch = max(1, min(2, int(out_info["max_output_channels"])))
        s = sd.Stream(
            samplerate=cfg.sample_rate,
            blocksize=cfg.block_size,
            device=(input_dev, output_dev),
            channels=(cfg.channels, out_ch),
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
                    # Flush BEFORE the flag flips (so no pre-toggle block is
                    # mistaken for fresh) and again AFTER (so a block the
                    # worker finished mid-toggle can't replay stale audio —
                    # the R5 race).  Worst residue after both: ≤1 block,
                    # bounded further by the backlog cap.
                    _flush(_proc_in); _flush(_proc_out)
                    enabled = bool(cmd.value)
                    _flush(_proc_in); _flush(_proc_out)
                    log.debug("Suppression enabled = %s", enabled)

                elif cmd.kind == CmdType.SET_STRENGTH:
                    strength = float(np.clip(cmd.value, 0.0, 1.0))
                    denoiser.strength = strength
                    log.debug("Suppression strength = %.2f", strength)

                elif cmd.kind == CmdType.SET_GAIN:
                    # Cap positive gain at +6 dB: with open speakers the
                    # mic→speaker loop howls once loop gain exceeds unity,
                    # and DFN3 preserves the returning speech by design.
                    gain_db = float(np.clip(cmd.value, -60.0, 6.0))
                    gain_lin = 10.0 ** (gain_db / 20.0)
                    log.debug("Output gain = %.1f dB", gain_db)

                elif cmd.kind == CmdType.SET_INPUT_DEVICE:
                    input_dev = cmd.value
                    log.info("Switching input device → %s", input_dev)
                    if stream is not None:
                        stream.stop()
                        stream.close()
                    _flush(_proc_in); _flush(_proc_out)
                    stream = _open_stream()

                elif cmd.kind == CmdType.SET_OUTPUT_DEVICE:
                    output_dev = cmd.value
                    log.info("Switching output device → %s", output_dev)
                    if stream is not None:
                        stream.stop()
                        stream.close()
                    _flush(_proc_in); _flush(_proc_out)
                    stream = _open_stream()

                elif cmd.kind == CmdType.SET_PASSTHROUGH:
                    _flush(_proc_in); _flush(_proc_out)
                    passthrough = bool(cmd.value)
                    _flush(_proc_in); _flush(_proc_out)
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

    # Stop the DSP worker thread.
    _worker_alive.clear()
    if _worker is not None:
        _worker.join(timeout=1.0)

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

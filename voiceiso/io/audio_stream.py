"""
Live audio streaming: microphone → pipeline → speakers.

Uses 100 ms blocks — DFN3's efficient design point.  DFN3 does NOT persist GRU
state across calls (its ``_state`` keeps only STFT/ERB analysis state; see
stages/enhancement.py), so per-call temporal context is limited to the STFT
frames inside the block; at 20 ms (≈1–2 frames) measured SI-SDRi is negative,
at 100 ms it is clearly positive.  Because enhancement must not run on the
real-time audio callback thread, audio is shuttled through queues to a worker
thread; the callback only does lock-free enqueue/dequeue.  End-to-end latency
≈ block + one queue hop + compute (~200 ms).
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.pipeline import StreamingPipeline

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None


def resolve_device(spec, kind: str):
    """Resolve a device by index or case-insensitive name substring.

    ``kind`` is 'input' or 'output' (validates channel capability).
    Returns the device index, or None for the system default.
    """
    if spec is None or spec == "":
        return None
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return int(spec)
    want = str(spec).lower()
    ch_key = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = [i for i, d in enumerate(sd.query_devices())
               if want in d["name"].lower() and d[ch_key] > 0]
    if not matches:
        names = [d["name"] for d in sd.query_devices()
                 if d[ch_key] > 0]
        raise SystemExit(f"no {kind} device matching {spec!r}. Available: {names}")
    return matches[0]


class LiveStream:
    def __init__(self, cfg: Optional[PipelineConfig] = None, block_ms: float = 100.0,
                 enh_threads: int = 4, input_device=None, output_device=None) -> None:
        if sd is None:
            raise ImportError("sounddevice is required for LiveStream")
        self.in_dev = resolve_device(input_device, "input")
        self.out_dev = resolve_device(output_device, "output")
        self.cfg = cfg or PipelineConfig()
        self.block = int(self.cfg.sample_rate * block_ms / 1000.0)
        self.pipe = StreamingPipeline(self.cfg, enh_threads=enh_threads)
        # Bounded queues: keep end-to-end latency tight.  16-deep queues let
        # backpressure inflate latency to >300 ms silently under load.
        qsize = max(1, int(self.cfg.live_queue_maxsize))
        self._in_q: queue.Queue = queue.Queue(maxsize=qsize)
        self._out_q: queue.Queue = queue.Queue(maxsize=qsize)
        self._run = False
        self._worker: Optional[threading.Thread] = None
        # Telemetry: how many input/output drops we've taken (visible to caller).
        self.drops_in = 0
        self.drops_out = 0
        # Pipeline exceptions contained by the worker (dry-block fallbacks).
        self.worker_errors = 0

    @staticmethod
    def _put_drop_oldest(q: queue.Queue, item) -> int:
        """Put ``item`` on ``q``, dropping the oldest entry if full.
        Returns 1 if a drop happened, 0 otherwise."""
        dropped = 0
        if q.full():
            try:
                q.get_nowait()
                dropped = 1
            except queue.Empty:
                pass
        try:
            q.put_nowait(item)
        except queue.Full:
            dropped = 1   # extremely rare race; surface as a drop
        return dropped

    def _worker_loop(self) -> None:
        while self._run:
            try:
                block = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                out = self.pipe.process_block(block).audio
            except Exception:  # noqa: BLE001
                # Containment: one bad block must NOT kill the worker thread —
                # that would leave the stream "alive" playing silence forever.
                # Emit the dry input (passthrough beats silence) and continue.
                self.worker_errors += 1
                if self.worker_errors == 1 or self.worker_errors % 100 == 0:
                    import traceback
                    print(f"voiceiso live: pipeline error #{self.worker_errors} — "
                          f"emitting DRY audio for this block\n{traceback.format_exc()}",
                          flush=True)
                out = block
            self.drops_out += self._put_drop_oldest(self._out_q, out)

    def _callback(self, indata, outdata, frames, time_info, status):  # pragma: no cover
        mono = indata[:, 0].copy()
        # Drop-OLDEST policy: if the worker is behind, discard the *stale* input
        # rather than the *fresh* one, so latency stays bounded.
        self.drops_in += self._put_drop_oldest(self._in_q, mono)
        # Backlog policy: cap at 2, take the oldest remaining — keeps ONE
        # spare block as a jitter cushion (a >100 ms worker spike plays the
        # spare instead of a silence gap) while bounding any post-stall
        # latency ratchet at +100 ms.  See app/audio/engine.py for the
        # red-team simulation behind this choice.
        try:
            while self._out_q.qsize() > 2:
                self._out_q.get_nowait()
                self.drops_out += 1
            y = self._out_q.get_nowait()
        except queue.Empty:
            y = np.zeros(frames, dtype=np.float32)
        if len(y) < frames:
            y = np.concatenate([y, np.zeros(frames - len(y), dtype=np.float32)])
        outdata[:, 0] = y[:frames]
        if outdata.shape[1] > 1:
            outdata[:, 1] = y[:frames]

    def run(self, duration_s: float = 0.0) -> None:  # pragma: no cover
        self.pipe.reset()
        print("voiceiso live: warming up models (~2 s)…", flush=True)
        try:
            self.pipe.warmup()
        except Exception as exc:  # noqa: BLE001 — degrade like the app engine does
            print(f"voiceiso live: warm-up failed ({type(exc).__name__}: {exc}) — "
                  "continuing; first blocks may stutter", flush=True)
        self._run = True
        self.drops_in = 0
        self.drops_out = 0
        self.worker_errors = 0
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        print("voiceiso live:", self.pipe.backend_summary)
        print(f"  block={self.block} samples ({1000.0*self.block/self.cfg.sample_rate:.1f} ms),"
              f" queue_maxsize={self._in_q.maxsize}, latency={self.cfg.live_latency_mode!r}")
        latency_mode = self.cfg.live_latency_mode if hasattr(self.cfg, "live_latency_mode") else "low"
        if self.in_dev is not None or self.out_dev is not None:
            names = sd.query_devices()
            print(f"  devices: in={names[self.in_dev]['name'] if self.in_dev is not None else '(default)'}"
                  f" → out={names[self.out_dev]['name'] if self.out_dev is not None else '(default)'}")
        # Open up to 2 output channels (clamped to the device's capability):
        # a mono stream into a stereo virtual device (BlackHole) would leave
        # channel 2 silent, and a conferencing app's stereo→mono downmix
        # would lose 6 dB.  The callback duplicates mono into both channels.
        out_info = sd.query_devices(self.out_dev, "output")
        out_ch = max(1, min(2, int(out_info["max_output_channels"])))
        with sd.Stream(samplerate=self.cfg.sample_rate, blocksize=self.block,
                       device=(self.in_dev, self.out_dev),
                       channels=(self.cfg.channels, out_ch), dtype="float32",
                       callback=self._callback, latency=latency_mode):
            try:
                if duration_s > 0:
                    time.sleep(duration_s)
                else:
                    while True:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                pass
        self._run = False
        # Wait briefly for the worker to drain so a subsequent run() doesn't
        # race against an old worker still holding pipe references.
        if self._worker is not None:
            self._worker.join(timeout=0.5)
            self._worker = None
        # Surface drop telemetry so users can diagnose latency / xrun issues
        # (the drop-oldest queue policy otherwise makes them invisible).
        if self.drops_in or self.drops_out:
            print(f"voiceiso live: drops_in={self.drops_in} drops_out={self.drops_out}"
                  " (consider increasing live_queue_maxsize or latency='high')")
        if self.worker_errors:
            print(f"voiceiso live: {self.worker_errors} pipeline errors were contained "
                  "(dry audio emitted for those blocks) — see traceback above")

"""
Live audio streaming: microphone → pipeline → speakers.

Uses 20 ms blocks for ~20 ms end-to-end latency.  DFN3's GRU state is maintained
across blocks (stateful streaming), so each call only processes the new 20 ms
chunk — no context buffer overhead.  Because enhancement must not run on the
real-time audio callback thread, audio is shuttled through queues to a worker
thread; the callback only does lock-free enqueue/dequeue.
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


class LiveStream:
    def __init__(self, cfg: Optional[PipelineConfig] = None, block_ms: float = 20.0,
                 enh_threads: int = 4) -> None:
        if sd is None:
            raise ImportError("sounddevice is required for LiveStream")
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
            ctx = self.pipe.process_block(block)
            self.drops_out += self._put_drop_oldest(self._out_q, ctx.audio)

    def _callback(self, indata, outdata, frames, time_info, status):  # pragma: no cover
        mono = indata[:, 0].copy()
        # Drop-OLDEST policy: if the worker is behind, discard the *stale* input
        # rather than the *fresh* one, so latency stays bounded.
        self.drops_in += self._put_drop_oldest(self._in_q, mono)
        # Emit previously-enhanced audio; underflow → silence (warmup/overload).
        try:
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
        self._run = True
        self.drops_in = 0
        self.drops_out = 0
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        print("voiceiso live:", self.pipe.backend_summary)
        print(f"  block={self.block} samples ({1000.0*self.block/self.cfg.sample_rate:.1f} ms),"
              f" queue_maxsize={self._in_q.maxsize}, latency={self.cfg.live_latency_mode!r}")
        latency_mode = self.cfg.live_latency_mode if hasattr(self.cfg, "live_latency_mode") else "low"
        with sd.Stream(samplerate=self.cfg.sample_rate, blocksize=self.block,
                       channels=self.cfg.channels, dtype="float32",
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

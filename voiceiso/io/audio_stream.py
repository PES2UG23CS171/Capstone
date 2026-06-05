"""
Live audio streaming: microphone → pipeline → speakers.

Uses a block size matched to the enhancement core (~100 ms) so DFN runs at its
efficient operating point.  Because the heavy enhancement must not run on the
real-time audio callback thread (it would xrun), audio is shuttled through
queues to a worker thread that runs the pipeline — the callback only does
lock-free enqueue/dequeue.
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
    def __init__(self, cfg: Optional[PipelineConfig] = None, block_ms: float = 100.0,
                 enh_threads: int = 4) -> None:
        if sd is None:
            raise ImportError("sounddevice is required for LiveStream")
        self.cfg = cfg or PipelineConfig()
        self.block = int(self.cfg.sample_rate * block_ms / 1000.0)
        self.pipe = StreamingPipeline(self.cfg, enh_threads=enh_threads)
        self._in_q: queue.Queue = queue.Queue(maxsize=16)
        self._out_q: queue.Queue = queue.Queue(maxsize=16)
        self._run = False
        self._worker: Optional[threading.Thread] = None
        self._tail = np.zeros(0, dtype=np.float32)

    def _worker_loop(self) -> None:
        while self._run:
            try:
                block = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            ctx = self.pipe.process_block(block)
            try:
                self._out_q.put_nowait(ctx.audio)
            except queue.Full:
                pass  # drop under overload rather than stall the callback

    def _callback(self, indata, outdata, frames, time_info, status):  # pragma: no cover
        mono = indata[:, 0].copy()
        try:
            self._in_q.put_nowait(mono)
        except queue.Full:
            pass
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
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        print("voiceiso live:", self.pipe.backend_summary)
        with sd.Stream(samplerate=self.cfg.sample_rate, blocksize=self.block,
                       channels=self.cfg.channels, dtype="float32",
                       callback=self._callback, latency="high"):
            try:
                if duration_s > 0:
                    time.sleep(duration_s)
                else:
                    while True:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                pass
        self._run = False

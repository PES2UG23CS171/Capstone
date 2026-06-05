"""
Acoustic Echo Cancellation (AEC).

Should it be included?  **Yes, conditionally.**  AEC removes the far-end signal
(what the *other* participant says, played out your speakers) from your mic so
it isn't sent back as echo.  It only applies when you have a **reference**
signal — a loopback of the audio being played.  On headphones echo is minimal;
on open speakers it is essential.  Because capturing system loopback is
platform-specific, this stage is **opt-in** (``cfg.aec_enabled`` + a provided
``ctx.reference``) and is a no-op otherwise.

Architecture: **Partitioned-Block Frequency-Domain Adaptive Filter (PBFDAF)**,
constrained NLMS update — the same family used by WebRTC AEC3 and Speex.
  * Streaming, overlap-save, O(N log N) per block — far cheaper than time-domain
    NLMS for the long tails (100–250 ms) real rooms need.
  * Partitioning lets a long echo tail be covered with small FFTs (good cache
    behaviour, low latency).

Latency impact: one AEC block (~10–20 ms).  CPU cost: a few FFTs/block —
typically <5 % of one core.  Expected gain: 20–35 dB ERLE on stationary echo
paths, removing the dominant failure mode for speakerphone use.

Note: this is a linear AEC.  Residual-echo suppression (the non-linear tail) is
handled downstream by the enhancement + post-filter, which is exactly how
commercial stacks layer it.
"""

from __future__ import annotations

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage


class AEC(Stage):
    name = "aec"

    def __init__(self, cfg: PipelineConfig, block: int = 512, mu: float = 0.3) -> None:
        self.cfg = cfg
        self.enabled = cfg.aec_enabled
        self.N = block
        self.fft = 2 * block
        self.mu = mu
        # Partitions needed to cover the configured tail length.
        tail = int(cfg.sample_rate * cfg.aec_filter_ms / 1000.0)
        self.P = max(1, int(np.ceil(tail / block)))

        self._W = np.zeros((self.P, self.fft // 2 + 1), dtype=np.complex128)
        self._Xp = np.zeros((self.P, self.fft // 2 + 1), dtype=np.complex128)
        self._x_prev = np.zeros(block, dtype=np.float64)   # overlap-save history
        self._in_near = np.zeros(0, dtype=np.float64)
        self._in_ref = np.zeros(0, dtype=np.float64)
        self._out = np.zeros(0, dtype=np.float64)
        self._erle = 0.0

    def reset(self) -> None:
        self._W[:] = 0
        self._Xp[:] = 0
        self._x_prev[:] = 0
        self._in_near = np.zeros(0)
        self._in_ref = np.zeros(0)
        self._out = np.zeros(0)

    def _process_block(self, d: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Cancel echo from one N-sample near-end block ``d`` given far-end ``x``."""
        frame = np.concatenate([self._x_prev, x])          # 2N overlap-save
        X = np.fft.rfft(frame)
        self._Xp = np.roll(self._Xp, 1, axis=0)
        self._Xp[0] = X

        Y = np.sum(self._W * self._Xp, axis=0)
        y = np.fft.irfft(Y)[self.N:]                       # echo estimate (last N)
        e = d - y                                          # error = cleaned near-end

        # Constrained NLMS update in the frequency domain.
        E = np.fft.rfft(np.concatenate([np.zeros(self.N), e]))
        power = np.sum(np.abs(self._Xp) ** 2, axis=0) + 1e-6
        for p in range(self.P):
            G = np.conj(self._Xp[p]) * E / power
            g = np.fft.irfft(G)
            g[self.N:] = 0.0                               # gradient constraint
            self._W[p] += self.mu * np.fft.rfft(g)

        self._x_prev = x
        # Track ERLE (echo return loss enhancement) for diagnostics.
        de = float(np.mean(d * d)) + 1e-12
        ee = float(np.mean(e * e)) + 1e-12
        self._erle = 0.95 * self._erle + 0.05 * (10.0 * np.log10(de / ee))
        return e

    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.reference is None:
            return ctx
        self._in_near = np.concatenate([self._in_near, ctx.audio.astype(np.float64)])
        self._in_ref = np.concatenate([self._in_ref, ctx.reference.astype(np.float64)])

        while len(self._in_near) >= self.N and len(self._in_ref) >= self.N:
            d = self._in_near[: self.N]; self._in_near = self._in_near[self.N:]
            x = self._in_ref[: self.N];  self._in_ref = self._in_ref[self.N:]
            self._out = np.concatenate([self._out, self._process_block(d, x)])

        # Emit the same number of samples we were given (delayed by ≤1 block).
        n = len(ctx.audio)
        if len(self._out) >= n:
            out = self._out[:n]; self._out = self._out[n:]
            ctx.audio = out.astype("float32")
        ctx.meta["erle_db"] = self._erle
        return ctx

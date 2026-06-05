"""
Acoustic Echo Cancellation (AEC) — V2.

V1 was a single-shot partitioned-block FDAF (PBFDAF) NLMS.  V2 closes the
production-grade gaps:

1. **Geigel double-talk detector (DTD).**  Cheap, robust: declares double-talk
   whenever ``max(|d|) > T * max(|x_history|)``.  During double-talk the NLMS
   gradient is dominated by the near-end signal and the adaptive filter
   *diverges* — V1 updated unconditionally, which is the AEC version of a
   landmine.  V2 freezes ``W`` while DT is active and for ~1 s afterwards (to
   ride out the trailing instability).

2. **Reduced step size.**  ``mu = 0.15`` (was 0.3); converges a bit slower but
   far more stable under bursty far-end signals.  The DTD does the rest.

3. **Per-band residual-echo coherence.**  After cancellation, we compute the
   coherence between the residual ``e`` and the delayed far-end ``x`` in 3
   bands (low/mid/high).  Bands with high coherence indicate the linear AEC
   missed something — the controller uses this to raise post-DFN per-band
   attenuation (residual-echo suppression, RES).

4. **State surfaced into FrameContext** (``dt_active``, ``res_band``) so the
   controller can see and react.  Diagnostics still go to ``ctx.meta``.

Note: this is still a linear AEC.  Non-linear residual echo (speaker
distortion, late reverberation) is handled jointly by DFN3 + post-filter,
informed by the RES coherence flags.
"""

from __future__ import annotations

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage


class _GeigelDTD:
    """Geigel double-talk detector.

    Declares DT when ``max(|d|) > T * max(|x[-D:]|)``.  T ~ 0.5 is standard;
    we keep ~100 ms of far-end history so transient far-end pauses do not
    immediately drop the DT flag.
    """

    def __init__(self, sr: int, history_ms: float = 100.0, threshold: float = 0.5) -> None:
        self.D = int(sr * history_ms / 1000.0)
        self.T = threshold
        self._x_hist = np.zeros(self.D, dtype=np.float64)

    def reset(self) -> None:
        self._x_hist[:] = 0.0

    def __call__(self, d: np.ndarray, x: np.ndarray) -> bool:
        n = len(x)
        if n >= self.D:
            self._x_hist = x[-self.D:].astype(np.float64).copy()
        else:
            self._x_hist = np.roll(self._x_hist, -n)
            self._x_hist[-n:] = x.astype(np.float64)
        max_d = float(np.max(np.abs(d))) if len(d) else 0.0
        max_x = float(np.max(np.abs(self._x_hist))) + 1e-9
        return max_d > self.T * max_x


def _band_coherence(e: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Approximate magnitude-squared coherence between residual ``e`` and far-end
    ``x`` in 3 bands (low / mid / high).  Returns scalars in [0, 1] — higher
    means the linear AEC has left correlated echo in that band.
    """
    n = min(len(e), len(x))
    if n < 32:
        return 0.0, 0.0, 0.0
    e = e[:n].astype(np.float64)
    x = x[:n].astype(np.float64)
    E = np.fft.rfft(e * np.hanning(n))
    X = np.fft.rfft(x * np.hanning(n))
    eps = 1e-12
    coh = (np.abs(E) * np.abs(X)) / (np.sqrt(np.abs(E) ** 2 + eps) *
                                      np.sqrt(np.abs(X) ** 2 + eps) + eps)
    # Three equal-bin bands across the spectrum.
    nb = len(coh) // 3
    if nb == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(coh[:nb])),
        float(np.mean(coh[nb:2 * nb])),
        float(np.mean(coh[2 * nb:])),
    )


class AEC(Stage):
    name = "aec"

    def __init__(self, cfg: PipelineConfig, block: int = 512,
                 mu: float | None = None) -> None:
        self.cfg = cfg
        self.enabled = cfg.aec_enabled
        self.N = block
        self.fft = 2 * block
        # V2: use the configured (lower) step-size; legacy callers can pass mu.
        self.mu_base = float(cfg.aec_step_size) if mu is None else float(mu)
        # Partitions needed to cover the configured tail length.
        tail = int(cfg.sample_rate * cfg.aec_filter_ms / 1000.0)
        self.P = max(1, int(np.ceil(tail / block)))

        self._W = np.zeros((self.P, self.fft // 2 + 1), dtype=np.complex128)
        self._Xp = np.zeros((self.P, self.fft // 2 + 1), dtype=np.complex128)
        self._x_prev = np.zeros(block, dtype=np.float64)   # overlap-save history
        self._in_near = np.zeros(0, dtype=np.float64)
        self._in_ref = np.zeros(0, dtype=np.float64)
        self._out = np.zeros(0, dtype=np.float64)
        self._out_x = np.zeros(0, dtype=np.float64)        # delayed far-end (for RES)
        self._erle = 0.0
        # Double-talk detector + freeze countdown (blocks).
        self._dtd = _GeigelDTD(cfg.sample_rate, threshold=cfg.aec_dtd_threshold)
        # Convert dt_freeze_ms → blocks at the AEC block rate.
        block_ms = 1000.0 * block / cfg.sample_rate
        self._dt_hold_blocks = max(1, int(cfg.aec_dt_freeze_ms / max(block_ms, 1e-3)))
        self._dt_remaining = 0
        # Latest per-band residual-echo coherence for context.
        self._last_res = (0.0, 0.0, 0.0)
        self._last_dt = False

    def reset(self) -> None:
        self._W[:] = 0
        self._Xp[:] = 0
        self._x_prev[:] = 0
        self._in_near = np.zeros(0)
        self._in_ref = np.zeros(0)
        self._out = np.zeros(0)
        self._out_x = np.zeros(0)
        self._dtd.reset()
        self._dt_remaining = 0
        self._last_res = (0.0, 0.0, 0.0)
        self._last_dt = False

    def _process_block(self, d: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Cancel echo from one N-sample near-end block ``d`` given far-end ``x``."""
        # Geigel DTD: is the near-end clearly louder than the far-end history?
        dt = self._dtd(d, x)
        if dt:
            self._dt_remaining = self._dt_hold_blocks
        elif self._dt_remaining > 0:
            self._dt_remaining -= 1
        self._last_dt = dt or self._dt_remaining > 0

        frame = np.concatenate([self._x_prev, x])          # 2N overlap-save
        X = np.fft.rfft(frame)
        self._Xp = np.roll(self._Xp, 1, axis=0)
        self._Xp[0] = X

        Y = np.sum(self._W * self._Xp, axis=0)
        y = np.fft.irfft(Y)[self.N:]                       # echo estimate (last N)
        e = d - y                                          # error = cleaned near-end

        # Constrained NLMS update — **freeze during double-talk** so the filter
        # does not diverge while near-end energy dominates the gradient.
        mu_eff = self.mu_base if not self._last_dt else 0.0
        if mu_eff > 0.0:
            E = np.fft.rfft(np.concatenate([np.zeros(self.N), e]))
            power = np.sum(np.abs(self._Xp) ** 2, axis=0) + 1e-6
            for p in range(self.P):
                G = np.conj(self._Xp[p]) * E / power
                g = np.fft.irfft(G)
                g[self.N:] = 0.0                           # gradient constraint
                self._W[p] += mu_eff * np.fft.rfft(g)

        self._x_prev = x
        # Track ERLE (echo return loss enhancement) for diagnostics.
        de = float(np.mean(d * d)) + 1e-12
        ee = float(np.mean(e * e)) + 1e-12
        self._erle = 0.95 * self._erle + 0.05 * (10.0 * np.log10(de / ee))
        # Per-band residual-echo coherence (delayed far-end vs residual).
        self._last_res = _band_coherence(e, x)
        return e

    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.reference is None:
            ctx.dt_active = False
            ctx.res_band = (0.0, 0.0, 0.0)
            return ctx
        self._in_near = np.concatenate([self._in_near, ctx.audio.astype(np.float64)])
        self._in_ref = np.concatenate([self._in_ref, ctx.reference.astype(np.float64)])

        while len(self._in_near) >= self.N and len(self._in_ref) >= self.N:
            d = self._in_near[: self.N]; self._in_near = self._in_near[self.N:]
            x = self._in_ref[: self.N];  self._in_ref = self._in_ref[self.N:]
            self._out = np.concatenate([self._out, self._process_block(d, x)])

        # Emit the same number of samples we were given.  Warmup (first call
        # whose input is larger than the AEC block) may not yet have N output
        # samples ready — in that case zero-pad up to ``n`` so downstream
        # stages see deterministic timing and never receive un-AEC'd audio.
        # V1 left ctx.audio as the raw mic input here, silently leaking echo
        # into the rest of the pipeline.
        n = len(ctx.audio)
        if len(self._out) >= n:
            out = self._out[:n]; self._out = self._out[n:]
            ctx.audio = out.astype("float32")
        else:
            # Not enough output yet: emit what we have, pad the rest with zero.
            ready = len(self._out)
            out = np.concatenate([self._out, np.zeros(n - ready, dtype=np.float64)])
            self._out = np.zeros(0, dtype=np.float64)
            ctx.audio = out.astype("float32")
            ctx.meta["aec_warmup"] = 1.0
        ctx.meta["erle_db"] = self._erle
        ctx.meta["aec_dt"] = 1.0 if self._last_dt else 0.0
        ctx.meta["res_lo"] = self._last_res[0]
        ctx.meta["res_md"] = self._last_res[1]
        ctx.meta["res_hi"] = self._last_res[2]
        ctx.dt_active = self._last_dt
        ctx.res_band = self._last_res
        return ctx

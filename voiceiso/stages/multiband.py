"""
MultiBandGainModulator — post-DFN per-band gain control.

DFN3 exposes only a single ``atten_lim_db`` cap that applies uniformly across
the spectrum.  But real-world noise is band-localised: fan / HVAC dominate
0–300 Hz, keyboard energy concentrates at 2–8 kHz, dog barks at ~500 Hz–4 kHz.
A single global cap either over-suppresses speech where the noise isn't, or
under-suppresses where it is.  Apple Voice Isolation and Krisp both use
per-band gain control.

Filterbank design — 3-band perfect-reconstruction
-------------------------------------------------
::

    y_lo  = lowpass(x,  f_lo)                          # < 300 Hz
    y_hi  = highpass(x, f_hi)                          # > 4000 Hz
    x_dly = delay_line(x, (n_taps - 1) // 2)           # match FIR delay
    y_md  = x_dly - y_lo - y_hi                        # mid by SUBTRACTION

At unity gain ``g_lo·y_lo + g_md·y_md + g_hi·y_hi == x_dly`` exactly — no
destructive interference at band edges, no coloration.  This is the classic
complementary-filter trick used in audio crossovers.

Always-on, constant delay, per-sample gain ramps
------------------------------------------------
The stage runs its filterbank on EVERY block and emits a constant
``(n_taps−1)/2`` = 15-sample (0.31 ms @ 48 kHz) group delay.  An earlier
design bypassed (zero-delay) at unity gain and spliced into the delayed path
when gains went active — each toggle inserted/removed 15 samples of timeline
and re-primed the FIR state with zeros (an audible click), and the resulting
variable delay corrupted SI-SDR whenever the classifier fired.  The benchmark
now time-aligns output to reference before scoring (bench C4 fix), so the
honest constant delay costs nothing, and the always-on filterbank keeps FIR
state warm so there are no mode transitions at all.

Gains are additionally interpolated PER SAMPLE from the previous block's
values to the current ones, so even a large controller step (already
attack/release-smoothed upstream) glides over 100 ms instead of stepping at a
block edge (zipper noise).

CPU
---
* ~0.1 ms / block (2 × 31-tap FIR @ 4800 samples + mixing)
* Latency added: constant 15 samples ≈ 0.31 ms
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage


class MultiBandGainModulator(Stage):
    name = "multiband"

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.sr = cfg.sample_rate
        n_taps = int(cfg.band_fir_taps)
        if n_taps % 2 == 0:
            n_taps += 1                       # odd length → integer group delay
        self.n_taps = n_taps
        self._group_delay = (n_taps - 1) // 2
        nyq = self.sr / 2.0
        f_lo = cfg.band_low_hz / nyq
        f_hi = cfg.band_high_hz / nyq

        self._fir_lo = firwin(n_taps, f_lo, pass_zero=True, window="hamming")
        self._fir_hi = firwin(n_taps, f_hi, pass_zero=False, window="hamming")

        self._zi_lo = np.zeros(n_taps - 1, dtype=np.float64)
        self._zi_hi = np.zeros(n_taps - 1, dtype=np.float64)
        self._delay_buf = np.zeros(self._group_delay, dtype=np.float64)
        # Previous block's applied gains — start point of this block's ramp.
        self._g_prev = np.ones(3, dtype=np.float64)

    def reset(self) -> None:
        self._zi_lo[:] = 0.0
        self._zi_hi[:] = 0.0
        self._delay_buf[:] = 0.0
        self._g_prev[:] = 1.0

    def _delay(self, x: np.ndarray) -> np.ndarray:
        """Delay ``x`` by ``self._group_delay`` samples with state across blocks."""
        d = self._group_delay
        if d == 0:
            return x.copy()
        n = len(x)
        out = np.empty(n, dtype=np.float64)
        if n >= d:
            out[:d] = self._delay_buf
            out[d:] = x[: n - d]
            self._delay_buf = x[n - d :].astype(np.float64).copy()
        else:
            out[:] = self._delay_buf[:n]
            self._delay_buf = np.concatenate(
                [self._delay_buf[n:], x.astype(np.float64)]
            )
        return out

    def process(self, ctx: FrameContext) -> FrameContext:
        # Gain targets: the controller's (already attack/release-smoothed)
        # band gains — forced to unity when DFN3 didn't run on this frame
        # (the multi-band's job is cleaning up *residuals* DFN3 left behind;
        # attenuating the raw noisy signal just darkens it).
        if float(ctx.meta.get("enh_wet", 1.0)) < 0.5:
            g_t = np.ones(3, dtype=np.float64)
        else:
            g_t = np.asarray(ctx.band_gain, dtype=np.float64)

        # Filterbank runs unconditionally: constant 15-sample delay, warm FIR
        # state, no bypass↔active splices (see module docstring).
        x = ctx.audio.astype(np.float64)
        n = len(x)
        y_lo, self._zi_lo = lfilter(self._fir_lo, [1.0], x, zi=self._zi_lo)
        y_hi, self._zi_hi = lfilter(self._fir_hi, [1.0], x, zi=self._zi_hi)
        x_dly = self._delay(x)
        y_md = x_dly - y_lo - y_hi                       # perfect-reconstruction residual

        if np.allclose(g_t, 1.0, atol=1e-6) and np.allclose(self._g_prev, 1.0, atol=1e-6):
            # Unity throughout the block: PR identity says the mix is exactly
            # x_dly — skip the ramp math (bit-exact, saves a few multiplies).
            y = x_dly
        else:
            # Per-sample linear ramp from last block's gains to this block's.
            ramp = np.linspace(0.0, 1.0, n, endpoint=True)[:, None]
            g = self._g_prev[None, :] * (1.0 - ramp) + g_t[None, :] * ramp
            y = g[:, 0] * y_lo + g[:, 1] * y_md + g[:, 2] * y_hi
        self._g_prev = g_t.copy()

        ctx.audio = y.astype(np.float32)
        ctx.meta["mb_gain_lo"] = float(g_t[0])
        ctx.meta["mb_gain_md"] = float(g_t[1])
        ctx.meta["mb_gain_hi"] = float(g_t[2])
        ctx.meta["mb_active"] = float(not np.allclose(g_t, 1.0, atol=1e-6))
        return ctx

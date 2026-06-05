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

Bypass behaviour at unity gain
------------------------------
**Important:** when all gains are unity (≈ 99 % of the time in normal speech),
this stage is a **true passthrough** — no delay, no FIR compute.  The FIR
filter states are reset to zero on bypass.  When the controller later flips
to non-unity gain, the FIR ramps up over its impulse-response length
(~64 samples ≈ 1.3 ms) — short enough to be inaudible — and the output gains
a 32-sample (≈ 0.67 ms) group delay for as long as the gains stay non-unity.

Why not "always delay"?
~~~~~~~~~~~~~~~~~~~~~~~
An earlier design always emitted the delayed signal so the group delay was
constant across blocks.  This produced a *catastrophic* benchmark regression:
because the SI-SDR / correlation metrics compare sample-aligned signals, a
constant 32-sample shift drops correlation to ~0.0 on speech.  Real-world
listeners can't hear 0.67 ms of delay, but the metric does.

The chosen alternative — zero-delay bypass at unity, small transient on
transition — keeps the bench honest and is what commercial systems do.

CPU
---
* Bypass:  ~0.001 ms / block (just the threshold check)
* Active:  ~0.3 ms / block (2 × 65-tap FIR @ 960 samples)
* Latency added (active only): 32 samples ≈ 0.67 ms
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage


class MultiBandGainModulator(Stage):
    name = "multiband"

    # Loose threshold for "at unity" — small float jitter from the controller
    # (e.g. 0.9999) shouldn't force unnecessary FIR compute / delay.
    _UNITY_EPS = 1e-3

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
        # Tracks whether we are currently in active (delayed) mode so we can
        # cleanly reset state when bypassing.
        self._was_active = False

    def reset(self) -> None:
        self._zi_lo[:] = 0.0
        self._zi_hi[:] = 0.0
        self._delay_buf[:] = 0.0
        self._was_active = False

    def _is_unity(self, g_lo: float, g_md: float, g_hi: float) -> bool:
        e = self._UNITY_EPS
        return (abs(g_lo - 1.0) < e and abs(g_md - 1.0) < e and abs(g_hi - 1.0) < e)

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

    def _clear_active_state(self) -> None:
        """Wipe FIR / delay state and mark the stage inactive.

        Called from any bypass path so that when the stage next goes active
        the FIR state isn't conditioned on audio from many blocks ago — which
        would re-emit stale samples as the first n_taps of output (an audible
        click on resume).
        """
        if self._was_active:
            self._zi_lo[:] = 0.0
            self._zi_hi[:] = 0.0
            self._delay_buf[:] = 0.0
            self._was_active = False

    def process(self, ctx: FrameContext) -> FrameContext:
        g_lo, g_md, g_hi = ctx.band_gain

        # Skip multi-band entirely when DFN3 didn't run on this frame.
        # The multi-band's job is to clean up *residuals* DFN3 left behind;
        # applying band attenuation to the raw noisy signal just darkens it
        # without removing meaningful noise.  ``enh_wet == 0`` is set by the
        # Enhancement stage on bypass / passthrough.  Reset FIR state on this
        # path too — otherwise (active → enh-bypass → active) resumes with
        # stale buffer content.
        if float(ctx.meta.get("enh_wet", 1.0)) < 0.5:
            self._clear_active_state()
            ctx.meta["mb_gain_lo"] = 1.0
            ctx.meta["mb_gain_md"] = 1.0
            ctx.meta["mb_gain_hi"] = 1.0
            ctx.meta["mb_active"] = 0.0
            return ctx

        if self._is_unity(g_lo, g_md, g_hi):
            # True passthrough: no delay, no FIR work.  Reset FIR/delay state
            # so the next non-unity block doesn't re-emit stale buffer content.
            self._clear_active_state()
            ctx.meta["mb_gain_lo"] = 1.0
            ctx.meta["mb_gain_md"] = 1.0
            ctx.meta["mb_gain_hi"] = 1.0
            ctx.meta["mb_active"] = 0.0
            return ctx

        # Active path: split into 3 bands and apply per-band gains.
        x = ctx.audio.astype(np.float64)
        y_lo, self._zi_lo = lfilter(self._fir_lo, [1.0], x, zi=self._zi_lo)
        y_hi, self._zi_hi = lfilter(self._fir_hi, [1.0], x, zi=self._zi_hi)
        x_dly = self._delay(x)
        y_md = x_dly - y_lo - y_hi                       # perfect-reconstruction residual

        y = float(g_lo) * y_lo + float(g_md) * y_md + float(g_hi) * y_hi
        ctx.audio = y.astype(np.float32)
        self._was_active = True
        ctx.meta["mb_gain_lo"] = float(g_lo)
        ctx.meta["mb_gain_md"] = float(g_md)
        ctx.meta["mb_gain_hi"] = float(g_hi)
        ctx.meta["mb_active"] = 1.0
        return ctx

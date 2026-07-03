"""
Preprocessing stage — conditions the raw mic signal before enhancement.

Operations (all streaming, stateful across frames):
  1. DC-offset removal (one-pole high-pass at ~20 Hz).
  2. Running noise-floor + SNR estimate (minimum-statistics) used downstream by
     the dynamic controller.

Why this matters: commercial systems never feed the raw ADC stream straight
into a neural net — DC bias degrades the model.  Note the high-pass is kept
deliberately gentle (1st-order, ~25 Hz): a steeper rumble filter was measured to
cost ~8 dB SNR (it eats low speech harmonics and adds group delay), and DFN3
already removes low-frequency noise far better — so we only block DC/subsonic
here and let the network do the rest.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage


class Preprocessing(Stage):
    name = "preprocessing"

    def __init__(self, cfg: PipelineConfig, highpass_hz: float = 25.0) -> None:
        self.cfg = cfg
        self.sr = cfg.sample_rate
        # Two pre-built 1st-order Butterworth high-pass filters:
        #   * default  — 25 Hz (transparent to speech)
        #   * wind     — 80 Hz (configurable via cfg.wind_hp_cutoff_hz)
        # Selection is driven by the *previous frame's* noise_class, which
        # the pipeline writes back to ``self._prev_noise_class`` after each
        # block.  One-frame lag is acceptable.
        self._sos_default = butter(1, highpass_hz / (self.sr / 2.0), btype="high", output="sos")
        self._sos_wind = butter(
            1, cfg.wind_hp_cutoff_hz / (self.sr / 2.0), btype="high", output="sos"
        )
        # Active filter and its persistent state.
        self._active_mode = "default"
        self._sos = self._sos_default
        self._zi = np.zeros_like(sosfilt_zi(self._sos)).astype(np.float64)
        # Previous-frame class hint — updated by the pipeline (see set_prev_noise_class).
        self._prev_noise_class = "clean"
        # Hysteresis: count consecutive frames that agree with the *opposite*
        # of the current mode.  Only switch once the count exceeds threshold.
        # Prevents periodic state-reset thumps when the classifier flutters
        # between wind and traffic in mixed scenes.
        self._switch_streak = 0
        self._switch_threshold = 3   # frames; ~30 ms at 10 ms hops, ~60 ms at 20 ms blocks
        # Minimum-statistics noise floor (power) + speech-power tracker.
        self._noise_pow = 1e-6
        self._sig_pow = 1e-6
        self._floor_hist: list[float] = []

    def reset(self) -> None:
        self._active_mode = "default"
        self._sos = self._sos_default
        self._zi = np.zeros_like(sosfilt_zi(self._sos)).astype(np.float64)
        self._prev_noise_class = "clean"
        self._switch_streak = 0
        self._noise_pow = 1e-6
        self._sig_pow = 1e-6
        self._floor_hist.clear()

    def set_prev_noise_class(self, noise_class: str) -> None:
        """Called by the pipeline after each block.  The next call to
        :meth:`process` uses ``noise_class`` to decide which HP cutoff to apply."""
        self._prev_noise_class = noise_class

    def _select_filter(self) -> None:
        """Switch between default and wind filters based on prev_noise_class,
        with hysteresis to avoid flipping on classifier flutter.

        Requires ``_switch_threshold`` consecutive frames of disagreement
        with the current mode before actually flipping.  Without this, a
        learned classifier whose smoothed posteriors hover near a tie
        between two classes (e.g. wind vs traffic) can ping-pong the HP
        cutoff, producing an audible thump on every switch from the filter
        state-reset.
        """
        want = "wind" if self._prev_noise_class == "wind" else "default"
        if want == self._active_mode:
            # Currently agreeing → reset the streak.
            self._switch_streak = 0
            return
        # Disagreement: count consecutive frames before switching.
        self._switch_streak += 1
        if self._switch_streak < self._switch_threshold:
            return
        # Committed switch — reset state and counter.
        self._active_mode = want
        self._sos = self._sos_wind if want == "wind" else self._sos_default
        self._zi = np.zeros_like(sosfilt_zi(self._sos)).astype(np.float64)
        self._switch_streak = 0

    def process(self, ctx: FrameContext) -> FrameContext:
        # Update HP cutoff based on prev-frame class hint.
        self._select_filter()
        x = ctx.audio.astype(np.float64)
        y, self._zi = sosfilt(self._sos, x, zi=self._zi)
        y = y.astype(np.float32)
        ctx.meta["hp_mode"] = 1.0 if self._active_mode == "wind" else 0.0

        # Frame power → smoothed signal power + running noise floor.
        p = float(np.mean(y * y) + 1e-12)
        self._sig_pow = 0.9 * self._sig_pow + 0.1 * p
        # Track a slow minimum as the noise floor (robust to speech bursts).
        self._floor_hist.append(p)
        if len(self._floor_hist) > 50:           # ~0.5 s window @ 10 ms hops
            self._floor_hist.pop(0)
        self._noise_pow = max(min(self._floor_hist), 1e-10)

        snr = 10.0 * np.log10(self._sig_pow / self._noise_pow)
        ctx.audio = y
        ctx.snr_db = float(np.clip(snr, -10.0, 60.0))
        ctx.meta["noise_floor_db"] = float(10.0 * np.log10(self._noise_pow))
        return ctx

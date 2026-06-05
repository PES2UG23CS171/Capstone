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
        # Gentle 1st-order DC/subsonic blocker — transparent to speech.
        self._sos = butter(1, highpass_hz / (self.sr / 2.0), btype="high", output="sos")
        self._zi = sosfilt_zi(self._sos).astype(np.float64)
        # Minimum-statistics noise floor (power) + speech-power tracker.
        self._noise_pow = 1e-6
        self._sig_pow = 1e-6
        self._floor_hist: list[float] = []

    def reset(self) -> None:
        self._zi = sosfilt_zi(self._sos).astype(np.float64)
        self._noise_pow = 1e-6
        self._sig_pow = 1e-6
        self._floor_hist.clear()

    def process(self, ctx: FrameContext) -> FrameContext:
        x = ctx.audio.astype(np.float64)
        y, self._zi = sosfilt(self._sos, x, zi=self._zi)
        y = y.astype(np.float32)

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

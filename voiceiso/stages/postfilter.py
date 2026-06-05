"""
Residual post-filter + comfort noise (final cleanup stage).

After enhancement there are two perceptual problems left:
  1. **Residual hiss / musical noise** in noise-only gaps.
  2. **Dead silence** — fully gated gaps sound unnatural and make listeners
     think the call dropped.  Commercial products inject low-level *comfort
     noise* to keep the background "alive".

This stage, steered by ``ctx.postfilter_strength`` from the controller:
  * applies extra residual attenuation during non-speech frames (safe — no
    speech to damage), and
  * injects spectrally-shaped comfort noise at ``cfg.comfort_noise_db`` so the
    floor never collapses to digital zero.

Cheap (time-domain), runs after the wet/dry mix.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage


class PostFilter(Stage):
    name = "postfilter"

    # One-pole low-pass for comfort-noise shaping:  y = 0.05·x + 0.95·y[-1]
    _CN_B = np.array([0.05], dtype="float64")
    _CN_A = np.array([1.0, -0.95], dtype="float64")

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._cn_lin = 10.0 ** (cfg.comfort_noise_db / 20.0)
        self._floor_lin = 10.0 ** (cfg.postfilter_floor_db / 20.0)
        self._cn_zi = np.zeros(1, dtype="float64")   # lfilter state across blocks
        self._rng = np.random.default_rng(1234)

    def reset(self) -> None:
        self._cn_zi = np.zeros(1, dtype="float64")

    def process(self, ctx: FrameContext) -> FrameContext:
        x = ctx.audio
        n = len(x)

        # 1. Extra residual attenuation in non-speech frames only.
        if not ctx.is_speech and ctx.postfilter_strength > 0.0:
            # Interpolate gain between 1.0 and the floor by post-filter strength.
            g = 1.0 - ctx.postfilter_strength * (1.0 - self._floor_lin)
            x = x * np.float32(g)

        # 2. Comfort noise: pink-ish (one-pole low-passed white) at a fixed low
        #    level, so gaps keep a natural, constant ambience.
        white = self._rng.standard_normal(n)
        cn, self._cn_zi = lfilter(self._CN_B, self._CN_A, white, zi=self._cn_zi)
        cn = cn.astype("float32")
        cn *= np.float32(self._cn_lin / (np.sqrt(np.mean(cn * cn)) + 1e-9))

        ctx.audio = np.clip(x + cn, -1.0, 1.0).astype("float32")
        return ctx

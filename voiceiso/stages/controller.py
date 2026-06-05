"""
Dynamic Suppression Controller (quality-first).

This is the heart of the "integrated system" novelty.  An earlier version tried
to be clever — lowering DFN's strength to "protect" speech and per-class
suppression targets.  Empirically that *hurt*: DFN preserves speech at corr 0.99
and gives +16 dB SNR at full strength, so throttling it in noise only leaks
noise back.  The corrected policy is therefore **quality-first**:

* **Default to full enhancement** whenever any noise/speech-in-noise is present.
  DFN is trusted; we do not blend dry noise back.
* **Bypass only when confidently, *sustainedly* clean** (high SNR + "clean"
  class for several consecutive frames).  This saves CPU and avoids processing
  already-clean audio — without risking noise leakage from a single
  misclassified frame.
* **Per-class nuance drives the post-filter, not DFN's strength.**  The noise
  class and SNR set ``postfilter_strength`` (residual cleanup + comfort-noise
  behaviour), where being wrong is harmless, instead of throttling the network.

Inputs: VAD confidence, noise class, SNR, environment.
Outputs: ``ctx.suppression`` (→ DFN atten cap / bypass) and
``ctx.postfilter_strength``.

Transparent + CPU-free; swappable for a learned policy behind the same API.
"""

from __future__ import annotations

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage

# Per-class POST-FILTER weight (residual cleanup aggressiveness in gaps).
# Higher for noise types whose residual is most annoying (tonal/speech-like).
_CLASS_POSTFILTER = {
    "clean": 0.0, "fan": 0.4, "hvac": 0.4, "traffic": 0.6, "wind": 0.7,
    "keyboard": 0.5, "mouse_click": 0.5, "dog_bark": 0.6, "door_slam": 0.6,
    "music": 0.8, "television": 0.8, "competing_speech": 0.9,
}


class DynamicController(Stage):
    name = "controller"

    def __init__(self, cfg: PipelineConfig, clean_snr_db: float = 22.0,
                 clean_hold_ms: float = 400.0) -> None:
        self.cfg = cfg
        self._supp = cfg.supp_max
        self._a_attack = 1.0 - np.exp(-cfg.hop_ms / max(cfg.supp_attack_ms, 1e-3))
        self._a_release = 1.0 - np.exp(-cfg.hop_ms / max(cfg.supp_release_ms, 1e-3))
        self._clean_snr = clean_snr_db
        self._clean_run = 0
        # Frames of sustained-clean required before we trust a bypass.
        self._clean_needed = max(1, int(clean_hold_ms / max(cfg.hop_ms, 1.0)))

    def reset(self) -> None:
        self._supp = self.cfg.supp_max
        self._clean_run = 0

    def process(self, ctx: FrameContext) -> FrameContext:
        cfg = self.cfg

        # Sustained-clean detector: high SNR, no speech, "clean" class.
        confidently_clean = (
            (not ctx.is_speech)
            and ctx.snr_db >= self._clean_snr
            and ctx.noise_class == "clean"
        )
        self._clean_run = self._clean_run + 1 if confidently_clean else 0

        # Target: bypass only after a sustained clean run; otherwise full DFN.
        if self._clean_run >= self._clean_needed:
            target = cfg.supp_min          # → enhancement bypass (passthrough)
        else:
            target = cfg.supp_max          # → full enhancement (trust DFN)

        # Attack/release smoothing (rise fast, fall slow → no pumping/chopping).
        a = self._a_attack if target > self._supp else self._a_release
        self._supp += a * (target - self._supp)
        ctx.suppression = float(np.clip(self._supp, cfg.supp_min, cfg.supp_max))

        # Post-filter aggressiveness: per-class, scaled down at high SNR.
        pf = _CLASS_POSTFILTER.get(ctx.noise_class, 0.5)
        snr_scale = float(np.interp(ctx.snr_db, [0.0, 25.0], [1.0, 0.3]))
        ctx.postfilter_strength = float(np.clip(pf * snr_scale, 0.0, 1.0))
        ctx.meta["ctrl_target"] = target
        ctx.meta["clean_run"] = self._clean_run
        return ctx

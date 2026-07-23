"""
Residual post-filter + comfort noise (final cleanup stage).

After enhancement there are two perceptual problems left:
  1. **Residual hiss / musical noise** in noise-only gaps.
  2. **Dead silence** — fully gated gaps sound unnatural and listeners think
     the call dropped.  Commercial products inject low-level *comfort noise*.

V2 additions:
  * **Class-shaped comfort noise.**  V1 used a single one-pole low-passed white
    pattern regardless of the actual room.  V2 picks a per-class IIR colour
    (LF-heavy for fan/HVAC, broad for traffic/wind, light for keyboard) so the
    injected ambience subjectively matches the room it's supposed to disguise.
  * **Comfort-noise level adapts to the estimated noise floor.**  CN is set
    ~10 dB above the smoothed noise floor (with a hard ``comfort_noise_db``
    ceiling) — when the room is quiet, CN is quieter; when the room is noisy,
    CN is slightly louder so the listener doesn't notice the *attenuation* of
    the residual.
  * **Soft limiter.**  After all stages have applied their gains (including the
    multi-band modulator's per-band scaling), the post-filter applies a
    ``tanh``-based soft limiter so any accidental overshoot above ±1 is
    rounded off smoothly rather than hard-clipped (which sounds like a click).

This stage is cheap (pure time-domain), runs after the wet/dry mix.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage


# Per-class one-pole comfort-noise colour:  y = b0·x + b1·x[-1] − a1·y[-1].
# Lower-pole closer to 1 → LF-heavy; closer to 0 → broadband white.
_CLASS_CN_FILTER: dict[str, tuple[np.ndarray, np.ndarray]] = {
    # default LF-shaped pink-ish
    "clean":           (np.array([0.05]),         np.array([1.0, -0.95])),
    # LF-heavy: fan / HVAC rooms
    "fan":             (np.array([0.025]),        np.array([1.0, -0.975])),
    "hvac":            (np.array([0.025]),        np.array([1.0, -0.975])),
    # broad: traffic / wind
    "traffic":         (np.array([0.10]),         np.array([1.0, -0.90])),
    "wind":            (np.array([0.08]),         np.array([1.0, -0.92])),
    # light / broadband for transient-dominant rooms
    "keyboard":        (np.array([0.15]),         np.array([1.0, -0.85])),
    "mouse_click":     (np.array([0.15]),         np.array([1.0, -0.85])),
    "dog_bark":        (np.array([0.10]),         np.array([1.0, -0.90])),
    "door_slam":       (np.array([0.08]),         np.array([1.0, -0.92])),
    # tonal sources: don't inject music-shaped CN, fall back to pink
    "music":           (np.array([0.05]),         np.array([1.0, -0.95])),
    "television":      (np.array([0.05]),         np.array([1.0, -0.95])),
    "competing_speech":(np.array([0.05]),         np.array([1.0, -0.95])),
}


class PostFilter(Stage):
    name = "postfilter"

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._cn_lin_max = 10.0 ** (cfg.comfort_noise_db / 20.0)
        self._floor_lin = 10.0 ** (cfg.postfilter_floor_db / 20.0)
        # Filter state for whichever class IIR is currently selected.
        self._cn_filter_class = "clean"
        self._cn_b, self._cn_a = _CLASS_CN_FILTER["clean"]
        self._cn_zi = np.zeros(max(len(self._cn_a), len(self._cn_b)) - 1, dtype=np.float64)
        self._rng = np.random.default_rng(1234)
        # Previous block's residual-gate gain — start point of this block's
        # per-sample ramp (the gate previously stepped up to −13 dB at every
        # is_speech flip: audible pumping at each utterance boundary).
        self._gate_prev = 1.0

    def reset(self) -> None:
        self._gate_prev = 1.0
        # Restore the startup CN colour AND re-seed the noise generator —
        # without the re-seed, a reset stream continues the old RNG sequence
        # and two identical inputs produce different comfort noise (breaks
        # sample-exact benchmark pair independence; caught by
        # test_reset_gives_sample_exact_independence).
        self._cn_filter_class = "clean"
        self._cn_b, self._cn_a = _CLASS_CN_FILTER["clean"]
        self._cn_zi = np.zeros(max(len(self._cn_a), len(self._cn_b)) - 1, dtype=np.float64)
        self._rng = np.random.default_rng(1234)

    def _select_cn_filter(self, noise_class: str) -> None:
        """Swap comfort-noise colour when the class changes."""
        if noise_class == self._cn_filter_class:
            return
        b, a = _CLASS_CN_FILTER.get(noise_class, _CLASS_CN_FILTER["clean"])
        self._cn_b = b
        self._cn_a = a
        # Re-seed state to a zero vector of the new filter's order.
        self._cn_zi = np.zeros(max(len(a), len(b)) - 1, dtype=np.float64)
        self._cn_filter_class = noise_class

    def _comfort_noise(self, n: int, noise_floor_db: float, echo_boost_db: float = 0.0) -> np.ndarray:
        """Generate ``n`` samples of class-shaped, level-adapted comfort noise.

        ``echo_boost_db`` lifts the CN level during DT-suppressed gaps so the
        listener doesn't perceive the AEC + NLES kill-zone as a dropped call.
        Default 0 (no boost) for the regular non-speech residual case.
        """
        white = self._rng.standard_normal(n)
        cn, self._cn_zi = lfilter(self._cn_b, self._cn_a, white, zi=self._cn_zi)
        cn = cn.astype(np.float32)
        rms = float(np.sqrt(np.mean(cn * cn)) + 1e-9)
        cn /= np.float32(rms)
        # Base level: 10 dB above the running noise floor, capped at comfort_noise_db.
        # Echo boost is added ON TOP, but still bounded by the configured cap
        # (+ the echo boost itself if echo mode is engaged).
        target_db = min(noise_floor_db + 10.0, self.cfg.comfort_noise_db)
        target_db += echo_boost_db
        # Allow the boost to push slightly above the default ceiling — capped
        # at comfort_noise_db + cn_echo_boost_db so it can't run away.
        ceiling_db = self.cfg.comfort_noise_db + self.cfg.cn_echo_boost_db
        target_db = min(target_db, ceiling_db)
        target_lin = 10.0 ** (target_db / 20.0)
        ceiling_lin = 10.0 ** (ceiling_db / 20.0)
        target_lin = float(np.clip(target_lin, 0.0, ceiling_lin))
        return cn * np.float32(target_lin)

    @staticmethod
    def _soft_limit(x: np.ndarray, knee: float = 0.95) -> np.ndarray:
        """tanh-based soft limiter.  Below ``knee`` it's near-linear; above it
        compresses smoothly so transient overshoots don't clip into clicks."""
        # Operate only on samples above the knee, leave the rest untouched —
        # avoids the tonal coloration tanh imparts to linear signals.
        out = x.copy()
        mask = np.abs(x) > knee
        if np.any(mask):
            over = x[mask]
            sign = np.sign(over)
            # Soft compress: |x| > knee → knee + (1 − knee) · tanh((|x|−knee)/(1−knee))
            mag = knee + (1.0 - knee) * np.tanh((np.abs(over) - knee) / (1.0 - knee))
            out[mask] = sign * mag
        return out

    def process(self, ctx: FrameContext) -> FrameContext:
        x = ctx.audio
        n = len(x)

        # Update comfort-noise filter to match current noise class so the colour
        # of the injected ambience tracks the room's character.
        self._select_cn_filter(ctx.noise_class)

        # 1. Residual attenuation in non-speech frames only — ramped per
        #    sample from the previous block's gain so the gate GLIDES across
        #    the block instead of stepping at its edge (with strength 0.9 the
        #    step was −13 dB: audible pumping at every speech offset/onset).
        if not ctx.is_speech and ctx.postfilter_strength > 0.0:
            g_target = 1.0 - ctx.postfilter_strength * (1.0 - self._floor_lin)
        else:
            g_target = 1.0
        if abs(g_target - self._gate_prev) > 1e-6:
            gain = np.linspace(self._gate_prev, g_target, n,
                               endpoint=True).astype(np.float32)
            x = x * gain
        elif g_target != 1.0:
            x = x * np.float32(g_target)
        self._gate_prev = g_target

        # 2. Class-shaped, level-adapted comfort noise.  Boost the level when
        #    AEC + NLES are aggressively suppressing echo (echo_conf high) and
        #    the frame isn't dominated by near-end speech (vad_prob low).  The
        #    boosted CN masks the otherwise-unnatural silence that aggressive
        #    DT suppression creates, preventing the "dropped call" perception.
        noise_floor_db = float(ctx.meta.get("noise_floor_db", -60.0))
        echo_boost_db = 0.0
        if (ctx.echo_conf > self.cfg.echo_conf_threshold
                and ctx.vad_prob < 0.4):
            # Scale boost linearly from 0 at threshold up to cn_echo_boost_db at 1.0.
            scale = float(np.clip(
                (ctx.echo_conf - self.cfg.echo_conf_threshold)
                / max(1.0 - self.cfg.echo_conf_threshold, 1e-3),
                0.0, 1.0,
            ))
            echo_boost_db = scale * float(self.cfg.cn_echo_boost_db)
        cn = self._comfort_noise(n, noise_floor_db, echo_boost_db=echo_boost_db)
        if echo_boost_db > 0.0:
            ctx.meta["cn_echo_boost_db"] = echo_boost_db
        y = x + cn

        # 3. Soft limit (prevents accidental overshoot from the band modulator
        #    or rollback blends from clipping to a click).
        y = self._soft_limit(y, knee=0.95)
        ctx.audio = np.clip(y, -1.0, 1.0).astype("float32")
        return ctx

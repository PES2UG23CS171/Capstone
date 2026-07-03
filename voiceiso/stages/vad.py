"""
Voice Activity Detection stage.

Primary: **Silero VAD** (silero-vad), an ONNX/torch model ~1 MB, ~1 ms/frame on
CPU, far more robust than energy or GMM VADs (handles music/noise without
false-triggering).  Runs at 16 kHz on 512-sample (32 ms) windows; we resample
the 48 kHz frame down for the VAD only.

V2 additions
~~~~~~~~~~~~
* **Adaptive threshold.**  Silero's raw probability distribution depends on the
  room's noise floor — in very quiet rooms even noise registers ~0.3, in very
  noisy rooms speech may peak at 0.5–0.7.  A fixed 0.5 threshold misses the
  edges.  V2 tracks the 5th- and 95th-percentile of vad_prob over ~30 s and
  sets the threshold at ``low + 0.4·(high − low)``, bounded to [0.35, 0.65].
* **Adaptive hangover.**  Word-final fricatives (/s/, /f/, /sh/) have very
  short voice tails; vowel endings have longer tails.  V2 chooses a short
  hangover (~80 ms) when offset frames look fricative (HF energy fraction
  above threshold), else the default 200 ms.  This tightens offset behaviour
  without chopping word tails.

Fallback: an adaptive energy/SNR gate if silero-vad is unavailable, so the
pipeline always has *some* speech probability.

Output:
  * ``ctx.vad_prob``  — raw P(speech) ∈ [0, 1]
  * ``ctx.is_speech`` — thresholded + hangover-smoothed boolean
  * ``ctx.meta['vad_threshold']`` — current adaptive threshold (diagnostic)
  * ``ctx.meta['vad_hangover_ms']`` — current adaptive hangover (diagnostic)
"""

from __future__ import annotations

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage

try:
    import torch
    from silero_vad import load_silero_vad
    _HAS_SILERO = True
except Exception:  # pragma: no cover
    _HAS_SILERO = False


def _resample_poly(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


class VAD(Stage):
    name = "vad"

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.vsr = cfg.vad_sample_rate
        self.win = cfg.vad_window
        # Base threshold and hangover (overridable by adaptation).
        self.base_threshold = cfg.vad_speech_threshold
        self.threshold = cfg.vad_speech_threshold
        # Hangover tracked in samples so it's robust to caller block size.
        self._hangover_samples_default = int(cfg.sample_rate * cfg.vad_hangover_ms / 1000.0)
        self._hangover_samples_fricative = int(cfg.sample_rate * cfg.vad_hangover_fricative_ms / 1000.0)
        self._hang_samples = 0
        self._buf = np.zeros(0, dtype=np.float32)   # accumulates 16 kHz samples
        self._last_prob = 0.0
        self.backend = "none"

        # ── Adaptive-threshold state ─────────────────────────────────────
        # EWMA over vad_prob (low percentile floor + high percentile ceiling).
        # Initialise to the base threshold so the first ~30 s behave like fixed-threshold.
        self._p_floor = 0.0
        self._p_ceil = 1.0
        self._adapt_alpha = float(cfg.vad_adapt_alpha)
        self._adapt = bool(cfg.vad_adaptive)

        if _HAS_SILERO:
            torch.set_num_threads(1)
            self._model = load_silero_vad(onnx=True)
            self.backend = "silero"
        else:
            self._model = None
            self.backend = "energy"
            self._noise = 1e-4

    def reset(self) -> None:
        self._hang_samples = 0
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0
        self._p_floor = 0.0
        self._p_ceil = 1.0
        self.threshold = self.base_threshold
        if self.backend == "silero" and hasattr(self._model, "reset_states"):
            self._model.reset_states()

    # ── backends ─────────────────────────────────────────────────────────
    def _silero_prob(self, frame48: np.ndarray) -> float:
        self._buf = np.concatenate([self._buf, _resample_poly(frame48, self.sr, self.vsr)])
        prob = self._last_prob
        while len(self._buf) >= self.win:
            chunk = self._buf[: self.win]
            self._buf = self._buf[self.win :]
            with torch.no_grad():
                prob = float(self._model(torch.from_numpy(chunk).float(), self.vsr).item())
        self._last_prob = prob
        return prob

    def _energy_prob(self, frame48: np.ndarray) -> float:
        p = float(np.mean(frame48 * frame48) + 1e-12)
        self._noise = min(self._noise * 1.02, max(self._noise, p)) if p > self._noise else \
            0.95 * self._noise + 0.05 * p
        ratio = 10.0 * np.log10(p / max(self._noise, 1e-10))
        return float(np.clip((ratio - 3.0) / 12.0, 0.0, 1.0))

    def _update_adaptive_threshold(self, prob: float) -> None:
        """Asymmetric EWMA tracking floor (5th-pct proxy) and ceiling (95th-pct).

        Floor pulls down on every sample below it; ceiling pulls up on every
        sample above it.  This gives a stable estimate of the room's vad_prob
        distribution that's robust to brief outliers.
        """
        if not self._adapt:
            return
        a = self._adapt_alpha
        # Asymmetric decay: ceiling rises 5× faster than it falls (so a quiet
        # period doesn't immediately collapse it); floor falls 5× faster than
        # it rises (so a brief loud event doesn't permanently raise it).
        if prob > self._p_ceil:
            self._p_ceil += (5.0 * a) * (prob - self._p_ceil)
        else:
            self._p_ceil += a * (prob - self._p_ceil)
        if prob < self._p_floor:
            self._p_floor += (5.0 * a) * (prob - self._p_floor)
        else:
            self._p_floor += a * (prob - self._p_floor)
        # Threshold: 40 % of the way from floor to ceiling, clamped.
        spread = max(self._p_ceil - self._p_floor, 1e-3)
        self.threshold = float(np.clip(
            self._p_floor + 0.4 * spread,
            self.cfg.vad_threshold_min,
            self.cfg.vad_threshold_max,
        ))

    @staticmethod
    def _hf_ratio(x: np.ndarray, sr: int, cutoff_hz: float = 3500.0) -> float:
        n = len(x)
        if n < 32:
            return 0.0
        mag = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        tot = float(mag.sum()) + 1e-12
        hf = float(mag[freqs >= cutoff_hz].sum())
        return hf / tot

    def process(self, ctx: FrameContext) -> FrameContext:
        prob = self._silero_prob(ctx.audio) if self.backend == "silero" else self._energy_prob(ctx.audio)
        # Echo-aware downweight: when AEC is confident the frame contains
        # echo, leaked far-end speech may inflate Silero's prob above the
        # threshold even though there's no near-end voice.  Use the same
        # threshold gate as other consumers (only fire above
        # echo_conf_threshold) so we don't silently degrade VAD in the
        # 0–threshold range where other consumers are quiescent.
        if ctx.echo_conf > self.cfg.echo_conf_threshold:
            prob = prob * (1.0 - 0.5 * ctx.echo_conf)
        ctx.vad_prob = prob
        # During AEC calibration the audio downstream is the raw mic
        # (echo-laden); don't update adaptive state on those samples.
        if float(ctx.meta.get("aec_calibrating", 0.0)) < 0.5:
            self._update_adaptive_threshold(prob)
        block_samples = len(ctx.audio)

        if prob >= self.threshold:
            # Speech detected.  At onset, pick the hangover length we'd use
            # IF this frame turned out to be the last voiced frame.  We can
            # only know the right length retroactively; the simple rule is:
            # check this frame's HF ratio — if it's HF-dominant, the speech
            # is likely ending fricative-like and the tail will be short.
            hf = self._hf_ratio(ctx.audio, self.sr)
            if hf >= self.cfg.vad_fricative_hf_ratio:
                self._hang_samples = self._hangover_samples_fricative
            else:
                self._hang_samples = self._hangover_samples_default
            ctx.is_speech = True
        else:
            if self._hang_samples > 0:
                self._hang_samples = max(0, self._hang_samples - block_samples)
                ctx.is_speech = True       # hangover: don't chop word tails
            else:
                ctx.is_speech = False

        ctx.meta["vad_backend"] = 1.0 if self.backend == "silero" else 0.0
        ctx.meta["vad_threshold"] = float(self.threshold)
        ctx.meta["vad_p_floor"] = float(self._p_floor)
        ctx.meta["vad_p_ceil"] = float(self._p_ceil)
        ctx.meta["vad_hangover_samples"] = float(self._hang_samples)
        return ctx

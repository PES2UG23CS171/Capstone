"""
Streaming noise classification.

Knowing *what* the noise is lets the controller pick a strategy: hard-suppress
stationary fan/HVAC, briefly spike on keyboard/door transients, and lean on the
post-filter for music/TV/competing-speech where residual is most annoying.

Two implementations behind one interface:

* :class:`HeuristicNoiseClassifier` (default, zero-dependency, ~CPU-free) —
  classifies from cheap spectral/temporal features (flatness, centroid, HF
  ratio, flux, low-freq modulation).  Reliable for the coarse decision the
  controller needs (stationary vs transient vs tonal vs speech-like); it does
  *not* finely separate e.g. fan-vs-HVAC.

* :class:`YamnetNoiseClassifier` (interface/stub) — the production upgrade: a
  MobileNet-class AudioSet tagger (YAMNet/PANNs) exported to ONNX, ~1–3 ms/frame
  on CPU, mapping 521 AudioSet tags onto our label set.  Drop-in, same output.

Output: ``ctx.noise_class`` (top label) + ``ctx.noise_probs`` (distribution).
"""

from __future__ import annotations

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage


class _BaseClassifier(Stage):
    name = "noise_classifier"

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.classes = cfg.noise_classes


class HeuristicNoiseClassifier(_BaseClassifier):
    """Cheap spectral/temporal heuristic classifier with EWMA frame smoothing."""

    # α for exponential smoothing of per-class probabilities across frames.
    # Low α = slow response (stable labels); high α = fast but noisy.
    _SMOOTH_ALPHA = 0.35

    def __init__(self, cfg: PipelineConfig) -> None:
        super().__init__(cfg)
        self._prev_mag = None
        self._energy_hist: list[float] = []
        self._smoothed: dict[str, float] = {c: 0.0 for c in cfg.noise_classes}
        # Running average flux (used as the transient detector's baseline).
        # Compare instantaneous flux against this slow average — anything
        # ≥ 3× the running average is a credible onset.
        self._flux_avg = 1e-6
        # Previous-frame spectral centroid — used by the high-VAD competing-
        # speech detector (rapid centroid drift = two-talker overlap signature).
        self._prev_centroid: float | None = None

    def reset(self) -> None:
        self._prev_mag = None
        self._energy_hist.clear()
        self._smoothed = {c: 0.0 for c in self.classes}
        self._flux_avg = 1e-6
        self._prev_centroid = None

    @staticmethod
    def _flatness(mag: np.ndarray) -> float:
        m = mag + 1e-10
        return float(np.exp(np.mean(np.log(m))) / np.mean(m))   # geo/arith mean

    def process(self, ctx: FrameContext) -> FrameContext:
        x = ctx.audio
        win = x * np.hanning(len(x))
        mag = np.abs(np.fft.rfft(win))
        freqs = np.fft.rfftfreq(len(x), 1.0 / self.sr)
        energy = float(np.mean(x * x) + 1e-12)

        self._energy_hist.append(energy)
        if len(self._energy_hist) > 50:
            self._energy_hist.pop(0)
        stationarity = 1.0 / (1.0 + np.std(self._energy_hist) / (np.mean(self._energy_hist) + 1e-9))

        flat = self._flatness(mag)                              # 1 = white, →0 tonal
        centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-9))
        hf = float(mag[freqs >= 3500].sum() / (mag.sum() + 1e-9))
        lf = float(mag[freqs < 500].sum() / (mag.sum() + 1e-9))
        flux = 0.0
        if self._prev_mag is not None:
            flux = float(np.mean(np.maximum(mag - self._prev_mag, 0.0)))
        self._prev_mag = mag

        # Update running-average flux (slow EMA — ~1 s time constant at 20 ms
        # block rate, α=0.04).  This is the baseline against which we test
        # transient onsets.  A previous version compared flux to mag.mean(),
        # which is mathematically impossible because flux ≤ mean(mag) always.
        self._flux_avg = 0.96 * self._flux_avg + 0.04 * flux

        probs: dict[str, float] = {c: 0.0 for c in self.classes}

        # 1. Effectively silent / clean relative to floor.
        if ctx.snr_db < 3.0 and energy < 10 ** (ctx.meta.get("noise_floor_db", -60) / 10) * 3:
            probs["clean"] = 1.0
        # 2. Transient: instantaneous flux is much larger than its running average
        #    (sharp onset).  Threshold 3× the EMA with an absolute floor so a
        #    sustained quiet signal doesn't fire on tiny relative spikes.
        elif flux > 3.0 * self._flux_avg and flux > 1e-4:
            if lf > 0.5:
                probs["door_slam"] = 0.8
            elif hf > 0.4:
                probs["keyboard"] = 0.8
            else:
                probs["dog_bark"] = 0.7
        # 3. Tonal / harmonic & sustained → music or TV.
        elif flat < 0.15 and stationarity > 0.6:
            probs["music"] = 0.6
            probs["television"] = 0.3
        # 4. Speech-like spectrum.  Two reachability cases:
        #    (a) VAD low (gap between primary-speaker words) but the spectrum
        #        still looks like speech → background talker.
        #    (b) VAD high but centroid is *unstably* drifting frame-to-frame —
        #        the hallmark of two-talker overlap (each speaker pulls the
        #        composite centroid toward their own formant centre).  Single
        #        speaker formants drift slowly; competing talkers drift fast.
        elif 200 < centroid < 2500 and 0.15 < flat < 0.5:
            if ctx.vad_prob < 0.35:
                probs["competing_speech"] = 0.6     # case (a)
            elif self._prev_centroid is not None:
                # Frame-to-frame relative centroid change normalised by centroid
                # magnitude.  Two-talker overlap typically gives > 0.15.
                drift = abs(centroid - self._prev_centroid) / max(centroid, 1.0)
                if drift > 0.15 and ctx.vad_prob >= 0.4:
                    probs["competing_speech"] = 0.5     # case (b)
        # 5. Stationary broadband → fan/HVAC (low) or traffic (mid/high).
        elif stationarity > 0.5:
            if centroid < 800:
                probs["fan"] = 0.5; probs["hvac"] = 0.3
            elif centroid < 2500:
                probs["traffic"] = 0.5
            else:
                probs["wind"] = 0.4; probs["traffic"] = 0.3
        else:
            probs["traffic"] = 0.4

        # EWMA smoothing across frames: prevents single-frame outliers from
        # triggering controller reactions through the 30 ms attack path.
        alpha = self._SMOOTH_ALPHA
        for c in self.classes:
            self._smoothed[c] = (1.0 - alpha) * self._smoothed[c] + alpha * probs.get(c, 0.0)

        top = max(self._smoothed, key=self._smoothed.get)
        if self._smoothed[top] < 1e-6:
            top = "clean"
        ctx.noise_class = top
        ctx.noise_probs = dict(self._smoothed)
        # Top-class probability — used by the controller's confidence scaling.
        ctx.noise_conf = float(self._smoothed[top])
        ctx.meta["spec_flatness"] = flat
        ctx.meta["spec_centroid"] = centroid
        ctx.meta["noise_conf"] = ctx.noise_conf
        self._prev_centroid = centroid
        return ctx


class YamnetNoiseClassifier(_BaseClassifier):
    """Production upgrade — AudioSet tagger (YAMNet/PANNs) via ONNX. (interface stub)

    Implementation plan:
      1. Export YAMNet (or PANNs CNN14) to ONNX; run via onnxruntime on 0.96 s
         log-mel patches with a streaming hop.
      2. Reduce its 521 AudioSet posteriors onto ``cfg.noise_classes`` with a
         fixed label-map (e.g. AudioSet "Computer keyboard" → "keyboard").
      3. Smooth across frames; emit top label + distribution into ``ctx``.
    CPU: ~1–3 ms/frame quantized.  Memory: ~4–15 MB.
    """

    def __init__(self, cfg: PipelineConfig, onnx_path: str | None = None) -> None:
        super().__init__(cfg)
        raise NotImplementedError(
            "YamnetNoiseClassifier is the learned-model upgrade path; "
            "use HeuristicNoiseClassifier until the ONNX tagger is wired in."
        )

    def process(self, ctx: FrameContext) -> FrameContext:  # pragma: no cover
        raise NotImplementedError

"""
Streaming noise classification.

Knowing *what* the noise is lets the controller pick a strategy: hard-suppress
stationary fan/HVAC, briefly spike on keyboard/door transients, and lean on the
post-filter for music/TV/competing-speech where residual is most annoying.

Two implementations behind one ``ctx.noise_class`` / ``ctx.noise_probs`` /
``ctx.noise_conf`` contract:

* :class:`EfficientATNoiseClassifier` — **primary path**, learned AudioSet
  tagger (EfficientAT mn10) exported to ONNX, ~11 ms inference / 4 s window
  re-run every 500 ms.  With a native 12-label head the AudioSet map is
  skipped; otherwise 527 AudioSet posteriors map onto the 12-class label set
  via a hand-coded map.  Meaningfully more discriminative than the heuristic.

* :class:`HeuristicNoiseClassifier` — **fallback**, zero-dependency, ~CPU-free.
  Used when no ONNX runtime / no checkpoint.  Coarse 5-cluster discrimination
  (clean / transient / tonal / speech-like / stationary-broadband).

Both produce the same ``ctx`` interface, so the rest of the pipeline (controller,
multi-band gain modulator, etc.) doesn't know or care which one is active.

Output: ``ctx.noise_class`` (top label), ``ctx.noise_probs`` (distribution),
``ctx.noise_conf`` (top-label posterior; calibrated for EfficientAT, fixed for
heuristic).
"""

from __future__ import annotations

import logging

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage

logger = logging.getLogger(__name__)


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
        # Skip during AEC calibration so the raw mic doesn't contaminate
        # the per-class EWMA at session start.
        calibrating = float(ctx.meta.get("aec_calibrating", 0.0)) >= 0.5
        if calibrating:
            top = max(self._smoothed, key=self._smoothed.get) if self._smoothed else "clean"
            if not self._smoothed or self._smoothed[top] < 1e-6:
                top = "clean"
            ctx.noise_class = top
            ctx.noise_probs = dict(self._smoothed)
            ctx.noise_conf = float(self._smoothed.get(top, 0.0))
            ctx.meta["noise_conf"] = ctx.noise_conf
            return ctx

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
        # Guard against a variable block size (e.g. the trailing partial block
        # of an offline signal): only compute flux when the spectra align.
        if self._prev_mag is not None and self._prev_mag.shape == mag.shape:
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


# ── Two-stage label mapping ──────────────────────────────────────────────────
#
# Stage A:  AudioSet ontology label  →  approved *target* taxonomy (12 labels).
# Stage B:  target taxonomy          →  internal pipeline class (``ctx.noise_class``)
#           that the controller / post-filter already consume — so the
#           controller is NOT redesigned.
#
# The target taxonomy is the externally-meaningful one; it is surfaced on
# ``ctx.meta["noise_label"]``.  The internal class drives suppression policy.
#
# Division of labour for SPEECH:  the classifier only decides "speech present"
# (target label ``speech``) vs "multiple talkers / babble" (``competing_speech``).
# Whether speech is the *target* or a *competing* talker is the SpeakerEmbedder's
# job (ECAPA cosine), NOT the classifier's — so generic AudioSet "Speech" maps to
# ``speech`` (treated gently downstream), and only multi-talker cues
# (Babble / Crowd / Conversation / Hubbub) map to ``competing_speech``.

# Approved target taxonomy (canonical lowercase keys; display names in comments).
_TARGET_LABELS: tuple[str, ...] = (
    "clean", "fan", "hvac", "traffic", "keyboard", "dog", "wind",
    "music", "tv", "speech", "competing_speech", "other",
)

# Stage B: target → internal ctx.noise_class.  Internal classes are exactly
# those the controller / PostFilter key their per-class tables on.  Unknown
# internal labels ("other") fall through to the controller's safe defaults
# (moderate full-band suppression), and never trigger the clean-run bypass.
_TARGET_TO_INTERNAL: dict[str, str] = {
    "clean":            "clean",
    "fan":              "fan",
    "hvac":             "hvac",          # display: HVAC
    "traffic":          "traffic",
    "keyboard":         "keyboard",
    "dog":              "dog_bark",      # controller keys on dog_bark
    "wind":             "wind",
    "music":            "music",
    "tv":               "television",    # display: TV; controller keys on television
    # Generic speech present → map to "other", NOT "clean".  "speech" says the
    # window's FOREGROUND is the user talking — it says nothing about the
    # background noise they are talking over.  Mapping it to clean set the
    # suppression modifier to 0.0, which disabled DFN3 for the entire time the
    # user was speaking in noise — the exact scenario a voice isolator exists
    # for (measured: SI-SDRi ≈ 0 on speech+noise mixes because enhancement
    # never engaged).  Via "other" the controller applies its SNR-driven
    # default: strong suppression in bad SNR, ~none in a genuinely quiet room.
    # The clean-hold bypass still fires in confirmed-quiet silence (it
    # requires noise_class=="clean" AND no speech AND SNR ≥ 22 dB).
    "speech":           "other",
    "competing_speech": "competing_speech",
    # Unknown noise → pass through; controller .get(..., default) applies a
    # moderate full-band suppression and PostFilter uses the clean CN colour.
    "other":            "other",
}

# Stage A: AudioSet label (lowercase) → target label.  Matched case-insensitively
# against the model's class-name list.  Multiple AudioSet labels collapse to one
# target (max-pooled).
_AUDIOSET_TO_TARGET: dict[str, str] = {
    # Keyboard / typing
    "computer keyboard":            "keyboard",
    "typing":                       "keyboard",
    "typewriter":                   "keyboard",
    "tap":                          "keyboard",
    "mouse click":                  "keyboard",
    # Dog
    "dog":                          "dog",
    "bark":                         "dog",
    "yip":                          "dog",
    "howl":                         "dog",
    "growling":                     "dog",
    # Fan / electrical hum → fan; air-conditioning/ventilation → hvac
    "fan":                          "fan",
    "mechanical fan":               "fan",
    "air conditioning":             "hvac",
    "ventilation":                  "hvac",
    "mains hum":                    "hvac",
    "hum":                          "hvac",
    # Traffic
    "vehicle":                      "traffic",
    "traffic noise, roadway noise": "traffic",
    "car":                          "traffic",
    "car passing by":               "traffic",
    "truck":                        "traffic",
    "motorcycle":                   "traffic",
    "engine":                       "traffic",
    "engine starting":              "traffic",
    # Wind
    "wind":                         "wind",
    "wind noise (microphone)":      "wind",
    "rustling leaves":              "wind",
    "rustle":                       "wind",
    # Music
    "music":                        "music",
    "musical instrument":           "music",
    "background music":             "music",
    "song":                         "music",
    "guitar":                       "music",
    "piano":                        "music",
    # TV / radio
    "television":                   "tv",
    "radio":                        "tv",
    # Single-talker speech → "speech" (target-vs-competing decided by ECAPA)
    "speech":                       "speech",
    "narration, monologue":         "speech",
    "male speech, man speaking":    "speech",
    "female speech, woman speaking": "speech",
    "child speech, kid speaking":   "speech",
    "speech synthesizer":           "speech",
    # Multi-talker / babble → competing_speech
    "conversation":                 "competing_speech",
    "babbling":                     "competing_speech",
    "hubbub, speech noise, speech babble": "competing_speech",
    "crowd":                        "competing_speech",
    "chatter":                      "competing_speech",
    # Misc real noise with no specific bucket → "other" (exercises the catch-all)
    "clatter":                      "other",
    "mechanisms":                   "other",
    "tools":                        "other",
    "glass":                        "other",
    "water":                        "other",
    "rain":                         "other",
    "thunderstorm":                 "other",
    "white noise":                  "other",
    "static":                       "other",
    # Silence
    "silence":                      "clean",
}


class EfficientATNoiseClassifier(_BaseClassifier):
    """Learned AudioSet tagger (EfficientAT-S) — primary noise classifier.

    Architecture: MobileNetV3-class backbone (EfficientAT ``mn10_as``, width 1.0)
    distilled on AudioSet; ~4.9 M params, ONNX INT8 ~5 MB resident,
    ~2–4 ms / 200 ms window single-thread on CPU (amortises to <1 ms/block at
    the 50 ms rerun cadence).

    Two model output conventions are supported automatically:
      * **AudioSet head** — model emits 527 AudioSet posteriors; we max-pool
        them into the 12 target labels via ``_AUDIOSET_TO_TARGET``.
      * **Native 12-label head** — a fine-tuned head whose ``labels`` metadata
        is exactly the 12 target labels; the AudioSet map is skipped (identity).

    Window cadence: a 4 s sliding window (the head's static mel input) re-run
    every ``noise_classifier_hop_ms`` of audio; between updates the
    previously-computed posteriors are reused (~11 ms inference amortised over
    500 ms ≈ 2 % of one core).  Until the window first fills, a partial buffer
    of ≥ ``noise_classifier_min_infer_s`` is tile-repeated to the window
    length (mirrors the training pad policy for short clips).

    Output contract (controller-compatible):
      * ``ctx.noise_class``  — INTERNAL class (clean/fan/hvac/traffic/keyboard/
        dog_bark/wind/music/television/competing_speech/other) consumed by the
        controller + post-filter.
      * ``ctx.noise_probs``  — distribution over the 12 TARGET labels (contains
        "competing_speech", which downstream stages look up by key).
      * ``ctx.noise_conf``   — top target-label posterior.
      * ``ctx.meta["noise_label"]`` — the human-meaningful TARGET label
        (clean/fan/hvac/traffic/keyboard/dog/wind/music/tv/speech/
        competing_speech/other).

    Bypass: when ``onnxruntime`` isn't installed, the model path is missing,
    or the model lacks label metadata, ``__init__`` raises ``RuntimeError`` so
    the pipeline can catch and instantiate :class:`HeuristicNoiseClassifier`
    transparently.
    """

    # EfficientAT mn10_as is a 32 kHz model with a specific AugmentMelSTFT
    # front-end.  We compute that exact mel at runtime (vendored) and feed a
    # mel spectrogram (1, 1, 128, T) to a MODEL-ONLY ONNX graph — robust
    # (no STFT-in-ONNX) and guaranteed to match training.
    _MODEL_SR = 32000
    _N_MELS = 128

    def __init__(self, cfg: PipelineConfig) -> None:
        super().__init__(cfg)
        try:
            import onnxruntime as ort  # type: ignore[import]
        except Exception as exc:       # pragma: no cover
            raise RuntimeError(f"onnxruntime not available: {exc}")
        from pathlib import Path
        ckpt = cfg.noise_classifier_model_path
        if not ckpt or not Path(ckpt).exists():
            raise RuntimeError(
                f"noise_classifier_model_path not set or missing: {ckpt!r}"
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            ckpt, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        # Read the model's class-name list from its metadata.  Without
        # labels we cannot map AudioSet posteriors to our 12-class space,
        # so the entire stage is functionally useless — raise so the
        # pipeline can fall back to HeuristicNoiseClassifier transparently
        # rather than silently classifying every frame as "clean".
        meta = self._session.get_modelmeta().custom_metadata_map
        labels_str = meta.get("labels") or meta.get("class_names") or ""
        # Split on "|" (the export delimiter) — NOT "," — because AudioSet
        # display names contain commas (e.g. "Male speech, man speaking").
        # Fall back to "," only if no "|" is present (legacy models).
        if "|" in labels_str:
            self._labels = [s.strip() for s in labels_str.split("|") if s.strip()]
        else:
            self._labels = [s.strip() for s in labels_str.split(",") if s.strip()] if labels_str else []
        if not self._labels:
            raise RuntimeError(
                "noise_classifier ONNX has no 'labels' or 'class_names' "
                "metadata; cannot map AudioSet posteriors → voiceiso classes"
            )
        # Detect a native 12-label head (fine-tuned directly on the target
        # taxonomy): its labels are exactly _TARGET_LABELS, so the AudioSet
        # map is skipped and posteriors are used as target probs directly.
        self._native_target = (
            {l.strip().lower() for l in self._labels} == set(_TARGET_LABELS)
        )
        # Required mel frame count, read from the ONNX input shape.  The shipped
        # native-12 head is exported with a STATIC time axis (e.g. [1,1,128,400]
        # for a 4 s window); feeding any other frame count raises InvalidArgument.
        # We capture it here and pad/crop the runtime mel to match in
        # ``_posteriors`` so a window/model mismatch can never silently fail.
        self._req_frames: int | None = None
        try:
            t_dim = self._session.get_inputs()[0].shape[-1]
            if isinstance(t_dim, int) and t_dim > 0:
                self._req_frames = t_dim
        except Exception:  # pragma: no cover - defensive
            self._req_frames = None
        # Runtime mel front-end (vendored EfficientAT AugmentMelSTFT, eval, 32 kHz).
        from voiceiso.stages._efficientat_mel import EfficientATMel
        self._mel = EfficientATMel(n_mels=self._N_MELS, sr=self._MODEL_SR)
        # Buffer at the MODEL sample rate (32 kHz); rerun cadence in MILLISECONDS
        # of input audio (not process() calls — caller block size varies).
        self._buf_model = np.zeros(0, dtype=np.float32)
        self._window_samples = int(self._MODEL_SR * cfg.noise_classifier_window_s)
        # Minimum buffered audio before the first inference: a partial buffer
        # (≥ this, < window) is tile-repeated to the window length (H2).
        self._min_infer_samples = int(self._MODEL_SR * float(
            getattr(cfg, "noise_classifier_min_infer_s", 1.0)))
        # One-shot guard so a recurring inference error logs once, not per block.
        self._warned_infer = False
        if self._req_frames is not None:
            logger.info(
                "EfficientAT classifier: model expects %d mel frames; "
                "runtime window %.2f s @ %d Hz → ~%d frames",
                self._req_frames, cfg.noise_classifier_window_s, self._MODEL_SR,
                int(self._window_samples / self._mel.hopsize),
            )
        self._hop_ms = float(cfg.noise_classifier_hop_ms)
        self._ms_since_update = 0.0
        # Smoothing happens in the TARGET-label space (the 12 approved labels).
        self._smoothed: dict[str, float] = {t: 0.0 for t in _TARGET_LABELS}
        self._alpha = float(cfg.noise_classifier_smooth_alpha)
        self._clean_threshold = float(cfg.noise_classifier_clean_threshold)

    def reset(self) -> None:
        self._buf_model = np.zeros(0, dtype=np.float32)
        self._ms_since_update = 0.0
        self._smoothed = {t: 0.0 for t in _TARGET_LABELS}

    def _resample_to_model_sr(self, x: np.ndarray) -> np.ndarray:
        if self.sr == self._MODEL_SR:
            return x.astype(np.float32)
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(self.sr, self._MODEL_SR)
        return resample_poly(x, self._MODEL_SR // g, self.sr // g).astype(np.float32)

    def _posteriors(self, window_model_sr: np.ndarray) -> np.ndarray:
        """Compute the 32 kHz mel and run one MODEL-ONLY inference.

        ONNX input is the mel spectrogram (1, 1, 128, T); output is the
        sigmoid posterior vector.
        """
        mel = self._mel(window_model_sr)                       # torch (1, 128, T)
        feats = mel.unsqueeze(1).detach().cpu().numpy().astype(np.float32)  # (1,1,128,T)
        # Force the time axis to exactly what the (statically-shaped) ONNX wants:
        # crop to the most-recent frames, or edge-pad if a frame short.  This
        # makes the classifier robust to ±1-frame rounding and to any
        # window/model mismatch instead of throwing InvalidArgument every block.
        if self._req_frames is not None and feats.shape[-1] != self._req_frames:
            t = feats.shape[-1]
            if t > self._req_frames:
                feats = feats[..., -self._req_frames:]
            else:
                pad = self._req_frames - t
                feats = np.pad(feats, ((0, 0), (0, 0), (0, 0), (pad, 0)), mode="edge")
        out = self._session.run(None, {self._session.get_inputs()[0].name: feats})
        return np.asarray(out[0]).reshape(-1).astype(np.float32)

    def _map_to_target_labels(self, post: np.ndarray) -> dict[str, float]:
        """Reduce model posteriors to the 12 TARGET labels.

        Native 12-label head → identity (posteriors ARE target probs).
        AudioSet head → max-pool via ``_AUDIOSET_TO_TARGET``.
        """
        out: dict[str, float] = {t: 0.0 for t in _TARGET_LABELS}
        if not self._labels or len(post) != len(self._labels):
            return out
        if self._native_target:
            for i, lbl in enumerate(self._labels):
                out[lbl.strip().lower()] = float(post[i])
            return out
        for i, lbl in enumerate(self._labels):
            tgt = _AUDIOSET_TO_TARGET.get(lbl.lower())
            if tgt is None:
                continue
            # Max-pool: multiple AudioSet labels mapping to one target take the
            # strongest, not the sum.
            out[tgt] = max(out[tgt], float(post[i]))
        # "clean" = complement of the strongest non-clean target.  This lets a
        # dominant "speech" label surface in telemetry (it maps to internal
        # "clean" downstream, so suppression is unchanged), while genuine
        # silence — where every non-clean target is ~0 — still wins "clean".
        max_non_clean = max((out[t] for t in _TARGET_LABELS if t != "clean"), default=0.0)
        out["clean"] = max(out["clean"], 1.0 - max_non_clean)
        return out

    def _emit(self, ctx: FrameContext) -> FrameContext:
        """Write controller-compatible outputs from the current target-space
        smoothed distribution.

        Selection: take the strongest NON-clean target.  Only fall back to
        "clean" when even that is below ``clean_threshold`` — the synthesized
        "clean" complement must not auto-win the argmax (AudioSet posteriors
        for a correct coarse class on a short clip are modest, ~0.2–0.4).
        """
        non_clean = [(t, self._smoothed[t]) for t in _TARGET_LABELS if t != "clean"]
        top, top_v = max(non_clean, key=lambda kv: kv[1])
        if top_v < self._clean_threshold:
            top = "clean"
        internal = _TARGET_TO_INTERNAL.get(top, "other")
        ctx.noise_class = internal                       # controller consumes this
        ctx.noise_probs = dict(self._smoothed)           # target-space distribution
        ctx.noise_conf = float(self._smoothed[top])
        ctx.meta["noise_conf"] = ctx.noise_conf
        ctx.meta["noise_label"] = top                    # human-meaningful target
        ctx.meta["noise_classifier_backend"] = 1.0       # 1.0 = learned EfficientAT
        return ctx

    def process(self, ctx: FrameContext) -> FrameContext:
        # Skip buffer updates while AEC is calibrating — ctx.audio is the raw
        # echo-laden mic then, and contaminating the EWMA would mis-label the
        # first ~0.5–1 s of the session.  Still emit the current state.
        calibrating = float(ctx.meta.get("aec_calibrating", 0.0)) >= 0.5
        if calibrating:
            return self._emit(ctx)

        # Accumulate at the model SR (32 kHz); trim to twice the window.
        block_ms = 1000.0 * len(ctx.audio) / max(self.sr, 1)
        self._buf_model = np.concatenate([self._buf_model, self._resample_to_model_sr(ctx.audio)])
        if len(self._buf_model) > self._window_samples * 2:
            self._buf_model = self._buf_model[-self._window_samples:]
        self._ms_since_update += block_ms

        # Rerun inference at the configured cadence.  A FULL window is not
        # required: with ≥ noise_classifier_min_infer_s buffered, a partial
        # buffer is tile-repeated to the model's window — mirroring the
        # training pad policy for short clips.  Without this the classifier
        # physically could not label the first 4 s of every session or any
        # clip shorter than 4 s (18.4 % of the FSD50K eval set was forced to
        # "clean" in the reported numbers).
        min_samples = self._min_infer_samples
        have = len(self._buf_model)
        if self._ms_since_update >= self._hop_ms and have >= min_samples:
            self._ms_since_update = 0.0
            if have >= self._window_samples:
                win = self._buf_model[-self._window_samples:]
            else:
                reps = int(np.ceil(self._window_samples / have))
                win = np.tile(self._buf_model, reps)[-self._window_samples:]
            try:
                post = self._posteriors(win)
                fresh = self._map_to_target_labels(post)
                for t in _TARGET_LABELS:
                    self._smoothed[t] = (1.0 - self._alpha) * self._smoothed[t] + self._alpha * fresh.get(t, 0.0)
            except Exception as exc:  # pragma: no cover
                # On inference error, keep the prior smoothed values rather than
                # spiking to a wrong class — but make the failure VISIBLE once so
                # a shape/model regression can never again silently degrade the
                # classifier to all-"clean".
                if not self._warned_infer:
                    self._warned_infer = True
                    logger.warning(
                        "EfficientAT inference failed (%s: %s); holding prior "
                        "posteriors. Check that the runtime mel frame count "
                        "matches the model's expected %s frames.",
                        type(exc).__name__, exc, self._req_frames,
                    )

        return self._emit(ctx)


# Back-compat alias — keep the YAMNet name available so external code that
# constructed it by name doesn't break.  Functionally identical to the new
# EfficientAT class.
YamnetNoiseClassifier = EfficientATNoiseClassifier

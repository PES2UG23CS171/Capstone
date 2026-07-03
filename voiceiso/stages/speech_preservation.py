"""
Speech-preservation branch.

Enhancers are most destructive to **consonants** (fricatives /s/ /f/ /sh/ look
like HF hiss; plosives /p/ /t/ /k/ look like clicks) and to **quiet voiced
speech** (the network may treat it as noise).  This stage runs *after* the
controller and detects three cues:

1. **Consonant evidence** — HF energy ratio (fricative) and broadband onset
   (plosive).  Triggers protect-mode on the post-filter.
2. **Voiced score (V2)** — autocorrelation-based pitch detector.  Quiet voiced
   speech that VAD nearly misses still has a clear pitch peak in 70–400 Hz.
3. **Fused speech confidence** — combines VAD prob, voiced score, consonant
   score, and 1−competing prob; downstream stages (Enhancement rollback,
   Controller speech cap) read it.

The fused ``speech_conf`` is **sticky**: it can't drop below ``stickiness ×
prior_frame`` in one step.  This prevents brief Silero dips during a held
vowel from crashing speech_conf and triggering PostFilter aggressiveness
mid-utterance.

DFN3 at full strength preserves speech well, so this stage does **not**
throttle the enhancer (that was measured to cost SNR).  Instead the cues
protect the *post-filter* and supply confidence signals upstream stages
already consume.
"""

from __future__ import annotations

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage


def _voiced_score_autocorr(x: np.ndarray, sr: int,
                            pitch_lo_hz: float, pitch_hi_hz: float) -> float:
    """Cheap voicing strength via normalised autocorrelation peak in the
    pitch range.

    Returns a score in [0, 1]:
      * Strong harmonic / voiced signal → close to 1.0
      * Unvoiced / noise → close to 0.0

    Algorithm: compute autocorrelation up to ``sr / pitch_lo_hz`` lag samples;
    take the maximum in the lag range corresponding to ``[pitch_hi_hz,
    pitch_lo_hz]``; normalise by the zero-lag value.  Strong voicing gives a
    peak ratio > 0.4; noise gives < 0.2.
    """
    n = len(x)
    if n < 64 or sr <= 0:
        return 0.0
    lag_min = max(1, int(sr / pitch_hi_hz))
    lag_max = min(n - 1, int(sr / pitch_lo_hz))
    if lag_max <= lag_min:
        return 0.0
    # Pre-zero-mean to avoid DC bias.
    x = x - float(np.mean(x))
    # Truncated autocorrelation (compute only the range we need).
    zero = float(np.dot(x, x)) + 1e-12
    seg_a = x[: n - lag_min]
    # Vectorised correlation: for each lag in [lag_min, lag_max], dot product.
    # For 960-sample @ 48 kHz: lag range ~120-685, ~566 lags. Total ~500k mults.
    # ~0.3 ms in NumPy — acceptable.
    lags = np.arange(lag_min, lag_max + 1)
    ac = np.empty(len(lags), dtype=np.float64)
    for i, lag in enumerate(lags):
        ac[i] = float(np.dot(x[: n - lag], x[lag:]))
    peak = float(ac.max()) / zero
    # Map [0.3, 0.7] peak range → [0, 1] voiced score.  Below 0.3 = unvoiced;
    # above 0.7 = strong voicing.
    return float(np.clip((peak - 0.3) / 0.4, 0.0, 1.0))


class SpeechPreservation(Stage):
    name = "speech_preservation"

    def __init__(self, cfg: PipelineConfig, detect_threshold: float = 0.3) -> None:
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.detect_threshold = detect_threshold
        # Initialise prior-RMS to a typical small value (NOT 1e-6) so the very
        # first frame doesn't see an apparent 1e5× onset and false-fire plosive.
        self._prev_rms = 0.01
        self._hf_cut = 3500.0
        # Sticky speech_conf — track the previous frame's value.
        self._prev_speech_conf = 0.0

    def reset(self) -> None:
        self._prev_rms = 0.01
        self._prev_speech_conf = 0.0

    def _hf_ratio(self, x: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / self.sr)
        tot = float(spec.sum()) + 1e-12
        hf = float(spec[freqs >= self._hf_cut].sum())
        return hf / tot

    def process(self, ctx: FrameContext) -> FrameContext:
        x = ctx.audio
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))

        # ── Consonant cues (existing) ────────────────────────────────────
        hf = self._hf_ratio(x)
        onset = rms / (self._prev_rms + 1e-9)
        fricative = float(np.clip((hf - 0.25) / 0.35, 0.0, 1.0))
        plosive = float(np.clip((onset - 2.0) / 4.0, 0.0, 1.0))
        consonant = float(max(fricative, plosive))
        self._prev_rms = rms

        # ── Pitch-based voicing score (V2) ──────────────────────────────
        # Catches quiet voiced speech that VAD nearly misses and rules out
        # un-pitched "speech-shaped" noise (TV laughter, dog bark mimicking).
        voiced = _voiced_score_autocorr(
            x, self.sr, self.cfg.pitch_lo_hz, self.cfg.pitch_hi_hz
        )
        ctx.voiced_score = voiced

        # ── Whisper-specific cue (V2) ───────────────────────────────────
        # Whisper is **unvoiced** speech with HF-dominant spectrum, sustained
        # over many frames (unlike a transient click).  Signature:
        #   * high HF ratio (whisper energy concentrates above 2–3 kHz),
        #   * low voicing (no clear pitch peak),
        #   * non-silent (rules out room noise),
        #   * fricative-class consonant cue may be high (sustained /sh/-like)
        # When detected we treat the frame as confident speech and emit a
        # meta hint; Controller halves the DFN3 attenuation cap so the
        # network doesn't aggressively gate the (already-quiet) signal.
        silent = rms < 0.005
        whisper_score = 0.0
        if not silent and voiced < 0.25 and hf > 0.40:
            # Map (hf, voiced) into a whisper score.  Higher HF + lower
            # voicing → stronger whisper evidence.
            whisper_score = float(np.clip(
                (hf - 0.40) / 0.30 * (1.0 - voiced),
                0.0, 1.0,
            ))
        ctx.meta["whisper_score"] = whisper_score

        # ── Fused speech confidence ──────────────────────────────────────
        # Weights chosen so VAD dominates when confident, but voicing alone
        # can lift speech_conf above the 0.6 rollback threshold for quiet
        # voiced speech that Silero missed.
        comp_prob = ctx.noise_probs.get("competing_speech", 0.0) if ctx.noise_probs else 0.0
        raw = (
            0.55 * float(ctx.vad_prob)
            + 0.20 * voiced
            + 0.15 * consonant
            + 0.10 * (1.0 - float(comp_prob))
        )
        # Whisper boost: when the whisper detector fires strongly, ensure
        # speech_conf can clear the 0.6 rollback threshold even when VAD
        # missed (whispers are unvoiced and Silero is weaker on them).
        if whisper_score > 0.5:
            raw = max(raw, 0.65)
        # Stickiness: current frame can't drop below stickiness × prior frame.
        # Prevents brief Silero/voicing dips during a held vowel from crashing
        # confidence and triggering PostFilter aggressiveness mid-utterance.
        sticky_floor = self.cfg.speech_conf_stickiness * self._prev_speech_conf
        speech_conf = float(np.clip(max(raw, sticky_floor), 0.0, 1.0))
        ctx.speech_conf = speech_conf
        self._prev_speech_conf = speech_conf

        # ── Consonant protection (post-filter only — DFN3 is not throttled) ─
        if consonant >= self.detect_threshold:
            ctx.is_speech = True
            ctx.postfilter_strength *= (1.0 - 0.6 * consonant)
        ctx.meta["consonant"] = consonant
        ctx.meta["voiced_score"] = voiced
        ctx.meta["speech_conf"] = speech_conf
        return ctx

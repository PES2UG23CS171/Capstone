"""
Speech-preservation branch.

Enhancers are most destructive to **consonants**: fricatives (/s/, /f/, /sh/)
look like high-frequency hiss, and plosives (/p/, /t/, /k/) look like transient
clicks — exactly the things a noise suppressor wants to remove.  Low-volume
speech is also easily mistaken for noise.

This stage runs *after* the controller.  Note: DFN3 at full strength already
preserves consonants well (corr ≈ 0.99), so — unlike a naive design — we do
**not** throttle the enhancer here (that was measured to cost SNR).  Instead the
detector *protects consonants from the post-filter*: it forces the speech state
on (so the residual gate never attenuates a fricative/plosive that VAD nearly
missed) and eases ``postfilter_strength`` on those frames.  This is where over-
attenuation actually damages consonants — right before comfort-noise/residual
cleanup — so that is where we guard.

Detection (cheap, time + light spectral):
  * **HF energy ratio** (energy above ~3.5 kHz / total) → fricative cue.
  * **Broadband onset** (sudden RMS jump) → plosive cue.
  * **Low-level speech** (speech present but quiet) → raise protection.

This is the lightweight, deployable analogue of "speech masks / speech
confidence"; a learned speech-presence-probability model can later replace the
heuristic behind the same ``protection`` output.
"""

from __future__ import annotations

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage


class SpeechPreservation(Stage):
    name = "speech_preservation"

    def __init__(self, cfg: PipelineConfig, detect_threshold: float = 0.3) -> None:
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.detect_threshold = detect_threshold
        self._prev_rms = 1e-6
        self._hf_cut = 3500.0

    def reset(self) -> None:
        self._prev_rms = 1e-6

    def _hf_ratio(self, x: np.ndarray) -> float:
        # rFFT magnitude; fraction of energy above the HF cutoff.
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / self.sr)
        tot = float(spec.sum()) + 1e-12
        hf = float(spec[freqs >= self._hf_cut].sum())
        return hf / tot

    def process(self, ctx: FrameContext) -> FrameContext:
        x = ctx.audio
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))

        # Consonant cues are computed regardless of VAD, so we can rescue
        # fricatives/plosives that VAD nearly missed at word boundaries.
        hf = self._hf_ratio(x)
        onset = rms / (self._prev_rms + 1e-9)
        fricative = np.clip((hf - 0.25) / 0.35, 0.0, 1.0)          # HF-dominant
        plosive = np.clip((onset - 2.0) / 4.0, 0.0, 1.0)           # sharp jump
        consonant = float(max(fricative, plosive))
        self._prev_rms = rms

        # ── Fused speech confidence (read downstream by Enhancement) ────
        # Combines:
        #   * VAD probability (continuous; primary signal)
        #   * Consonant evidence (catches fricatives VAD nearly missed)
        #   * Inverse of the competing-speech probability (high competing
        #     prob → less confidence that THIS is target speech)
        comp_prob = ctx.noise_probs.get("competing_speech", 0.0) if ctx.noise_probs else 0.0
        speech_conf = (
            0.65 * float(ctx.vad_prob)
            + 0.25 * consonant
            + 0.10 * (1.0 - float(comp_prob))
        )
        ctx.speech_conf = float(np.clip(speech_conf, 0.0, 1.0))

        if consonant >= self.detect_threshold:
            # Protect from the post-filter: keep speech state, ease residual gate.
            ctx.is_speech = True
            ctx.postfilter_strength *= (1.0 - 0.6 * consonant)
        ctx.meta["consonant"] = consonant
        ctx.meta["speech_conf"] = ctx.speech_conf
        return ctx

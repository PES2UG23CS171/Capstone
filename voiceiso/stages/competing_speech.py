"""
Competing-speech handling — speaker-identity-conditioned suppression.

The hardest case for any noise suppressor: a *second human voice* (nearby
talker, TV dialogue, background conversation).  Generic denoisers keep it
because, spectrally, it looks exactly like the speech we want.

V2 integrates with :class:`SpeakerEmbedder`:
* When a target speaker has been **enrolled**, the embedder produces
  ``ctx.target_speaker_sim`` ∈ [-1, 1] (cosine similarity to enrolled
  x-vector).  A score near 1 → "this is the target"; near 0 or below → "this
  is a competing speaker".
* When the score is low *while voice is present* (``vad_prob`` clearly above
  noise), we flag competing-speech and emit a mid-band gain hint.  The
  controller picks that up and asks the :class:`MultiBandGainModulator` to
  attenuate 300 Hz – 4 kHz (where speech information concentrates).

When no enrollment exists, the stage degrades to the V1 behaviour (heuristic
classifier's ``competing_speech`` label) — non-target speech still gets
flagged via the noise class.
"""

from __future__ import annotations

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage


class CompetingSpeech(Stage):
    name = "competing_speech"

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.enabled = True
        self.sim_threshold = float(cfg.speaker_sim_threshold)
        self._target_embedding: np.ndarray | None = None   # set by enroll()
        # Hysteresis: a single low-similarity frame shouldn't open the gate.
        self._low_sim_run = 0

    def enroll(self, embedding: np.ndarray) -> None:
        """Register the primary speaker's d-vector (from a speaker encoder)."""
        self._target_embedding = embedding / (np.linalg.norm(embedding) + 1e-9)

    def reset(self) -> None:
        self._low_sim_run = 0

    def process(self, ctx: FrameContext) -> FrameContext:
        sim = float(ctx.target_speaker_sim)
        # When sim == 1.0 (passthrough — no enrolled vector / no model) we rely
        # solely on the heuristic noise-class flag.  When sim is meaningful
        # (< 1.0 after embedder ran), we use it as the primary signal.
        meaningful_sim = sim < 0.999

        # Per-frame raw flag.
        competing_now = False
        if meaningful_sim and ctx.vad_prob >= 0.4 and sim < self.sim_threshold:
            competing_now = True
        elif ctx.noise_class == "competing_speech" or \
                ctx.noise_probs.get("competing_speech", 0.0) > 0.4:
            competing_now = True

        # Hysteresis: require 2 consecutive frames below threshold before
        # opening the mid-band attenuation gate, so a single noisy similarity
        # doesn't chop a target-speaker frame.
        self._low_sim_run = self._low_sim_run + 1 if competing_now else 0
        gated = self._low_sim_run >= 2

        if gated:
            ctx.meta["competing_speech"] = 1.0
            # Hint for the controller: how aggressively to cut the mid band.
            # Stronger cut the lower the similarity (when we have similarity).
            if meaningful_sim:
                # Map sim ∈ [-0.2, threshold] → cut_strength ∈ [1.0, 0.0].
                cut = float(np.clip(
                    (self.sim_threshold - sim) / max(self.sim_threshold + 0.2, 1e-3),
                    0.0, 1.0,
                ))
            else:
                cut = 0.6   # classifier-only fallback: moderate cut
            ctx.meta["competing_cut"] = cut
        else:
            ctx.meta["competing_speech"] = 0.0
            ctx.meta["competing_cut"] = 0.0
        return ctx

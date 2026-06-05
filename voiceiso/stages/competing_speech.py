"""
Competing-speech handling (target-speaker extraction).

The hardest case for any suppressor: a *second human voice* (nearby talker, TV
dialogue, background conversation).  Generic denoisers keep it because it looks
exactly like the speech we want.  The fix is **target-speaker extraction**:
condition enhancement on an embedding of the enrolled (primary) speaker and keep
only voice that matches.

Feasibility / capstone verdict
------------------------------
* **Full blind speech separation** (SepFormer, MossFormer2) — NOT CPU-real-time.
  Out of scope.
* **Target-speaker extraction, streaming** — feasible on CPU.  The right
  reference is **VoiceFilter-Lite** (Google, on-device): a speaker d-vector
  conditions a lightweight mask network, ~2 MB, designed for streaming ASR
  front-ends.  Realistic as a **stretch goal**, not the core deliverable.

Recommended design (behind this interface)
------------------------------------------
1. **Enrollment**: capture ~5–10 s of the primary speaker; compute a d-vector
   with a small ECAPA-TDNN/ResNet speaker encoder exported to ONNX.
2. **Detection**: per frame, compare a running embedding of the active voice to
   the enrolled d-vector (cosine).  Low similarity while speech is present →
   ``competing_speech``.
3. **Extraction**: feed the d-vector as a conditioning input to the enhancement
   mask (VoiceFilter-Lite-style), or, as a cheaper interim, raise suppression /
   post-filter aggressiveness on non-target speech via the controller.

CPU: encoder ~3–6 ms on enrollment (one-off) + ~1 ms/frame similarity; mask
conditioning adds ~10–20 % to enhancement.  Latency: none beyond enhancement.

This stage currently only *flags* likely competing speech (using the heuristic
classifier's label) so the controller can react; the extraction network is the
documented next step.
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
        self._target_embedding: np.ndarray | None = None   # set by enroll()

    def enroll(self, embedding: np.ndarray) -> None:
        """Register the primary speaker's d-vector (from a speaker encoder)."""
        self._target_embedding = embedding / (np.linalg.norm(embedding) + 1e-9)

    def reset(self) -> None:
        pass

    def process(self, ctx: FrameContext) -> FrameContext:
        # Interim behaviour: trust the classifier's competing_speech flag and
        # mark it so the controller can push post-filter aggressiveness.  When a
        # speaker encoder is wired in, replace this with d-vector similarity.
        if ctx.noise_class == "competing_speech" or ctx.noise_probs.get("competing_speech", 0) > 0.4:
            ctx.meta["competing_speech"] = 1.0
            if self._target_embedding is None:
                # No enrollment → can't safely extract; flag only.
                ctx.meta["competing_speech_action"] = 0.0
        else:
            ctx.meta["competing_speech"] = 0.0
        return ctx

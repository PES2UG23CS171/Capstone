"""
Stage contract + the per-frame context that flows through the pipeline.

Every subsystem (preprocessing, AEC, VAD, classification, enhancement,
post-filter, …) is a :class:`Stage` that mutates/annotates a :class:`FrameContext`.
A stage is *streaming*: it sees one hop-sized frame at a time and keeps its own
internal state across calls.  This is what makes the whole pipeline real-time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class FrameContext:
    """Mutable state carried through the pipeline for a single frame (hop)."""

    # Audio (float32, mono, ±1).  Stages read/replace ``audio`` in place.
    audio: np.ndarray
    sample_rate: int

    # Optional far-end reference for AEC (loopback of what the speaker plays).
    reference: Optional[np.ndarray] = None

    # ── Analysis signals filled in by stages ─────────────────────────────
    vad_prob: float = 0.0              # P(speech) ∈ [0, 1]
    is_speech: bool = False            # thresholded + hangover-smoothed
    noise_class: str = "clean"         # top noise label this frame
    noise_probs: Dict[str, float] = field(default_factory=dict)
    snr_db: float = 0.0                # instantaneous estimate
    environment: str = "unknown"       # coarse env tag (quiet/office/outdoor/…)

    # ── Control signals (set by the dynamic controller) ──────────────────
    suppression: float = 1.0           # 0 = dry, 1 = full wet
    postfilter_strength: float = 1.0

    # ── Diagnostics ──────────────────────────────────────────────────────
    meta: Dict[str, float] = field(default_factory=dict)

    def copy_audio(self) -> np.ndarray:
        return self.audio.copy()


class Stage:
    """Base class for a streaming pipeline stage.

    Subclasses override :meth:`process` (and optionally :meth:`reset`).  Stages
    must be cheap and allocation-light on the hot path.
    """

    name: str = "stage"
    enabled: bool = True

    def process(self, ctx: FrameContext) -> FrameContext:  # pragma: no cover - abstract
        """Consume + annotate ``ctx`` for one frame and return it."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any streaming state (call on stream (re)start)."""

    def __call__(self, ctx: FrameContext) -> FrameContext:
        if not self.enabled:
            return ctx
        return self.process(ctx)

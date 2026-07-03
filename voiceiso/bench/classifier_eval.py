"""
Noise-classifier evaluation harness.

Distinct from ``benchmark.py`` (which measures *enhancement* quality): this
scores the *classification* accuracy of a noise classifier against labelled
clips, reporting per-class precision / recall / F1, macro-F1, and a confusion
matrix over the 12 target labels.

It is classifier-agnostic — it drives any object with the ``Stage`` interface
that writes ``ctx.meta["noise_label"]`` (EfficientAT) or ``ctx.noise_class``
(heuristic, mapped back to a target label).  This is the piece that turns
"we have no evaluation results" into a reproducible number.

Typical use (after a model exists)::

    from voiceiso.bench.classifier_eval import evaluate_classifier, EvalClip
    clips = load_fsd50k_eval(...)            # [(audio, sr, {target labels})]
    report = evaluate_classifier(classifier, clips)
    print(report.render())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from voiceiso.stages.base import FrameContext

# Internal class → target label inverse (for classifiers that only set
# ctx.noise_class, e.g. the heuristic).  Mirrors _TARGET_TO_INTERNAL.
_INTERNAL_TO_TARGET = {
    "clean": "clean", "fan": "fan", "hvac": "hvac", "traffic": "traffic",
    "keyboard": "keyboard", "dog_bark": "dog", "wind": "wind", "music": "music",
    "television": "tv", "competing_speech": "competing_speech", "other": "other",
    # heuristic-only internal classes that have no distinct target → map sensibly
    "mouse_click": "keyboard", "door_slam": "other",
}


@dataclass
class EvalClip:
    audio: np.ndarray
    sr: int
    labels: Set[str]            # ground-truth target labels (≥1)


@dataclass
class ClassifierEvalReport:
    labels: Sequence[str]
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)  # P/R/F1/support
    macro_f1: float = 0.0
    micro_acc: float = 0.0      # top-1 hit rate (predicted ∈ ground-truth set)
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    n_clips: int = 0

    def render(self) -> str:
        lines = ["=" * 62, "  NOISE-CLASSIFIER EVALUATION", "=" * 62,
                 f"  clips: {self.n_clips}   top-1 hit-rate: {self.micro_acc:.3f}   "
                 f"macro-F1: {self.macro_f1:.3f}", "-" * 62,
                 f"  {'label':<18}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}"]
        for lab in self.labels:
            m = self.per_class.get(lab)
            if not m or m["support"] == 0:
                continue
            lines.append(f"  {lab:<18}{m['precision']:>7.3f}{m['recall']:>7.3f}"
                         f"{m['f1']:>7.3f}{int(m['support']):>9}")
        lines.append("=" * 62)
        return "\n".join(lines)


def _predict_label(classifier, clip: EvalClip, block: int = 960) -> Optional[str]:
    """Stream a clip through a classifier and return its top target label.

    Uses the classifier's own streaming state; the prediction is the final
    ``noise_label`` (EfficientAT) or the target-mapped ``noise_class``.
    """
    classifier.reset()
    audio = clip.audio.astype(np.float32)
    last_label = None
    # Feed only full blocks so the classifier sees a constant block size
    # (matches the live 20 ms cadence); drop any trailing partial.
    n_full = (len(audio) // block) * block
    for s in range(0, n_full, block):
        seg = audio[s:s + block]
        ctx = FrameContext(audio=seg.copy(), sample_rate=clip.sr)
        ctx.meta = {}
        ctx = classifier.process(ctx)
        # Prefer the explicit target label (EfficientAT); else map internal class.
        if "noise_label" in ctx.meta:
            last_label = ctx.meta["noise_label"]
        else:
            last_label = _INTERNAL_TO_TARGET.get(ctx.noise_class, "other")
    return last_label


def evaluate_classifier(classifier, clips: Iterable[EvalClip],
                        labels: Optional[Sequence[str]] = None,
                        block: int = 960) -> ClassifierEvalReport:
    """Score ``classifier`` over labelled ``clips``.

    ``clips`` may be a list OR a lazy generator (loaded one clip at a time) —
    this function counts clips during iteration, so it never calls ``len()``
    and keeps memory O(1) for large eval sets.

    Top-1 protocol: the classifier emits one dominant label per clip; a hit is
    when that label is in the clip's ground-truth label set.  Per-class P/R/F1
    treat each target label as a one-vs-rest binary decision (predicted-top ==
    label  vs  label ∈ ground-truth).
    """
    if labels is None:
        from voiceiso.stages.noise_classifier import _TARGET_LABELS
        labels = list(_TARGET_LABELS)
    labset = list(labels)

    tp = {l: 0 for l in labset}
    fp = {l: 0 for l in labset}
    fn = {l: 0 for l in labset}
    support = {l: 0 for l in labset}
    confusion = {t: {p: 0 for p in labset} for t in labset}
    hits = 0
    n_clips = 0

    for clip in clips:
        n_clips += 1
        pred = _predict_label(classifier, clip, block=block) or "clean"
        truth = clip.labels or {"clean"}
        # top-1 hit
        if pred in truth:
            hits += 1
        # one-vs-rest tallies
        for l in labset:
            in_truth = l in truth
            is_pred = (pred == l)
            if in_truth:
                support[l] += 1
            if is_pred and in_truth:
                tp[l] += 1
            elif is_pred and not in_truth:
                fp[l] += 1
            elif (not is_pred) and in_truth:
                fn[l] += 1
        # confusion: charge the prediction against each true label
        primary_truth = sorted(truth)[0]
        confusion[primary_truth][pred] = confusion[primary_truth].get(pred, 0) + 1

    per_class: Dict[str, Dict[str, float]] = {}
    f1s = []
    for l in labset:
        p = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 0.0
        r = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[l] = {"precision": p, "recall": r, "f1": f1, "support": float(support[l])}
        if support[l] > 0:
            f1s.append(f1)

    return ClassifierEvalReport(
        labels=labset,
        per_class=per_class,
        macro_f1=float(np.mean(f1s)) if f1s else 0.0,
        micro_acc=hits / max(n_clips, 1),
        confusion=confusion,
        n_clips=n_clips,
    )

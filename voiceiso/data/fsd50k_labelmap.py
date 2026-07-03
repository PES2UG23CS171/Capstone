"""
FSD50K vocabulary → voiceiso 12-label target taxonomy.

FSD50K ships 200 labels (an AudioSet-ontology subset), one or more per clip,
in its ``dev.csv`` / ``eval.csv`` ground-truth files (comma-separated label
names in the ``labels`` column).  This module maps those names onto the 12
approved target labels so FSD50K can be used to *evaluate* (and optionally
train a head for) the EfficientAT classifier.

Target taxonomy (must match noise_classifier._TARGET_LABELS):
    clean, fan, hvac, traffic, keyboard, dog, wind,
    music, tv, speech, competing_speech, other

Coverage honesty
----------------
FSD50K is event-tagged Freesound audio, so several of our buckets are only
partially covered or absent.  ``coverage_report()`` documents this:

    well covered : keyboard, dog, traffic, wind, music, speech, competing_speech, other
    weak / absent: fan, hvac, tv, clean

Mitigations for the uncovered buckets (used by the evaluation plan, NOT by this
map): source ``fan``/``hvac`` from DEMAND/MUSAN, ``tv`` from a speech+music mix
or DynamicMixer, and ``clean`` from DNS5 clean speech / silent segments.
"""

from __future__ import annotations

from typing import Dict, List, Set

# FSD50K label name (exact, underscored as in dev.csv) → target label.
# Names not present here fall to "other" if they are noise-like, or are dropped
# for buckets we can't source from FSD50K (handled by the caller).
_FSD50K_TO_TARGET: Dict[str, str] = {
    # ── keyboard ──────────────────────────────────────────────────────────
    "Computer_keyboard": "keyboard",
    "Typing": "keyboard",
    "Typewriter": "keyboard",
    # ── dog ───────────────────────────────────────────────────────────────
    "Dog": "dog",
    "Bark": "dog",
    "Growling": "dog",
    "Bow-wow": "dog",
    # ── traffic ───────────────────────────────────────────────────────────
    "Traffic_noise_and_roadway_noise": "traffic",
    "Car": "traffic",
    "Car_passing_by": "traffic",
    "Truck": "traffic",
    "Motorcycle": "traffic",
    "Bus": "traffic",
    "Engine": "traffic",
    "Motor_vehicle_(road)": "traffic",
    "Vehicle_horn_and_car_horn_and_honking": "traffic",
    # ── wind ──────────────────────────────────────────────────────────────
    "Wind": "wind",
    "Wind_noise_(microphone)": "wind",
    # ── music ─────────────────────────────────────────────────────────────
    "Music": "music",
    "Musical_instrument": "music",
    "Guitar": "music",
    "Piano": "music",
    "Drum": "music",
    "Plucked_string_instrument": "music",
    "Keyboard_(musical)": "music",
    "Percussion": "music",
    "Singing": "music",
    "Bowed_string_instrument": "music",
    "Brass_instrument": "music",
    "Wind_instrument_and_woodwind_instrument": "music",
    # ── speech (single talker) ──────────────────────────────────────────────
    "Speech": "speech",
    "Male_speech_and_man_speaking": "speech",
    "Female_speech_and_woman_speaking": "speech",
    "Child_speech_and_kid_speaking": "speech",
    # ── competing_speech (multi-talker / babble) ────────────────────────────
    "Conversation": "competing_speech",
    "Babbling": "competing_speech",
    "Crowd": "competing_speech",
    "Chatter": "competing_speech",
    "Hubbub_and_speech_noise_and_speech_babble": "competing_speech",
    # ── fan (weak coverage — FSD50K rarely tags these) ───────────────────────
    "Mechanical_fan": "fan",
    # ── other (real noise with no specific bucket) ──────────────────────────
    "Clatter": "other",
    "Glass": "other",
    "Water": "other",
    "Rain": "other",
    "Thunderstorm": "other",
    "Tools": "other",
    "Power_tool": "other",
    "Mechanisms": "other",
    "Tap": "other",
    "Sink_(filling_or_washing)": "other",
}

# Buckets FSD50K covers poorly or not at all — eval/train must source elsewhere.
UNCOVERED_BY_FSD50K: Set[str] = {"hvac", "tv", "clean"}
WEAK_IN_FSD50K: Set[str] = {"fan"}


def fsd50k_labels_to_target(label_field: str) -> Set[str]:
    """Map an FSD50K clip's comma-separated ``labels`` field to target labels.

    Returns the set of target labels the clip carries (a clip can be multi-label,
    e.g. "Speech,Music" → {speech, music}).  FSD50K labels with no mapping are
    ignored (they belong to buckets we don't evaluate from FSD50K).
    """
    out: Set[str] = set()
    for name in label_field.split(","):
        name = name.strip()
        tgt = _FSD50K_TO_TARGET.get(name)
        if tgt is not None:
            out.add(tgt)
    return out


def coverage_report() -> str:
    """Human-readable summary of which target buckets FSD50K can evaluate."""
    covered: Dict[str, int] = {}
    for tgt in _FSD50K_TO_TARGET.values():
        covered[tgt] = covered.get(tgt, 0) + 1
    lines = ["FSD50K → target coverage (count of source FSD50K labels per bucket):"]
    from voiceiso.stages.noise_classifier import _TARGET_LABELS
    for t in _TARGET_LABELS:
        n = covered.get(t, 0)
        tag = ""
        if t in UNCOVERED_BY_FSD50K:
            tag = "  ← NOT in FSD50K (source elsewhere)"
        elif t in WEAK_IN_FSD50K:
            tag = "  ← weak coverage"
        lines.append(f"  {t:18s}: {n} source labels{tag}")
    return "\n".join(lines)

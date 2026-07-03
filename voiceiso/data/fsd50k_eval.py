"""
FSD50K eval-split loader for classifier evaluation.

Reads the downloaded FSD50K layout::

    dataset/FSD50K/FSD50K.eval_audio/{fname}.wav        (44.1 kHz mono PCM16)
    dataset/FSD50K/FSD50K.ground_truth/eval.csv         (fname, labels, mids)

and yields :class:`~voiceiso.bench.classifier_eval.EvalClip` objects lazily
(one decoded clip at a time → O(1) memory), with FSD50K label names mapped to
the 12 target labels via :mod:`voiceiso.data.fsd50k_labelmap`.

Clips with no mapped target label are skipped (they belong to FSD50K classes
outside our taxonomy).  ``clean``/``hvac``/``tv`` have no FSD50K source and so
never appear here — that is expected and documented.
"""

from __future__ import annotations

import csv
from math import gcd
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

from voiceiso.bench.classifier_eval import EvalClip
from voiceiso.data.fsd50k_labelmap import fsd50k_labels_to_target


def _load_resampled(path: Path, sr_out: int, max_samples: int) -> Optional[np.ndarray]:
    """Load a mono clip, resample to ``sr_out``, truncate to ``max_samples``."""
    try:
        data, sr_in = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return None
    x = data.mean(axis=1)
    if sr_in != sr_out:
        from scipy.signal import resample_poly
        g = gcd(sr_in, sr_out)
        x = resample_poly(x, sr_out // g, sr_in // g).astype("float32")
    if max_samples and len(x) > max_samples:
        x = x[:max_samples]
    return x.astype("float32")


def iter_fsd50k_eval(root: str = "dataset/FSD50K",
                     sr: int = 48_000,
                     clip_seconds: float = 6.0,
                     max_per_class: Optional[int] = 120,
                     seed: int = 0,
                     only_fnames: Optional[set] = None) -> Iterator[EvalClip]:
    """Yield EvalClip objects from the FSD50K eval split (lazy / O(1) memory).

    Parameters
    ----------
    root          : FSD50K root containing FSD50K.eval_audio + FSD50K.ground_truth
    sr            : pipeline sample rate to resample clips to (match cfg.sample_rate)
    clip_seconds  : truncate each clip to this length (the 200 ms classifier only
                    needs a few seconds to settle; bounds time + memory)
    max_per_class : cap clips per target label for a balanced, fast eval
                    (None = use all).  Counted by each of the clip's targets.
    seed          : RNG seed for the per-class subsample order
    """
    if sf is None:
        raise ImportError("soundfile is required for FSD50K evaluation")
    root_p = Path(root)
    audio_dir = root_p / "FSD50K.eval_audio"
    gt = root_p / "FSD50K.ground_truth" / "eval.csv"
    if not audio_dir.is_dir() or not gt.exists():
        raise FileNotFoundError(
            f"FSD50K eval not found under {root!r} "
            f"(expected {audio_dir} and {gt})"
        )

    max_samples = int(clip_seconds * sr)

    # First pass: read CSV, map labels, build a balanced selection plan without
    # decoding any audio.
    rows: List[tuple[str, set]] = []
    for row in csv.DictReader(open(gt, newline="")):
        if only_fnames is not None and row["fname"] not in only_fnames:
            continue
        tgts = fsd50k_labels_to_target(row["labels"])
        if tgts:
            rows.append((row["fname"], tgts))
    # When restricted to an explicit fname set (held-out test split), use ALL
    # of them — don't sub-sample per class.
    if only_fnames is not None:
        max_per_class = None

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    per_class_count: Dict[str, int] = {}
    selected: List[int] = []
    for idx in order:
        _fname, tgts = rows[idx]
        if max_per_class is None:
            selected.append(int(idx))
            continue
        # Keep the clip if any of its targets is still under the cap.
        if any(per_class_count.get(t, 0) < max_per_class for t in tgts):
            selected.append(int(idx))
            for t in tgts:
                per_class_count[t] = per_class_count.get(t, 0) + 1

    # Second pass: decode + yield lazily, in selection order.
    for idx in selected:
        fname, tgts = rows[idx]
        audio = _load_resampled(audio_dir / f"{fname}.wav", sr, max_samples)
        if audio is None or len(audio) < sr // 10:   # skip <100 ms / unreadable
            continue
        yield EvalClip(audio=audio, sr=sr, labels=set(tgts))


def fsd50k_eval_available(root: str = "dataset/FSD50K") -> bool:
    root_p = Path(root)
    return (root_p / "FSD50K.eval_audio").is_dir() and \
           (root_p / "FSD50K.ground_truth" / "eval.csv").exists()


def load_uploader_map(root: str = "dataset/FSD50K", split: str = "eval") -> Dict[str, str]:
    """Map ``fname`` → Freesound ``uploader`` from FSD50K's clip-info metadata.

    FSD50K's official dev/eval partition is grouped by uploader/source so the
    same recording session never straddles train and test.  This map lets the
    splitter (and the eval harness) reproduce that grouping — without it, a
    random per-clip split leaks ~84% of held-out clips' sources into training.
    Returns ``{}`` if the metadata file is absent.
    """
    import json
    name = f"{split}_clips_info_FSD50K.json"
    p = Path(root) / "FSD50K.metadata" / name
    if not p.exists():
        return {}
    info = json.load(open(p))
    return {str(k): str(v.get("uploader", "")) for k, v in info.items()}

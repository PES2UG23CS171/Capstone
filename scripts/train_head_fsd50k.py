"""
Train a lightweight native 12-class head on the FROZEN EfficientAT backbone.

Contingency for the rejected pretrained-direct path (macro-F1 0.119): the
backbone's AudioSet posteriors are dominated by Speech/Music and never surface
fan/keyboard/dog/wind.  A small head trained on the backbone's 960-dim
embedding, targeting OUR 12 labels directly, fixes that.

Pipeline
--------
  1. Map FSD50K clips → 12 target labels (voiceiso.data.fsd50k_labelmap).
  2. Split clips DISJOINTLY into train / val / test (by fname).
     * If FSD50K.dev_audio is present → train/val from dev, test from eval
       (standard protocol).
     * If only eval_audio is present (current machine) → split the eval pool
       disjointly.  Non-standard but a valid feasibility experiment; flagged.
  3. Extract frozen backbone embeddings (960-dim) on 4 s windows
     (the backbone needs ~4 s context — verified; shorter windows saturate).
  4. Train a linear/MLP head with class-weighted multi-label BCE; early-stop
     on val macro-F1.
  5. Save head weights + label order + the split (esp. test fnames) so the
     export + eval steps use the identical held-out test set.

The backbone stays frozen throughout (no gradients into it).

Run from the EfficientAT clone dir with repo root on PYTHONPATH, e.g.:

    cd third_party/EfficientAT
    PYTHONPATH="<repo>:$PWD" python <repo>/scripts/train_head_fsd50k.py \\
        --root <repo>/dataset/FSD50K \\
        --out  <repo>/checkpoints/head_12.pt \\
        --split-out <repo>/checkpoints/fsd50k_split.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np

_TARGETS = ("clean", "fan", "hvac", "traffic", "keyboard", "dog", "wind",
            "music", "tv", "speech", "competing_speech", "other")
# Trainable buckets = those FSD50K can label (clean/hvac/tv have no source).
_TRAINABLE = ("fan", "traffic", "keyboard", "dog", "wind", "music",
              "speech", "competing_speech", "other")


def _load_32k(path: Path, sr_out: int, win_samples: int, np_):
    import soundfile as sf
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return None
    x = data.mean(axis=1)
    if sr != sr_out:
        from scipy.signal import resample_poly
        g = gcd(sr, sr_out)
        x = resample_poly(x, sr_out // g, sr // g).astype("float32")
    # Tile short clips up to the window, then take the first win_samples
    # (matches AugmentMelSTFT inference tiling).
    if len(x) < win_samples:
        reps = int(np_.ceil(win_samples / max(len(x), 1)))
        x = np_.tile(x, reps)
    return x[:win_samples].astype("float32")


def _read_mapped_clips(gt_csv: Path):
    from voiceiso.data.fsd50k_labelmap import fsd50k_labels_to_target
    rows = []
    for row in csv.DictReader(open(gt_csv, newline="")):
        tgts = fsd50k_labels_to_target(row["labels"])
        tgts = {t for t in tgts if t in _TRAINABLE}
        if tgts:
            rows.append((row["fname"], sorted(tgts)))
    return rows


def _grouped_split(rows, uploader_of, val_frac, test_frac, rng):
    """Uploader-grouped split: assign WHOLE Freesound uploaders to test/val/train
    so no recording source (uploader) straddles two splits.

    This is the fix for the data-leakage finding — a plain per-clip shuffle let
    ~84% of held-out clips share an uploader (same mic/room/session, often the
    same long source chopped into clips) with training.  ``rows`` is a list of
    (fname, targets); returns (train_rows, val_rows, test_rows) as the same
    tuples.  Deterministic given ``rng``.
    """
    from collections import defaultdict
    by_up = defaultdict(list)
    for fn, t in rows:
        by_up[uploader_of.get(fn, f"__nouploader__{fn}")].append((fn, t))
    uploaders = sorted(by_up.keys())
    rng.shuffle(uploaders)
    n_total = len(rows)
    n_test_target = int(n_total * test_frac)
    n_val_target = int(n_total * val_frac)
    test, val, train = [], [], []
    for up in uploaders:
        clips = by_up[up]
        if len(test) < n_test_target:
            test += clips
        elif len(val) < n_val_target:
            val += clips
        else:
            train += clips
    return train, val, test


def main() -> int:
    ap = argparse.ArgumentParser(description="Train 12-class head on frozen EfficientAT")
    ap.add_argument("--root", default="dataset/FSD50K")
    ap.add_argument("--out", default="checkpoints/head_12.pt")
    ap.add_argument("--split-out", default="checkpoints/fsd50k_split.json")
    ap.add_argument("--cache", default="checkpoints/fsd50k_emb_cache.npz")
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--max-per-class", type=int, default=180,
                    help="cap clips per class for train (rare classes use all)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=0, help="0 = linear probe; >0 = MLP hidden dim")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from models.mn.model import get_model
    from voiceiso.stages._efficientat_mel import EfficientATMel

    root = Path(args.root)
    dev_audio = root / "FSD50K.dev_audio"
    eval_audio = root / "FSD50K.eval_audio"
    gt = root / "FSD50K.ground_truth"
    win_samples = int(args.window_s * args.sr)
    rng = np.random.default_rng(args.seed)

    # ── Build split (UPLOADER-GROUPED — no source straddles two splits) ──
    from voiceiso.data.fsd50k_eval import load_uploader_map
    if dev_audio.is_dir() and any(dev_audio.iterdir()):
        mode = "standard grouped (dev→train/val by uploader, eval→test)"
        dev_rows = _read_mapped_clips(gt / "dev.csv")
        up_dev = load_uploader_map(str(root), "dev")
        train, val, _ = _grouped_split(dev_rows, up_dev, args.val_frac, 0.0, rng)
        train_rows = [(dev_audio, f, t) for f, t in train]
        val_rows = [(dev_audio, f, t) for f, t in val]
        test_rows = [(eval_audio, f, t) for f, t in _read_mapped_clips(gt / "eval.csv")]
    else:
        mode = "eval-only grouped split (dev_audio ABSENT) — uploader-disjoint feasibility"
        rows = _read_mapped_clips(gt / "eval.csv")
        up_eval = load_uploader_map(str(root), "eval")
        if not up_eval:
            print("  WARNING: no uploader metadata found — split will NOT be "
                  "uploader-disjoint (per-clip fallback). Provide "
                  "FSD50K.metadata/eval_clips_info_FSD50K.json to fix leakage.")
        train, val, test = _grouped_split(rows, up_eval, args.val_frac, args.test_frac, rng)
        train_rows = [(eval_audio, f, t) for f, t in train]
        val_rows = [(eval_audio, f, t) for f, t in val]
        test_rows = [(eval_audio, f, t) for f, t in test]

    print(f"split mode: {mode}")
    print(f"  train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    # Persist the split (test fnames are what the eval step must use).  We now
    # also persist train_fnames so the eval harness can verify/enforce
    # uploader-disjointness against training.
    Path(args.split_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "mode": mode, "window_s": args.window_s, "sr": args.sr,
        "labels": list(_TARGETS), "trainable": list(_TRAINABLE),
        "grouped_by": "uploader",
        "test_fnames": [f for _d, f, _t in test_rows],
        "val_fnames": [f for _d, f, _t in val_rows],
        "train_fnames": [f for _d, f, _t in train_rows],
    }, open(args.split_out, "w"), indent=0)
    print(f"  wrote split → {args.split_out}")

    # Cap train clips per class (rare classes keep all).
    per_class = {t: 0 for t in _TRAINABLE}
    capped = []
    for d, f, t in train_rows:
        if any(per_class[c] < args.max_per_class for c in t):
            capped.append((d, f, t))
            for c in t:
                per_class[c] += 1
    train_rows = capped
    print(f"  train after cap: {len(train_rows)}  per-class={per_class}")

    # ── Embedding extraction (frozen backbone) ──────────────────────────
    model = get_model(width_mult=1.0, pretrained_name="mn10_as").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mel = EfficientATMel(sr=args.sr)
    lab_idx = {t: i for i, t in enumerate(_TARGETS)}

    def embed_set(rowset, name):
        X, Y = [], []
        for i, (d, f, tgts) in enumerate(rowset):
            wav = _load_32k(d / f"{f}.wav", args.sr, win_samples, np)
            if wav is None:
                continue
            with torch.no_grad():
                sp = mel(wav)
                _logits, feats = model(sp.unsqueeze(0))
                X.append(feats.squeeze(0).numpy().astype("float32"))
            y = np.zeros(len(_TARGETS), dtype="float32")
            for t in tgts:
                y[lab_idx[t]] = 1.0
            Y.append(y)
            if (i + 1) % 200 == 0:
                print(f"    {name}: {i+1}/{len(rowset)}")
        return np.stack(X), np.stack(Y)

    print("extracting train embeddings…")
    Xtr, Ytr = embed_set(train_rows, "train")
    print("extracting val embeddings…")
    Xva, Yva = embed_set(val_rows, "val")
    np.savez(args.cache, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva)
    print(f"  embeddings: train {Xtr.shape} val {Xva.shape}  (cached → {args.cache})")

    # ── Train head (class-weighted multi-label BCE) ─────────────────────
    emb_dim = Xtr.shape[1]
    n_cls = len(_TARGETS)
    if args.hidden > 0:
        head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, args.hidden), torch.nn.ReLU(),
            torch.nn.Dropout(0.3), torch.nn.Linear(args.hidden, n_cls))
    else:
        head = torch.nn.Linear(emb_dim, n_cls)

    # pos_weight = (neg/pos) per class to counter imbalance; clamp.
    pos = Ytr.sum(0) + 1e-6
    neg = len(Ytr) - pos
    pos_weight = torch.tensor(np.clip(neg / pos, 1.0, 50.0), dtype=torch.float32)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
    Xtr_t, Ytr_t = torch.tensor(Xtr), torch.tensor(Ytr)
    Xva_t, Yva_t = torch.tensor(Xva), torch.tensor(Yva)

    def val_macro_f1():
        head.eval()
        with torch.no_grad():
            p = torch.sigmoid(head(Xva_t)).numpy()
        # top-1 protocol over TRAINABLE classes (matches runtime dominant-pick)
        ti = [lab_idx[t] for t in _TRAINABLE]
        f1s = []
        for c in ti:
            pred = (p.argmax(1) == c)
            truth = Yva[:, c] > 0.5
            tp = (pred & truth).sum(); fp = (pred & ~truth).sum(); fn = (~pred & truth).sum()
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        return float(np.mean(f1s))

    best_f1, best_state, patience, bad = -1.0, None, 40, 0
    for ep in range(args.epochs):
        head.train()
        opt.zero_grad()
        loss = crit(head(Xtr_t), Ytr_t)
        loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            f1 = val_macro_f1()
            if f1 > best_f1:
                best_f1, bad = f1, 0
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
            else:
                bad += 1
            if (ep + 1) % 50 == 0:
                print(f"  epoch {ep+1}: loss {loss.item():.4f}  val_macroF1(trainable) {f1:.3f}")
            if bad >= patience:
                print(f"  early stop at epoch {ep+1} (best val macroF1 {best_f1:.3f})")
                break
    if best_state is not None:
        head.load_state_dict(best_state)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "emb_dim": emb_dim,
                "hidden": args.hidden, "labels": list(_TARGETS),
                "window_s": args.window_s, "sr": args.sr,
                "best_val_macro_f1": best_f1}, args.out)
    print(f"saved head → {args.out}  (best val trainable-macroF1 {best_f1:.3f})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

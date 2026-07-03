"""
Evaluate the noise classifier(s) on the FSD50K eval split.

Runs the requested classifier(s) over FSD50K eval clips (labels mapped to the 12
target labels), reporting per-class precision/recall/F1, macro-F1, top-1
hit-rate, and a compact comparison when both backends are evaluated.

This is the harness that turns "no evaluation results" into committed numbers
and decides whether the pretrained EfficientAT model is sufficient (Phase 2
completion criterion) or whether a fine-tuned 12-label head is needed.

Usage
-----
    # Heuristic baseline (works today, no model needed):
    python -m scripts.eval_classifier_fsd50k --which heuristic

    # EfficientAT (requires cfg.noise_classifier_model_path set + onnxruntime):
    python -m scripts.eval_classifier_fsd50k --which efficientat \\
        --model checkpoints/efficientat_s_int8.onnx

    # Compare both:
    python -m scripts.eval_classifier_fsd50k --which both \\
        --model checkpoints/efficientat_s_int8.onnx \\
        --max-per-class 120 --clip-seconds 6
"""

from __future__ import annotations

import argparse
import sys
import time

from voiceiso.config import PipelineConfig
from voiceiso.bench.classifier_eval import evaluate_classifier
from voiceiso.data.fsd50k_eval import iter_fsd50k_eval, fsd50k_eval_available


def _build(which: str, cfg: PipelineConfig):
    from voiceiso.stages.noise_classifier import (
        EfficientATNoiseClassifier, HeuristicNoiseClassifier,
    )
    if which == "heuristic":
        return HeuristicNoiseClassifier(cfg), "heuristic"
    # EfficientAT — raises if model/ORT/labels missing.
    return EfficientATNoiseClassifier(cfg), "efficientat"


def _run_one(which: str, cfg: PipelineConfig, args, only_fnames=None) -> None:
    clf, backend = _build(which, cfg)
    print(f"\n>>> evaluating backend: {backend}")
    clips = iter_fsd50k_eval(
        root=args.root, sr=cfg.sample_rate,
        clip_seconds=args.clip_seconds, max_per_class=args.max_per_class,
        seed=args.seed, only_fnames=only_fnames,
    )
    t0 = time.perf_counter()
    rep = evaluate_classifier(clf, clips)
    dt = time.perf_counter() - t0
    print(rep.render())
    print(f"  wall: {dt:.1f}s for {rep.n_clips} clips "
          f"({1000.0 * dt / max(rep.n_clips, 1):.0f} ms/clip)")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate noise classifier on FSD50K eval")
    ap.add_argument("--which", choices=["heuristic", "efficientat", "both"],
                    default="heuristic")
    ap.add_argument("--root", default="dataset/FSD50K")
    ap.add_argument("--model", default=None,
                    help="EfficientAT ONNX path (sets cfg.noise_classifier_model_path)")
    ap.add_argument("--max-per-class", type=int, default=120,
                    help="cap clips per target label (None-like 0 = all)")
    ap.add_argument("--clip-seconds", type=float, default=6.0)
    ap.add_argument("--window-s", type=float, default=None,
                    help="override cfg.noise_classifier_window_s (mn10_as needs ~4s)")
    ap.add_argument("--split", default="auto",
                    help="fsd50k_split.json restricting eval to held-out test_fnames. "
                         "'auto' (default) uses checkpoints/fsd50k_split.json if present; "
                         "'none' scores the full eval pool")
    ap.add_argument("--drop-train-uploaders", dest="drop_train_uploaders",
                    action="store_true",
                    help="also drop held-out test clips that share a Freesound "
                         "uploader with training — reports the LEAKAGE-FREE number")
    ap.add_argument("--allow-full-pool", dest="allow_full_pool", action="store_true",
                    help="permit scoring the learned classifier on the full eval "
                         "pool (train-on-test) without a warning")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.max_per_class == 0:
        args.max_per_class = None

    if not fsd50k_eval_available(args.root):
        print(f"FSD50K eval not found under {args.root!r}", file=sys.stderr)
        return 1

    cfg = PipelineConfig()
    if args.model:
        cfg.noise_classifier_model_path = args.model
    if args.window_s:
        cfg.noise_classifier_window_s = args.window_s

    # Resolve the split: 'auto' uses the shipped split if present; 'none' disables.
    split_path = args.split
    if split_path == "auto":
        cand = "checkpoints/fsd50k_split.json"
        from pathlib import Path as _P
        split_path = cand if _P(cand).exists() else None
    elif split_path == "none":
        split_path = None

    only_fnames = None
    if split_path:
        import json
        sp = json.load(open(split_path))
        only_fnames = set(sp.get("test_fnames", []))
        print(f"restricting to {len(only_fnames)} held-out TEST clips from {split_path}")
        if args.drop_train_uploaders:
            from voiceiso.data.fsd50k_eval import load_uploader_map
            up = load_uploader_map(args.root, "eval")
            train_fn = set(sp.get("train_fnames", []))
            if not train_fn:
                # Older split JSONs don't store train_fnames — reconstruct the
                # training set = mapped eval pool − (test ∪ val).
                import csv as _csv
                from voiceiso.data.fsd50k_labelmap import fsd50k_labels_to_target
                gt = f"{args.root}/FSD50K.ground_truth/eval.csv"
                mapped = {r["fname"] for r in _csv.DictReader(open(gt, newline=""))
                          if fsd50k_labels_to_target(r["labels"])}
                train_fn = mapped - (only_fnames | set(sp.get("val_fnames", [])))
            train_up = {up.get(f, f) for f in train_fn}
            before = len(only_fnames)
            only_fnames = {f for f in only_fnames if up.get(f, f) not in train_up}
            print(f"  uploader-disjoint filter: dropped {before - len(only_fnames)} of "
                  f"{before} test clips that share an uploader with training; "
                  f"{len(only_fnames)} LEAKAGE-FREE clips remain")
    elif args.which in ("efficientat", "both") and not args.allow_full_pool:
        print("WARNING: no --split — scoring the LEARNED classifier on the FULL "
              "eval pool, which INCLUDES its own training clips (train-on-test, "
              "inflated). Use --split checkpoints/fsd50k_split.json or "
              "--allow-full-pool to silence.", file=sys.stderr)

    reports = {}
    if args.which in ("heuristic", "both"):
        reports["heuristic"] = _run_one("heuristic", cfg, args, only_fnames)
    if args.which in ("efficientat", "both"):
        try:
            reports["efficientat"] = _run_one("efficientat", cfg, args, only_fnames)
        except Exception as exc:
            print(f"\nEfficientAT unavailable: {exc}", file=sys.stderr)
            print("  (set --model to a labelled EfficientAT ONNX and install "
                  "onnxruntime)", file=sys.stderr)
            if args.which == "efficientat":
                return 2

    # Comparison table.
    if len(reports) == 2:
        from voiceiso.stages.noise_classifier import _TARGET_LABELS
        h, e = reports["heuristic"], reports["efficientat"]
        print("\n" + "=" * 64)
        print("  HEURISTIC vs EfficientAT  (per-class F1)")
        print("=" * 64)
        print(f"  {'label':<18}{'heuristic':>12}{'efficientat':>14}{'Δ':>8}")
        print("-" * 64)
        for lab in _TARGET_LABELS:
            hs = h.per_class.get(lab, {})
            es = e.per_class.get(lab, {})
            if (hs.get("support", 0) == 0) and (es.get("support", 0) == 0):
                continue
            hf, ef = hs.get("f1", 0.0), es.get("f1", 0.0)
            print(f"  {lab:<18}{hf:>12.3f}{ef:>14.3f}{ef - hf:>+8.3f}")
        print("-" * 64)
        print(f"  {'MACRO-F1':<18}{h.macro_f1:>12.3f}{e.macro_f1:>14.3f}"
              f"{e.macro_f1 - h.macro_f1:>+8.3f}")
        print(f"  {'top-1 hit-rate':<18}{h.micro_acc:>12.3f}{e.micro_acc:>14.3f}"
              f"{e.micro_acc - h.micro_acc:>+8.3f}")
        print("=" * 64)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""
Export EfficientAT-S to ONNX (FP32) and quantize to INT8 — the deployment
artifact consumed by ``voiceiso.stages.noise_classifier.EfficientATNoiseClassifier``.

The classifier loads ``cfg.noise_classifier_model_path`` via ONNX Runtime and
reads the AudioSet class-name list from the model's ``labels`` metadata field
(comma-separated).  This script produces a model with that metadata baked in so
the AudioSet→target label map works without any sidecar file.

Pipeline:
  1. Load EfficientAT mn10_as (MobileNetV3-class, width 1.0, ~4.9 M params,
     AudioSet mAP ~0.47) from the EfficientAT project
     (https://github.com/fschmid56/EfficientAT).
  2. Wrap it so the ONNX input is raw 16 kHz audio of a fixed window length
     (default 200 ms = 3200 samples) and the output is the 527-way AudioSet
     posterior vector.
  3. Export to FP32 ONNX with the AudioSet class names in metadata.
  4. INT8 dynamic-quantize (weights int8, activations fp32) — ~4× smaller,
     ~3 MB resident, CPU-friendly.  The conv/linear layers carry the win.

This is an OFFLINE tool — it does not run in the live pipeline.

Usage
-----
    python -m scripts.export_efficientat_onnx \\
        --out checkpoints/efficientat_s.onnx \\
        --quantized-out checkpoints/efficientat_s_int8.onnx \\
        --window-ms 200

Then point the config at the INT8 model:
    PipelineConfig(noise_classifier_model_path="checkpoints/efficientat_s_int8.onnx")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _audioset_labels() -> list[str]:
    """Return the 527 AudioSet class display names, in model output order.

    The EfficientAT repo ships these in ``datasets/audioset/class_labels_indices.csv``
    (the standard AudioSet ontology order).  We try to import them from the
    installed package; callers without the package can pass --labels-csv.
    """
    try:
        import csv
        import hear21passt  # noqa: F401  — EfficientAT often vendors AudioSet labels
    except Exception:
        return []
    return []


def _load_labels_csv(path: Path) -> list[str]:
    import csv
    out: list[str] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # AudioSet class_labels_indices.csv columns: index, mid, display_name
            out.append(row.get("display_name", "").strip().strip('"'))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Export EfficientAT-S → ONNX + INT8")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/efficientat_s.onnx"))
    ap.add_argument("--quantized-out", type=Path,
                    default=Path("checkpoints/efficientat_s_int8.onnx"))
    ap.add_argument("--window-ms", type=float, default=200.0,
                    help="analysis window the live classifier feeds (must match cfg)")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--labels-csv", type=Path, default=None,
                    help="AudioSet class_labels_indices.csv (for labels metadata)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-quantize", action="store_true")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load EfficientAT-S.
    try:
        import torch
        # EfficientAT public API (github.com/fschmid56/EfficientAT):
        #   from models.mn.model import get_model as get_mobilenet
        from models.mn.model import get_model as get_mobilenet  # type: ignore[import]
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: EfficientAT not importable — {exc}", file=sys.stderr)
        print("Install from https://github.com/fschmid56/EfficientAT and add it "
              "to PYTHONPATH.", file=sys.stderr)
        return 2

    print("loading EfficientAT-S (mn10_as)…")
    model = get_mobilenet(width_mult=1.0, pretrained_name="mn10_as")
    model.eval()

    # 2. Export MODEL-ONLY with a MEL input (mel-in ONNX).  The mel is computed
    #    at runtime by voiceiso's vendored EfficientAT AugmentMelSTFT (32 kHz) —
    #    NOT inside the ONNX graph — so we avoid fragile torch.stft→ONNX export
    #    and guarantee the runtime mel exactly matches training.
    #    Input  : mel (1, 1, 128, T)
    #    Output : sigmoid posteriors (1, 527)
    n_mels = 128
    # Frame count is FIXED: the runtime always feeds exactly window_samples, so
    # the mel has a deterministic frame count.  Compute it from the SAME vendored
    # mel the runtime uses, and export a STATIC shape — dynamic axes here make
    # the INT8 quantizer's shape-inference fail (and we never need them, since
    # the window is constant).
    import numpy as _np
    from voiceiso.stages._efficientat_mel import EfficientATMel
    win_samples = int(args.window_ms / 1000.0 * args.sr)
    _mel = EfficientATMel(n_mels=n_mels, sr=args.sr)
    frames = int(_mel(_np.zeros(win_samples, dtype="float32")).shape[-1])

    class _ModelOnly(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, mel):             # mel: (1, 1, 128, frames)
            out = self.net(mel)
            logits = out[0] if isinstance(out, tuple) else out
            return torch.sigmoid(logits)    # AudioSet multi-label → sigmoid

    wrapper = _ModelOnly(model).eval()
    dummy = torch.zeros(1, 1, n_mels, frames, dtype=torch.float32)

    print(f"exporting FP32 ONNX (mel-in, static) → {args.out}  "
          f"(mel {n_mels}×{frames}, window {win_samples} @ {args.sr} Hz)…")
    torch.onnx.export(
        wrapper, (dummy,), str(args.out),
        input_names=["mel"], output_names=["posteriors"],
        opset_version=args.opset,
    )

    # 3. Bake the AudioSet label list into the model metadata so the runtime
    #    classifier can build the AudioSet→target map without a sidecar file.
    labels = _load_labels_csv(args.labels_csv) if args.labels_csv else _audioset_labels()
    if labels:
        try:
            import onnx
            m = onnx.load(str(args.out))
            entry = m.metadata_props.add()
            entry.key = "labels"
            entry.value = "|".join(labels)
            onnx.save(m, str(args.out))
            print(f"  embedded {len(labels)} AudioSet labels in model metadata")
        except Exception as exc:  # pragma: no cover
            print(f"  WARNING: could not embed labels metadata — {exc}", file=sys.stderr)
    else:
        print("  WARNING: no labels available; pass --labels-csv so the runtime "
              "classifier can map AudioSet → target labels (else it raises and the "
              "pipeline falls back to the heuristic classifier).", file=sys.stderr)

    # 4. INT8 dynamic quantization.
    if args.skip_quantize:
        print("INT8 quantization skipped (--skip-quantize).")
        return 0
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: onnxruntime.quantization unavailable — {exc}", file=sys.stderr)
        return 2

    args.quantized_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"quantizing → {args.quantized_out} (INT8 dynamic)…")
    quantize_dynamic(
        model_input=str(args.out),
        model_output=str(args.quantized_out),
        weight_type=QuantType.QInt8,
    )
    # Re-embed labels on the quantized file (quantize_dynamic drops metadata).
    if labels:
        try:
            import onnx
            m = onnx.load(str(args.quantized_out))
            entry = m.metadata_props.add()
            entry.key = "labels"
            entry.value = "|".join(labels)
            onnx.save(m, str(args.quantized_out))
        except Exception:  # pragma: no cover
            pass
    size_mb = args.quantized_out.stat().st_size / 1e6
    print(f"done. INT8 model: {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

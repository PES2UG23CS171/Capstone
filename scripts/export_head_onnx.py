"""
Export the trained native 12-class head + frozen EfficientAT backbone to a
mel-in ONNX that the runtime classifier auto-detects as a native-12 head.

Composition:
    mel (1,1,128,T) → backbone → 960-dim embedding → trained head → sigmoid → (1,12)

The 12 target labels are embedded as metadata, so
``EfficientATNoiseClassifier`` sets ``_native_target=True`` and uses the
posteriors directly (no AudioSet→target map).

Run from the EfficientAT clone dir with repo root on PYTHONPATH:

    cd third_party/EfficientAT
    PYTHONPATH="<repo>:$PWD" python <repo>/scripts/export_head_onnx.py \\
        --head <repo>/checkpoints/head_12.pt \\
        --out  <repo>/checkpoints/efficientat_head12.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="checkpoints/head_12.pt")
    ap.add_argument("--out", default="checkpoints/efficientat_head12.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import torch
    from models.mn.model import get_model
    from voiceiso.stages._efficientat_mel import EfficientATMel

    ckpt = torch.load(args.head, map_location="cpu")
    labels = ckpt["labels"]
    emb_dim = ckpt["emb_dim"]
    hidden = ckpt.get("hidden", 0)
    sr = ckpt.get("sr", 32000)
    window_s = ckpt.get("window_s", 4.0)
    n_cls = len(labels)

    backbone = get_model(width_mult=1.0, pretrained_name="mn10_as").eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    if hidden > 0:
        head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, hidden), torch.nn.ReLU(),
            torch.nn.Dropout(0.3), torch.nn.Linear(hidden, n_cls))
    else:
        head = torch.nn.Linear(emb_dim, n_cls)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    class _Composed(torch.nn.Module):
        def __init__(self, bb, hd):
            super().__init__()
            self.bb = bb
            self.hd = hd

        def forward(self, mel):                 # (1,1,128,T)
            out = self.bb(mel)
            feats = out[1] if isinstance(out, tuple) else out
            return torch.sigmoid(self.hd(feats))  # (1, 12)

    model = _Composed(backbone, head).eval()

    # Static mel frame count for the configured window (quantization-friendly).
    mel_mod = EfficientATMel(sr=sr)
    frames = int(mel_mod(np.zeros(int(window_s * sr), dtype="float32")).shape[-1])
    dummy = torch.zeros(1, 1, 128, frames, dtype=torch.float32)

    print(f"exporting native-12 head ONNX → {args.out}  (mel 128×{frames}, window {window_s}s @ {sr} Hz)")
    torch.onnx.export(
        model, (dummy,), str(args.out),
        input_names=["mel"], output_names=["posteriors"],
        opset_version=args.opset,
    )

    # Embed the 12 target labels (pipe-joined; runtime detects native-12).
    try:
        import onnx
        m = onnx.load(str(args.out))
        e = m.metadata_props.add()
        e.key = "labels"
        e.value = "|".join(labels)
        onnx.save(m, str(args.out))
        print(f"  embedded {len(labels)} target labels: {labels}")
    except Exception as exc:  # pragma: no cover
        print(f"  WARNING: could not embed labels — {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

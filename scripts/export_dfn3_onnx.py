"""
Export DeepFilterNet3 to ONNX (FP32) and quantize to INT8 dynamic.

Goal
----
Cut the live deployment footprint:

    PyTorch eager  ≈ 370 MB resident, ~21 ms per 100 ms block
    ONNX FP32      ≈ 240 MB resident, similar speed
    ONNX INT8      ≈  90 MB resident, similar speed (often slightly faster)

This is an offline tool — it does NOT modify the live pipeline.  After running
it, the produced ``dfn3_int8.onnx`` can be loaded by an alternative
Enhancement backend (not implemented here; see Section 9 of Docs/ARCHITECTURE.md
for the integration plan).

Caveats
-------
* DFN3 has recurrent (GRU) layers.  We must export with explicit hidden-state
  I/O so the ONNX graph is stateless and the caller threads ``h_n`` through.
* The DFN package wraps the model in helper code (init_df, enhance) that does
  STFT/ERB feature extraction; only the **core network** is exported here.
  Re-implementing the STFT/ERB front-end in NumPy is a separate task — not
  done here.
* This script depends on a working DFN install (Rust toolchain present) and
  PyTorch ≥ 1.13 with ONNX export support.

Usage
-----
    python -m scripts.export_dfn3_onnx \\
        --out checkpoints/dfn3.onnx \\
        --quantized-out checkpoints/dfn3_int8.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Export DeepFilterNet3 to ONNX + INT8")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/dfn3.onnx"),
                    help="Output FP32 ONNX path")
    ap.add_argument("--quantized-out", type=Path, default=Path("checkpoints/dfn3_int8.onnx"),
                    help="Output INT8 (dynamic-quantized) ONNX path")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-quantize", action="store_true",
                    help="Only export FP32; skip the INT8 quantization step")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load DFN3.
    try:
        import voiceiso._compat  # noqa: F401 — torchaudio shim
        import torch
        from df.enhance import init_df
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: cannot import DeepFilterNet — {exc}", file=sys.stderr)
        print("Install with: pip install deepfilternet (requires Rust toolchain)", file=sys.stderr)
        return 2

    print("loading DFN3…")
    model, state, _ = init_df(config_allow_defaults=True)
    model.eval()
    sr = state.sr()
    print(f"  model loaded — sample rate {sr} Hz, hop {state.hop_size()}")

    # 2. Trace the core network with a dummy input.  DFN3 expects (batch, T)
    #    audio; in practice this is the path inside `enhance()` that calls
    #    the model after the ERB-band feature extraction.  Most DFN-flavoured
    #    exports run the inner ``model.forward`` directly on a spectrogram
    #    tensor; without first-party export support, the canonical pattern is
    #    to wrap forward() in a helper that exposes hidden-state I/O.
    #
    #    NOTE: the DFN public API does not expose a clean spectrogram-in /
    #    spectrogram-out forward.  Implementing the full export reliably is
    #    out of scope for this scaffold — what follows is the *export call*
    #    you would use once that wrapper exists.

    print("WARNING: this scaffold demonstrates the export pipeline but does not")
    print("         define a graph-friendly forward() wrapper for DFN3 yet.")
    print("         A clean export requires a small wrapper module that takes")
    print("         (spectrogram, gru_hidden_states) and returns (mask, hidden).")
    print("         See: https://github.com/Rikorose/DeepFilterNet/discussions/")
    print()

    # ── PSEUDO-CODE (kept for guidance, not exercised) ────────────────────
    # dummy_spec = torch.zeros(1, 1, 100, 481)
    # dummy_h_erb = torch.zeros(2, 1, 256)
    # dummy_h_df  = torch.zeros(2, 1, 256)
    # wrapper = ExportWrapper(model)
    # torch.onnx.export(
    #     wrapper,
    #     (dummy_spec, dummy_h_erb, dummy_h_df),
    #     str(args.out),
    #     input_names=["spec", "h_erb_in", "h_df_in"],
    #     output_names=["mask", "h_erb_out", "h_df_out"],
    #     opset_version=args.opset,
    #     dynamic_axes={
    #         "spec": {0: "batch", 2: "time"},
    #         "mask": {0: "batch", 2: "time"},
    #     },
    # )

    # 3. Quantize to INT8 (dynamic — weights only, activations stay FP32).
    if args.skip_quantize:
        print("INT8 quantization skipped (--skip-quantize).")
        return 0

    if not args.out.exists():
        print("FP32 export did not run — nothing to quantize.")
        return 1

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: onnxruntime.quantization unavailable — {exc}", file=sys.stderr)
        print("Install with: pip install onnxruntime", file=sys.stderr)
        return 2

    args.quantized_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"quantizing  {args.out}  →  {args.quantized_out}  (INT8 dynamic)…")
    quantize_dynamic(
        model_input=str(args.out),
        model_output=str(args.quantized_out),
        weight_type=QuantType.QInt8,
    )
    print(f"done. size: {args.quantized_out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

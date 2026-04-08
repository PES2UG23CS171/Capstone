"""
ONNX Export
===========
Export the quantized, pruned CombinedModel to ONNX for deployment via
ONNX Runtime.
"""

from __future__ import annotations

from pathlib import Path

import torch

import config as cfg


def export_to_onnx(
    model: torch.nn.Module,
    onnx_path: str = cfg.ONNX_MODEL_PATH,
    context_window: int = cfg.CONTEXT_WINDOW_SAMPLES,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Parameters
    ----------
    model : nn.Module
        The trained (and optionally quantized) model.
    onnx_path : str
        Output path for the ``.onnx`` file.
    context_window : int
        Input sequence length.

    Returns
    -------
    path : Path
        Absolute path to the saved ONNX file.
    """
    model.eval()
    save_path = Path(onnx_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, context_window)

    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        opset_version=18,
        input_names=["noisy_audio"],
        output_names=["clean_audio"],
        dynamic_axes={
            "noisy_audio": {0: "batch"},
            "clean_audio": {0: "batch"},
        },
    )

    print(f"  ONNX model exported → {save_path}")

    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(save_path), providers=["CPUExecutionProvider"])
        import numpy as np
        test_input = np.random.randn(1, context_window).astype(np.float32)
        result = sess.run(None, {"noisy_audio": test_input})
        assert result[0].shape == (1, context_window), (
            f"Shape mismatch: expected (1, {context_window}), got {result[0].shape}"
        )
        print(f"  ✓ ONNX verification passed  (output shape: {result[0].shape})")
    except ImportError:
        print("  ⚠ onnxruntime not installed — skipping verification")
    except Exception as exc:
        print(f"  ⚠ ONNX verification failed: {exc}")

    # Print graph summary
    try:
        import onnx
        onnx_model = onnx.load(str(save_path))
        print(f"  ONNX graph nodes: {len(onnx_model.graph.node)}")
        print(f"  ONNX IR version : {onnx_model.ir_version}")
    except ImportError:
        pass

    return save_path.resolve()


if __name__ == "__main__":
    from model.pretrained_stub import create_stub_model
    model = create_stub_model()
    export_to_onnx(model)

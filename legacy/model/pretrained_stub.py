"""
Pretrained Stub
===============
Generates a randomly initialised but structurally correct ``CombinedModel``
that can be dropped into the inference pipeline for demo purposes.

The model produces audio output (not silence) immediately, so the GUI is
fully functional even without a trained checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import torch

import config as cfg
from model.combined_model import CombinedModel


def create_stub_model(
    context_window: int = cfg.CONTEXT_WINDOW_SAMPLES,
    fir_length: int = cfg.FIR_FILTER_LENGTH,
    d_model: int = cfg.MAMBA_D_MODEL,
    d_state: int = cfg.MAMBA_D_STATE,
    n_layers: int = cfg.MAMBA_N_LAYERS,
) -> CombinedModel:
    """Create a randomly initialised CombinedModel for demo use.

    The weights are random but the architecture is correct, so the model
    can be exported to ONNX and run through the inference pipeline.
    """
    model = CombinedModel(
        context_window=context_window,
        fir_length=fir_length,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
    )
    model.eval()
    return model


def save_stub_checkpoint(
    path: str = "checkpoints/stub_model.pt",
    **kwargs,
) -> Path:
    """Create and save a stub model checkpoint.

    Returns the path to the saved file.
    """
    model = create_stub_model(**kwargs)
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  Stub model saved → {save_path}")
    model.count_parameters()
    return save_path


def load_model(path: str, **kwargs) -> CombinedModel:
    """Load a CombinedModel from a checkpoint file.

    Falls back to a stub model if the file doesn't exist.
    """
    model = CombinedModel(**kwargs)
    ckpt_path = Path(path)

    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  Loaded model from {ckpt_path}")
    else:
        print(f"  Checkpoint {ckpt_path} not found — using random weights (stub).")

    model.eval()
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("  Creating stub model …")
    print("=" * 60)
    path = save_stub_checkpoint()
    print(f"\n  Done.  Run the app and it will use → {path}")

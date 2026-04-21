"""
Stubbed inference engine — Phase 1 passthrough.

In Phase 2 this module will load the Mamba + DeepFIR model exported to ONNX
and run INT8-quantised inference via ONNX Runtime.  For now it simply passes
audio through unchanged (or applies a trivial gain-based "strength" knob so
the GUI slider has an audible effect for integration testing).
"""

from __future__ import annotations

import numpy as np


import torch
import config as cfg
from model.combined_model import CombinedModel

class PyTorchDenoiser:
    """Drop-in real model inference using PyTorch natively."""

    def __init__(
        self,
        model_path: str | None = None,
        sample_rate: int = 48_000,
        block_size: int = 256,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._strength: float = 1.0   # 0 = bypass, 1 = full processing
        
        self.device = "cpu"
        self.model = CombinedModel()
        
        # Load best.pt weights
        import os
        checkpoint = "checkpoints/best.pt"
        if os.path.exists(checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        self.model.eval()
        
        # Pre-allocate hidden states for Mamba
        self.hidden_states = self.model.mamba.init_hidden(batch_size=1, device=self.device)

    # ── public API ───────────────────────────────────────────────────────

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value: float) -> None:
        self._strength = float(np.clip(value, 0.0, 1.0))

    def process(self, block: np.ndarray) -> np.ndarray:
        if self._strength == 0.0:
            return block.copy()

        # Convert to tensor. `block` is typically (frames, 1). 
        # We need (1, frames) for [Batch, Time] context streaming.
        noisy_tensor = torch.tensor(block, dtype=torch.float32, device=self.device)
        if noisy_tensor.ndim == 2 and noisy_tensor.shape[1] == 1:
            noisy_tensor = noisy_tensor.squeeze(1)  # (frames,)
            
        chunk = noisy_tensor.unsqueeze(0)  # (1, frames)
        
        # Real-time inference implementation
        with torch.no_grad():
            outputs = []
            for i in range(len(noisy_tensor)):
                sample = noisy_tensor[i].unsqueeze(0)  # [1]
                out, self.hidden_states = self.model.forward_realtime(
                    sample, self.hidden_states
                )
                outputs.append(out)

        out = torch.cat(outputs)  # (frames,)

        denoised = out.numpy()
        if block.ndim == 2:
            denoised = denoised[:, np.newaxis]  # match original (frames, 1) shape
        denoised = denoised[:len(block)]

        if self._strength < 1.0:
            return (1.0 - self._strength) * block + self._strength * denoised

        return denoised



class ONNXDenoiser:
    """Placeholder for Phase 2 — will wrap ``onnxruntime.InferenceSession``.

    Importing this class without onnxruntime installed will raise
    ``ImportError`` at construction time so it is safe to reference in
    type annotations without crashing Phase 1 code.
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 48_000,
        block_size: int = 1024,
    ) -> None:
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNXDenoiser.  "
                "Install it with:  pip install onnxruntime"
            ) from exc

        self.sample_rate = sample_rate
        self.block_size = block_size
        self._strength: float = 1.0
        # TODO: load session, allocate IO binding, warm up
        raise NotImplementedError("ONNXDenoiser is a Phase 2 target.")

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value: float) -> None:
        self._strength = float(np.clip(value, 0.0, 1.0))

    def process(self, block: np.ndarray) -> np.ndarray:
        raise NotImplementedError

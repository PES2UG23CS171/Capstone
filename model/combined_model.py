"""
Combined Model — DeepFIR + Mamba SSM
=====================================
End-to-end model: raw noisy waveform → DeepFIR (stationary noise) → Mamba SSM
(transient suppression) → clean waveform.

Supports two forward modes:
* ``forward_train``    — full sequence, parallel scan (training)
* ``forward_realtime`` — single sample, recurrent scan (inference)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

import config as cfg
from model.deep_fir import DeepFIRPredictor
from model.mamba_ssm import MambaSSM


class CombinedModel(nn.Module):
    """Joint DeepFIR + Mamba SSM denoising model.

    Architecture::

        Input: [B, context_window]
          → DeepFIR → predict FIR taps → apply to audio → intermediate
          → reshape to [B, L, 1] → Linear(1, d_model) → MambaSSM
          → Linear(d_model, 1) → squeeze → clean_audio
        Output: [B, context_window]
    """

    def __init__(
        self,
        context_window: int = cfg.CONTEXT_WINDOW_SAMPLES,
        fir_length: int = cfg.FIR_FILTER_LENGTH,
        d_model: int = cfg.MAMBA_D_MODEL,
        d_state: int = cfg.MAMBA_D_STATE,
        n_layers: int = cfg.MAMBA_N_LAYERS,
    ) -> None:
        super().__init__()
        self.context_window = context_window
        self.fir_length = fir_length
        self.d_model = d_model

        # Layer 3: DeepFIR
        self.deep_fir = DeepFIRPredictor(context_window, fir_length)

        # Projection: 1-D audio sample → d_model features
        self.input_proj = nn.Linear(1, d_model)

        # Layer 4: Mamba SSM
        self.mamba = MambaSSM(d_model, d_state, n_layers)

        # Output projection: d_model → 1
        self.output_proj = nn.Linear(d_model, 1)

    def forward_train(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward pass for training.

        Parameters
        ----------
        noisy_audio : Tensor, shape ``[B, context_window]``

        Returns
        -------
        clean_audio : Tensor, shape ``[B, context_window]``
        """
        B, T = noisy_audio.shape

        # Stage 1: DeepFIR predicts FIR taps and applies them
        taps = self.deep_fir(noisy_audio)                  # [B, fir_length]
        intermediate = DeepFIRPredictor.apply_fir_torch(noisy_audio, taps)  # [B, T]

        # Stage 2: Mamba SSM processes the intermediate signal
        x = intermediate.unsqueeze(-1)                      # [B, T, 1]
        x = self.input_proj(x)                              # [B, T, d_model]
        x = self.mamba(x, use_parallel=True)                # [B, T, d_model]
        x = self.output_proj(x).squeeze(-1)                 # [B, T]

        return x

    def forward_realtime(
        self,
        sample: torch.Tensor,
        hidden_states: List[torch.Tensor],
        fir_taps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Single-sample forward pass for real-time inference.

        Parameters
        ----------
        sample : Tensor, shape ``[B, 1]`` or ``[B]``
        hidden_states : list of hidden state tensors
        fir_taps : Tensor, shape ``[B, fir_length]``, optional
            Pre-computed FIR taps (if reusing across samples in a chunk).

        Returns
        -------
        output : Tensor, shape ``[B]``
        new_hidden_states : list of Tensors
        """
        if sample.dim() == 1:
            sample = sample.unsqueeze(-1)  # [B, 1]

        # Project to d_model
        x = self.input_proj(sample)        # [B, d_model]

        # Mamba recurrent step
        x, new_states = self.mamba.forward_recurrent(x, hidden_states)

        # Output projection
        out = self.output_proj(x).squeeze(-1)  # [B]

        return out, new_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward = training mode."""
        return self.forward_train(x)

    # ── Utilities ────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        """Count and report model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size_fp32_mb = total * 4 / (1024 * 1024)
        size_fp16_mb = total * 2 / (1024 * 1024)
        size_int8_mb = total * 1 / (1024 * 1024)

        info = {
            "total_params": total,
            "trainable_params": trainable,
            "size_fp32_mb": round(size_fp32_mb, 2),
            "size_fp16_mb": round(size_fp16_mb, 2),
            "size_int8_mb": round(size_int8_mb, 2),
        }

        print(f"  Total parameters    : {total:,}")
        print(f"  Trainable           : {trainable:,}")
        print(f"  Size (FP32)         : {size_fp32_mb:.2f} MB")
        print(f"  Size (FP16)         : {size_fp16_mb:.2f} MB")
        print(f"  Size (INT8 est.)    : {size_int8_mb:.2f} MB")

        return info

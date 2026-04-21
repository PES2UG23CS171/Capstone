"""
DeepFIR — Layer 3
=================
A lightweight neural network that predicts FIR filter tap coefficients from
raw audio context to suppress stationary noise.  Based on Dementyev et al. 2024.

Architecture::

    Input: [batch, context_window_samples]  (e.g. 512 samples)
      → Conv1D(1, 32, kernel=8, causal)  → PReLU
      → Conv1D(32, 64, kernel=8, causal) → PReLU
      → AdaptiveAvgPool1d(1) → Flatten
      → Linear(64, FIR_FILTER_LENGTH)    → Tanh
    Output: [batch, FIR_FILTER_LENGTH]    (predicted FIR coefficients)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.signal import lfilter
except ImportError:
    lfilter = None

import config as cfg


class CausalConv1d(nn.Module):
    """Conv1D with causal (left-only) padding so the model never peeks ahead."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad on the left only
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class DeepFIRPredictor(nn.Module):
    """Predicts FIR filter taps from a window of raw audio.

    Parameters
    ----------
    context_window : int
        Number of input samples (default: ``CONTEXT_WINDOW_SAMPLES``).
    fir_length : int
        Number of FIR taps to predict (default: ``FIR_FILTER_LENGTH``).
    """

    def __init__(
        self,
        context_window: int = cfg.CONTEXT_WINDOW_SAMPLES,
        fir_length: int = cfg.FIR_FILTER_LENGTH,
    ) -> None:
        super().__init__()
        self.context_window = context_window
        self.fir_length = fir_length

        self.conv1 = CausalConv1d(1, 32, kernel_size=8)
        self.act1 = nn.PReLU(32)
        self.conv2 = CausalConv1d(32, 64, kernel_size=8)
        self.act2 = nn.PReLU(64)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, fir_length)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict FIR coefficients from audio context.

        Parameters
        ----------
        x : Tensor, shape ``[batch, context_window]``

        Returns
        -------
        taps : Tensor, shape ``[batch, fir_length]``
        """
        # Reshape to [B, 1, T] for Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))

        x = self.pool(x).squeeze(-1)  # [B, 64]
        taps = self.tanh(self.fc(x))   # [B, fir_length]
        return taps

    # ── Static FIR application ───────────────────────────────────────────

    @staticmethod
    def apply_fir(audio: np.ndarray, taps: np.ndarray) -> np.ndarray:
        """Convolve *audio* with predicted FIR *taps* using minimum-phase application.

        Parameters
        ----------
        audio : ndarray, shape ``(n_samples,)``
        taps : ndarray, shape ``(fir_length,)``

        Returns
        -------
        filtered : ndarray, same length as *audio*
        """
        if lfilter is None:
            raise ImportError("scipy is required for apply_fir")

        # Enforce minimum-phase via cepstral method
        min_phase_taps = DeepFIRPredictor._to_minimum_phase(taps)

        # Apply FIR using direct-form filter (minimum-phase ⇒ causal)
        filtered = lfilter(min_phase_taps, [1.0], audio).astype(np.float32)
        return filtered

    @staticmethod
    def _to_minimum_phase(taps: np.ndarray) -> np.ndarray:
        """Convert FIR taps to minimum-phase equivalent via cepstral method."""
        n = len(taps)
        # Compute magnitude spectrum
        spectrum = np.abs(np.fft.rfft(taps, n=max(n, 64)))
        # Avoid log(0)
        spectrum = np.maximum(spectrum, 1e-8)
        # Minimum-phase reconstruction via cepstral method
        log_spectrum = np.log(spectrum)
        cepstrum = np.fft.irfft(log_spectrum)
        # Apply lifter: double positive-time, zero negative-time
        cepstrum[1: len(cepstrum) // 2] *= 2.0
        cepstrum[len(cepstrum) // 2 + 1:] = 0.0
        min_phase = np.real(np.fft.irfft(np.exp(np.fft.rfft(cepstrum))))
        return min_phase[:n].astype(np.float32)

    @staticmethod
    def apply_fir_torch(audio: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
        """Apply FIR filter in PyTorch (differentiable, for training).

        Parameters
        ----------
        audio : Tensor, shape ``[batch, samples]``
        taps  : Tensor, shape ``[batch, fir_length]``

        Returns
        -------
        filtered : Tensor, shape ``[batch, samples]``
        """
        B, T = audio.shape
        fir_len = taps.shape[1]

        # Causal padding
        audio_padded = F.pad(audio, (fir_len - 1, 0))  # [B, T + fir_len - 1]

        if B == 1:
            # Safe ONNX export inference shape 
            x = audio_padded.unsqueeze(0)                    # [1, 1, T+K-1]
            w = taps.flip(1).unsqueeze(1)                    # [1, 1, K]
            y = F.conv1d(x, w, groups=1)                     # [1, 1, T]
        else:
            # Batched grouped convolution — all B elements at once
            x = audio_padded.unsqueeze(0)                    # [1, B, T+K-1]
            w = taps.flip(1).unsqueeze(1)                    # [B, 1, K]
            y = F.conv1d(x, w, groups=B)                     # [1, B, T]

        return y.squeeze(0)                              # [B, T]
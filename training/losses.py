"""
Training Losses
===============
Custom loss functions for transient noise suppression training.

* SI-SDR Loss  — scale-invariant signal-to-distortion ratio
* TSS Loss     — transient suppression score
* Plosive Loss — penalises distortion on speech plosives
* Combined     — weighted sum
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def si_sdr(clean: torch.Tensor, estimated: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-Invariant Signal-to-Distortion Ratio.

    Parameters
    ----------
    clean : Tensor, shape ``[B, T]``
        Ground-truth clean signal.
    estimated : Tensor, shape ``[B, T]``
        Model output.

    Returns
    -------
    si_sdr : Tensor, scalar
        Mean SI-SDR across the batch (higher is better).
    """
    # Zero-mean
    clean = clean - clean.mean(dim=-1, keepdim=True)
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)

    # Projection: alpha * s
    dot = (estimated * clean).sum(dim=-1, keepdim=True)
    s_target = dot / (clean.pow(2).sum(dim=-1, keepdim=True) + eps) * clean

    # Noise
    e_noise = estimated - s_target

    # SI-SDR in dB
    si_sdr_val = 10 * torch.log10(
        s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + eps) + eps
    )
    return si_sdr_val.mean()


def si_sdr_loss(clean: torch.Tensor, estimated: torch.Tensor) -> torch.Tensor:
    """Negative SI-SDR (to minimise)."""
    return -si_sdr(clean, estimated)


def tss_loss(
    noisy: torch.Tensor,
    estimated: torch.Tensor,
    clean: torch.Tensor,
    transient_mask: torch.Tensor,
) -> torch.Tensor:
    """Transient Suppression Score loss.

    Penalises model for leaving transient energy in the output and for
    distorting non-transient (speech) regions.

    Parameters
    ----------
    noisy : Tensor, ``[B, T]``
    estimated : Tensor, ``[B, T]``
    clean : Tensor, ``[B, T]``
    transient_mask : Tensor, ``[B, T]``
        1 where transient noise dominates, 0 elsewhere.
    """
    # Residual transient energy
    residual_energy = (estimated * transient_mask).pow(2).mean()

    # Speech preservation penalty
    speech_mask = 1.0 - transient_mask
    speech_penalty = ((estimated - clean) * speech_mask).pow(2).mean()

    return residual_energy + 0.5 * speech_penalty


def plosive_preservation_loss(
    estimated: torch.Tensor,
    clean: torch.Tensor,
    plosive_mask: torch.Tensor,
) -> torch.Tensor:
    """Heavily penalises distortion on plosive segments.

    Parameters
    ----------
    estimated : Tensor, ``[B, T]``
    clean : Tensor, ``[B, T]``
    plosive_mask : Tensor, ``[B, T]``
        1 on plosive segments, 0 elsewhere.
    """
    distortion = ((estimated - clean) * plosive_mask).pow(2).mean()
    return distortion * 10.0  # 10× weight


def combined_loss(
    noisy: torch.Tensor,
    estimated: torch.Tensor,
    clean: torch.Tensor,
    transient_mask: torch.Tensor,
    plosive_mask: torch.Tensor,
    weights: tuple = (1.0, 0.5, 1.0),
) -> torch.Tensor:
    """Weighted combination of SI-SDR, TSS, and plosive preservation losses."""
    loss_si = si_sdr_loss(clean, estimated)
    loss_tss = tss_loss(noisy, estimated, clean, transient_mask)
    loss_plos = plosive_preservation_loss(estimated, clean, plosive_mask)

    return weights[0] * loss_si + weights[1] * loss_tss + weights[2] * loss_plos

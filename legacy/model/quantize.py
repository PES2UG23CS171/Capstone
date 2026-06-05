"""
Quantization — Layer 2
======================
INT8 dynamic quantization and magnitude pruning for 7× speedup on CPU.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import config as cfg


# ---------------------------------------------------------------------------
#  Magnitude pruning
# ---------------------------------------------------------------------------

def apply_magnitude_pruning(
    model: nn.Module,
    prune_ratio: float = cfg.PRUNE_RATIO,
) -> nn.Module:
    """Zero out the smallest weights by magnitude (L1 unstructured).

    Parameters
    ----------
    model : nn.Module
        Model to prune (modified in-place).
    prune_ratio : float
        Fraction of weights to remove (default 0.50).

    Returns
    -------
    model : nn.Module
        The pruned model (same object, modified in-place).
    """
    total_params = 0
    pruned_params = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            total_before = module.weight.numel()
            prune.l1_unstructured(module, name="weight", amount=prune_ratio)
            prune.remove(module, "weight")  # make permanent
            zeros = (module.weight == 0).sum().item()
            total_params += total_before
            pruned_params += zeros

    sparsity = pruned_params / max(total_params, 1) * 100
    print(f"  Pruning complete: {pruned_params:,}/{total_params:,} "
          f"weights zeroed ({sparsity:.1f}% sparse)")

    return model


# ---------------------------------------------------------------------------
#  INT8 quantization
# ---------------------------------------------------------------------------

def quantize_model_int8(
    model: nn.Module,
    calibration_loader=None,
) -> nn.Module:
    """Apply dynamic INT8 quantization targeting Linear and Conv1d layers.

    Parameters
    ----------
    model : nn.Module
        FP32 model to quantize.
    calibration_loader : DataLoader, optional
        If provided, run calibration samples through the model first.

    Returns
    -------
    quantized_model : nn.Module
    """
    model.eval()

    # Run calibration data if available
    if calibration_loader is not None:
        with torch.no_grad():
            for i, (noisy, clean) in enumerate(calibration_loader):
                if i >= 100:
                    break
                _ = model(noisy)

    # Set quantized engine explicitly for Apple Silicon/ARM to prevent NoQEngine error
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    
    # Dynamic quantization (no GPU required)
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv1d},
        dtype=torch.qint8,
    )

    size_before = get_model_size_mb(model)
    size_after = get_model_size_mb(quantized)
    print(f"  Quantization complete: {size_before:.2f} MB → {size_after:.2f} MB "
          f"({size_before/max(size_after, 0.01):.1f}× reduction)")

    return quantized


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def get_model_size_mb(model: nn.Module) -> float:
    """Estimate model size in megabytes from parameter storage."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024 * 1024)


def benchmark_rtf(
    model: nn.Module,
    sample_rate: int = cfg.SAMPLE_RATE,
    duration_seconds: float = 10.0,
    context_window: int = cfg.CONTEXT_WINDOW_SAMPLES,
) -> float:
    """Measure Real-Time Factor on CPU.

    Parameters
    ----------
    model : nn.Module
    sample_rate : int
    duration_seconds : float
    context_window : int

    Returns
    -------
    rtf : float
        Processing time / audio duration.  Must be < 0.8.
    """
    model.eval()
    n_chunks = int(sample_rate * duration_seconds / context_window)
    dummy = torch.randn(1, context_window)

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_chunks):
            _ = model(dummy)
    elapsed = time.perf_counter() - start

    rtf = elapsed / duration_seconds
    headroom = 1.0 / rtf if rtf > 0 else 9999

    print(f"  RTF benchmark: {rtf:.4f}  ({headroom:.0f}× faster than real-time)")
    if rtf < cfg.TARGET_RTF:
        print(f"  ✓ PASS  (RTF {rtf:.4f} < {cfg.TARGET_RTF})")
    else:
        print(f"  ✗ FAIL  (RTF {rtf:.4f} >= {cfg.TARGET_RTF})")

    return rtf

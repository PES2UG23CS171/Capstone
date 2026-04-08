"""
Evaluation Metrics
==================
SI-SDRi, PESQ, TSS, and plosive evaluation with a formatted pass / fail report.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import config as cfg


# ---------------------------------------------------------------------------
#  SI-SDR
# ---------------------------------------------------------------------------

def _si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    """Compute SI-SDR between two 1-D signals."""
    ref = reference - np.mean(reference)
    est = estimate - np.mean(estimate)

    dot = np.sum(est * ref)
    s_target = dot / (np.sum(ref ** 2) + eps) * ref
    e_noise = est - s_target

    return float(10 * np.log10(np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + eps) + eps))


def compute_si_sdri(
    clean: np.ndarray,
    estimated: np.ndarray,
    noisy: np.ndarray,
) -> float:
    """SI-SDR improvement: SI-SDR(est, clean) − SI-SDR(noisy, clean).

    Target: > 4.0 dB.
    """
    si_sdr_est = _si_sdr(clean, estimated)
    si_sdr_noisy = _si_sdr(clean, noisy)
    return si_sdr_est - si_sdr_noisy


# ---------------------------------------------------------------------------
#  PESQ
# ---------------------------------------------------------------------------

def compute_pesq(
    clean: np.ndarray,
    estimated: np.ndarray,
    sample_rate: int = cfg.SAMPLE_RATE,
) -> float:
    """Perceptual Evaluation of Speech Quality.

    Target: ≥ 3.2.
    """
    try:
        from pesq import pesq as _pesq
        # PESQ only supports 8000 or 16000 Hz
        if sample_rate not in (8000, 16000):
            # Resample to 16 kHz
            from scipy.signal import resample_poly
            import math
            gcd = math.gcd(16000, sample_rate)
            clean_16k = resample_poly(clean, 16000 // gcd, sample_rate // gcd)
            est_16k = resample_poly(estimated, 16000 // gcd, sample_rate // gcd)
            return float(_pesq(16000, clean_16k, est_16k, "wb"))
        return float(_pesq(sample_rate, clean, estimated, "wb"))
    except ImportError:
        print("  ⚠ pesq library not installed — returning NaN")
        return float("nan")
    except Exception as exc:
        print(f"  ⚠ PESQ computation failed: {exc}")
        return float("nan")


# ---------------------------------------------------------------------------
#  TSS
# ---------------------------------------------------------------------------

def compute_tss(
    noisy: np.ndarray,
    estimated: np.ndarray,
    transient_mask: np.ndarray,
) -> float:
    """Transient Suppression Score: fraction of transient energy removed.

    TSS = 1 − (energy of estimated in transient regions) /
              (energy of noisy in transient regions)

    Target: > 0.65 (65%).
    """
    noisy_energy = np.sum((noisy * transient_mask) ** 2)
    est_energy = np.sum((estimated * transient_mask) ** 2)

    if noisy_energy < 1e-12:
        return 1.0  # No transients → perfect score

    return float(1.0 - est_energy / noisy_energy)


# ---------------------------------------------------------------------------
#  Plosive evaluation
# ---------------------------------------------------------------------------

def evaluate_on_plosives(
    clean: np.ndarray,
    estimated: np.ndarray,
    plosive_mask: np.ndarray,
) -> float:
    """Compute SI-SDR specifically on isolated plosive segments.

    Target: > 25 dB (ensures < 5% voice distortion).
    """
    # Extract plosive regions
    plosive_clean = clean * plosive_mask
    plosive_est = estimated * plosive_mask

    # Only compute if there's actual plosive content
    if np.sum(plosive_mask) < 10:
        return float("nan")

    return _si_sdr(plosive_clean, plosive_est)


# ---------------------------------------------------------------------------
#  Full evaluation report
# ---------------------------------------------------------------------------

def full_evaluation_report(
    clean: np.ndarray,
    estimated: np.ndarray,
    noisy: np.ndarray,
    transient_mask: Optional[np.ndarray] = None,
    plosive_mask: Optional[np.ndarray] = None,
    sample_rate: int = cfg.SAMPLE_RATE,
) -> dict:
    """Run all metrics and print a formatted pass/fail report."""

    print("\n" + "=" * 60)
    print("         EVALUATION REPORT")
    print("=" * 60)

    results = {}

    # SI-SDRi
    si_sdri = compute_si_sdri(clean, estimated, noisy)
    results["si_sdri"] = si_sdri
    status = "✓ PASS" if si_sdri > 4.0 else "✗ FAIL"
    print(f"  SI-SDRi           : {si_sdri:>8.2f} dB   (target > 4.0)    {status}")

    # PESQ
    pesq_score = compute_pesq(clean, estimated, sample_rate)
    results["pesq"] = pesq_score
    if not np.isnan(pesq_score):
        status = "✓ PASS" if pesq_score >= 3.2 else "✗ FAIL"
        print(f"  PESQ              : {pesq_score:>8.2f}       (target ≥ 3.2)    {status}")
    else:
        print(f"  PESQ              :      N/A       (library not installed)")

    # TSS
    if transient_mask is not None:
        tss = compute_tss(noisy, estimated, transient_mask)
        results["tss"] = tss
        status = "✓ PASS" if tss > 0.65 else "✗ FAIL"
        print(f"  TSS               : {tss*100:>7.1f}%       (target > 65%)    {status}")

    # Plosive SI-SDR
    if plosive_mask is not None:
        plos_sdr = evaluate_on_plosives(clean, estimated, plosive_mask)
        results["plosive_si_sdr"] = plos_sdr
        if not np.isnan(plos_sdr):
            status = "✓ PASS" if plos_sdr > 25.0 else "✗ FAIL"
            print(f"  Plosive SI-SDR    : {plos_sdr:>8.2f} dB   (target > 25.0)   {status}")

    print("=" * 60)
    return results


if __name__ == "__main__":
    # Quick demo with synthetic data
    np.random.seed(42)
    clean = np.sin(np.linspace(0, 100, 48000)).astype(np.float32)
    noise = np.random.randn(48000).astype(np.float32) * 0.1
    noisy = clean + noise
    estimated = clean + np.random.randn(48000).astype(np.float32) * 0.01
    mask = np.zeros(48000, dtype=np.float32)
    mask[10000:12000] = 1.0
    plosive_mask = np.zeros(48000, dtype=np.float32)
    plosive_mask[20000:20500] = 1.0

    full_evaluation_report(clean, estimated, noisy, mask, plosive_mask)

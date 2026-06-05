"""
Objective speech-enhancement metrics.

Implemented here (no extra deps): SI-SDR, segmental SNR, correlation.
Optional (used if installed): PESQ (``pesq``), STOI (``pystoi``), DNSMOS
(ONNX model).  DNSMOS (P.835 OVRL/SIG/BAK) is the metric that best tracks the
*subjective* quality Apple/Krisp optimise for, so it is the headline target when
available.
"""

from __future__ import annotations

from math import gcd
from typing import Dict, Optional

import numpy as np


def _align(ref: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(ref), len(est))
    return ref[:n].astype(np.float64), est[:n].astype(np.float64)


def si_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-9) -> float:
    ref, est = _align(ref, est)
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def seg_snr(ref: np.ndarray, est: np.ndarray, frame: int = 480, eps: float = 1e-9) -> float:
    ref, est = _align(ref, est)
    vals = []
    for s in range(0, len(ref) - frame, frame):
        r = ref[s:s + frame]; e = est[s:s + frame]
        rp = np.sum(r ** 2)
        if rp < eps:
            continue
        vals.append(10 * np.log10(rp / (np.sum((r - e) ** 2) + eps)))
    return float(np.clip(np.mean(vals), -10, 35)) if vals else 0.0


def correlation(ref: np.ndarray, est: np.ndarray) -> float:
    ref, est = _align(ref, est)
    return float(np.corrcoef(ref, est)[0, 1])


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    from scipy.signal import resample_poly
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g)


def pesq_wb(ref: np.ndarray, est: np.ndarray, sr: int) -> Optional[float]:
    try:
        from pesq import pesq as _pesq
    except Exception:
        return None
    ref, est = _align(ref, est)
    r = _resample(ref, sr, 16000); e = _resample(est, sr, 16000)
    try:
        return float(_pesq(16000, r.astype(np.float32), e.astype(np.float32), "wb"))
    except Exception:
        return None


def stoi(ref: np.ndarray, est: np.ndarray, sr: int) -> Optional[float]:
    try:
        from pystoi import stoi as _stoi
    except Exception:
        return None
    ref, est = _align(ref, est)
    try:
        return float(_stoi(ref, est, sr, extended=False))
    except Exception:
        return None


def all_metrics(ref: np.ndarray, est: np.ndarray, sr: int) -> Dict[str, float]:
    out = {
        "si_sdr": si_sdr(ref, est),
        "seg_snr": seg_snr(ref, est),
        "corr": correlation(ref, est),
    }
    p = pesq_wb(ref, est, sr)
    if p is not None:
        out["pesq_wb"] = p
    s = stoi(ref, est, sr)
    if s is not None:
        out["stoi"] = s
    return out

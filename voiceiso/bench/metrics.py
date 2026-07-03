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


# ── DNSMOS P.835 (non-intrusive) ────────────────────────────────────────────
#
# Microsoft DNS-Challenge DNSMOS local model ``sig_bak_ovr.onnx``.  It predicts
# three perceptual MOS sub-scores directly from RAW 16 kHz audio:
#   * SIG  — speech-signal quality
#   * BAK  — background-noise intrusiveness (higher = noise better removed)
#   * OVRL — overall quality
# Non-intrusive (no clean reference) → it is the only metric that works on real
# mic recordings and the one that best tracks the subjective quality Apple/Krisp
# optimise for.
#
# Protocol (matches dnsmos_local.py): resample to 16 kHz, tile up to 9.01 s,
# slide a 9.01 s window in 1 s hops, run the model per segment, apply the
# non-personalized polynomial calibration per segment, then average across
# segments.

_DNSMOS_SR = 16000
_DNSMOS_INPUT_LENGTH = 9.01          # seconds per inference segment
# Non-personalized polynomial calibration (degree-2; highest power first).
_DNSMOS_POLY = {
    "sig": (-0.08397278, 1.22083953, 0.0052439),
    "bak": (-0.13166888, 1.60915514, -0.39604546),
    "ovrl": (-0.06766283, 1.11546468, 0.04602535),
}

# Cache one ORT session per model path (loading is expensive).
_dnsmos_runners: Dict[str, "object"] = {}


class _DnsmosRunner:
    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, audio: np.ndarray, sr: int) -> Optional[Dict[str, float]]:
        a = _resample(np.asarray(audio, dtype=np.float64), sr, _DNSMOS_SR).astype(np.float32)
        seg_len = int(_DNSMOS_INPUT_LENGTH * _DNSMOS_SR)
        if len(a) == 0:
            return None
        # Tile up to one full segment.
        while len(a) < seg_len:
            a = np.concatenate([a, a])
        num_hops = int(np.floor(len(a) / _DNSMOS_SR) - _DNSMOS_INPUT_LENGTH) + 1
        num_hops = max(1, num_hops)
        sig_l, bak_l, ovr_l = [], [], []
        for idx in range(num_hops):
            seg = a[idx * _DNSMOS_SR: idx * _DNSMOS_SR + seg_len]
            if len(seg) < seg_len:
                continue
            feats = seg[np.newaxis, :].astype(np.float32)
            raw = self.sess.run(None, {self.input_name: feats})[0][0]
            s_raw, b_raw, o_raw = float(raw[0]), float(raw[1]), float(raw[2])
            sig_l.append(float(np.poly1d(_DNSMOS_POLY["sig"])(s_raw)))
            bak_l.append(float(np.poly1d(_DNSMOS_POLY["bak"])(b_raw)))
            ovr_l.append(float(np.poly1d(_DNSMOS_POLY["ovrl"])(o_raw)))
        if not sig_l:
            return None
        return {
            "sig": float(np.mean(sig_l)),
            "bak": float(np.mean(bak_l)),
            "ovrl": float(np.mean(ovr_l)),
        }


def dnsmos(audio: np.ndarray, sr: int, model_path: Optional[str]) -> Optional[Dict[str, float]]:
    """Return {sig, bak, ovrl} ∈ [~1, ~5] or None when unavailable.

    Unavailable cases (all silent — caller treats DNSMOS as optional):
      * ``model_path`` is None / missing
      * ``onnxruntime`` not installed
      * any inference error
    """
    if not model_path:
        return None
    try:
        runner = _dnsmos_runners.get(model_path)
        if runner is None:
            from pathlib import Path
            if not Path(model_path).exists():
                return None
            runner = _DnsmosRunner(model_path)
            _dnsmos_runners[model_path] = runner
        return runner(audio, sr)
    except Exception:
        return None


def all_metrics(ref: np.ndarray, est: np.ndarray, sr: int,
                dnsmos_model: Optional[str] = None) -> Dict[str, float]:
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
    if dnsmos_model:
        dm = dnsmos(est, sr, dnsmos_model)
        if dm is not None:
            out["dnsmos_sig"] = dm["sig"]
            out["dnsmos_bak"] = dm["bak"]
            out["dnsmos_ovrl"] = dm["ovrl"]
    return out

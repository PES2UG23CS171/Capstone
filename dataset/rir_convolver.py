"""
RIR Convolver
=============
Utilities for Room Impulse Response loading, convolution, and near/far-field
selection.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import fftconvolve

import config as cfg

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None


def load_rir(rir_path: str, sample_rate: int = cfg.SAMPLE_RATE) -> np.ndarray:
    """Load a WAV RIR, resample to *sample_rate* if needed, normalise to unit energy.

    Parameters
    ----------
    rir_path : str
        Path to the RIR WAV file.
    sample_rate : int
        Target sample rate.

    Returns
    -------
    rir : ndarray, 1-D float64
    """
    if sf is None:
        raise ImportError("soundfile is required for RIR loading")

    data, sr = sf.read(str(rir_path), dtype="float64", always_2d=True)
    data = data.mean(axis=1)  # mono

    if sr != sample_rate and resample_poly is not None:
        gcd = math.gcd(sample_rate, sr)
        data = resample_poly(data, sample_rate // gcd, sr // gcd)

    # Normalise to unit energy
    energy = np.sqrt(np.sum(data ** 2))
    if energy > 1e-12:
        data = data / energy

    return data


def apply_rir(
    audio: np.ndarray,
    rir: np.ndarray,
    sample_rate: int = cfg.SAMPLE_RATE,
) -> np.ndarray:
    """Convolve *audio* with *rir* using FFT, truncate to original length.

    Parameters
    ----------
    audio : ndarray, 1-D
    rir : ndarray, 1-D

    Returns
    -------
    convolved : ndarray, same length as *audio*
    """
    out = fftconvolve(audio, rir, mode="full")[:len(audio)]

    # Normalise to prevent clipping
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak

    return out


def _estimate_rt60(rir: np.ndarray, sample_rate: int) -> float:
    """Rough RT60 estimate from the RIR's energy decay curve."""
    energy = np.cumsum(rir[::-1] ** 2)[::-1]
    energy = energy / (energy[0] + 1e-12)
    energy_db = 10 * np.log10(energy + 1e-12)

    # Find time to decay by 60 dB
    below_60 = np.where(energy_db < -60)[0]
    if len(below_60) > 0:
        return float(below_60[0]) / sample_rate
    return float(len(rir)) / sample_rate


def get_near_field_rir(
    rir_dir: str,
    sample_rate: int = cfg.SAMPLE_RATE,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Return a random near-field RIR (RT60 < 0.3 s) from the directory.

    If no suitable RIR is found, returns a synthetic near-field impulse.
    """
    rng = rng or random.Random()
    rir_path = Path(rir_dir)

    if rir_path.exists():
        candidates = list(rir_path.glob("*.wav")) + list(rir_path.glob("*.flac"))
        rng.shuffle(candidates)

        for path in candidates[:20]:  # Check up to 20
            try:
                rir = load_rir(str(path), sample_rate)
                rt60 = _estimate_rt60(rir, sample_rate)
                if rt60 < 0.3:
                    return rir
            except Exception:
                continue

    # Fallback: synthetic near-field impulse
    rir = np.zeros(int(0.05 * sample_rate), dtype=np.float64)
    rir[0] = 1.0
    # Add minimal early reflections
    for i in range(3):
        pos = rng.randint(10, len(rir) // 2)
        rir[pos] = rng.uniform(0.1, 0.3) * (-1 if rng.random() < 0.5 else 1)
    return rir / np.sqrt(np.sum(rir ** 2) + 1e-12)


def get_far_field_rir(
    rir_dir: str,
    sample_rate: int = cfg.SAMPLE_RATE,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Return a random far-field RIR (RT60 > 0.5 s) from the directory.

    If no suitable RIR is found, returns a synthetic reverberant impulse.
    """
    rng = rng or random.Random()
    rir_path = Path(rir_dir)

    if rir_path.exists():
        candidates = list(rir_path.glob("*.wav")) + list(rir_path.glob("*.flac"))
        rng.shuffle(candidates)

        for path in candidates[:20]:
            try:
                rir = load_rir(str(path), sample_rate)
                rt60 = _estimate_rt60(rir, sample_rate)
                if rt60 > 0.5:
                    return rir
            except Exception:
                continue

    # Fallback: synthetic reverberant impulse
    n = int(0.8 * sample_rate)
    rir = np.random.RandomState(rng.randint(0, 2**31)).randn(n) * 0.01
    rir[0] = 1.0
    # Exponential decay (simulate reverb)
    decay = np.exp(-np.arange(n) / (0.3 * sample_rate))
    rir *= decay
    return rir / np.sqrt(np.sum(rir ** 2) + 1e-12)

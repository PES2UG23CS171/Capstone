"""
Synthetic Dataset Generator
============================
Generates (noisy, clean) audio pairs for training by mixing LibriSpeech
clean speech with FreeSound transient noises, convolved with Room Impulse
Responses at random SNRs.

This module complements the existing ``generate_dataset.py`` at the project
root — this version produces ``.npz`` files compatible with the PyTorch
``TransientNoiseDataset`` loader.
"""

from __future__ import annotations

import argparse
import logging
import math
import multiprocessing
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve

import config as cfg

try:
    import soundfile as sf
except ImportError:
    sf = None

from dataset.rir_convolver import get_near_field_rir, get_far_field_rir

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) or 1e-12


def _load_and_resample(path: str, sr: int) -> np.ndarray:
    if sf is None:
        raise ImportError("soundfile required")
    data, file_sr = sf.read(str(path), dtype="float64", always_2d=True)
    data = data.mean(axis=1)
    if file_sr != sr:
        from scipy.signal import resample_poly
        gcd = math.gcd(sr, file_sr)
        data = resample_poly(data, sr // gcd, file_sr // gcd)
    return data


def _random_crop(audio: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    if len(audio) >= length:
        start = rng.randint(0, len(audio) - length)
        return audio[start:start + length].copy()
    padded = np.zeros(length, dtype=audio.dtype)
    start = rng.randint(0, length - len(audio))
    padded[start:start + len(audio)] = audio
    return padded


# ---------------------------------------------------------------------------
#  Single pair generation
# ---------------------------------------------------------------------------

def generate_synthetic_pair(
    speech_path: str,
    noise_path: str,
    near_field_rir: np.ndarray,
    far_field_rir: np.ndarray,
    snr_db: float,
    segment_samples: int,
    sample_rate: int = cfg.SAMPLE_RATE,
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one (noisy, clean) pair.

    Returns
    -------
    noisy : ndarray, float32
    clean : ndarray, float32
    """
    rng = rng or random.Random()

    # Load and crop
    speech = _random_crop(_load_and_resample(speech_path, sample_rate), segment_samples, rng)
    noise = _random_crop(_load_and_resample(noise_path, sample_rate), segment_samples, rng)

    # Apply RIRs
    speech_reverb = fftconvolve(speech, near_field_rir, mode="full")[:segment_samples]
    noise_reverb = fftconvolve(noise, far_field_rir, mode="full")[:segment_samples]

    # Scale noise to target SNR
    speech_rms = _rms(speech_reverb)
    noise_rms = _rms(noise_reverb)
    target_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    noise_scaled = noise_reverb * (target_noise_rms / noise_rms)

    # Mix
    noisy = speech_reverb + noise_scaled
    clean = speech_reverb.copy()

    # Joint peak normalisation
    peak = max(np.max(np.abs(noisy)), np.max(np.abs(clean)), 1e-8)
    gain = 0.95 / peak
    noisy *= gain
    clean *= gain

    return noisy.astype(np.float32), clean.astype(np.float32)


# ---------------------------------------------------------------------------
#  Full dataset generation
# ---------------------------------------------------------------------------

def _discover_audio(root: str) -> List[str]:
    exts = {".wav", ".flac", ".ogg", ".mp3"}
    p = Path(root)
    if not p.exists():
        return []
    return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in exts)


def generate_full_dataset(
    output_dir: str = cfg.DATASET_DIR,
    n_pairs: int = cfg.NUM_SYNTHETIC_PAIRS,
    speech_dir: str = cfg.LIBRISPEECH_DIR,
    noise_dir: str = cfg.FREESOUND_DIR,
    rir_dir: str = cfg.RIR_DIR,
) -> None:
    """Generate the full synthetic dataset as .npz files."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    speech_files = _discover_audio(speech_dir)
    noise_files = _discover_audio(noise_dir)

    if not speech_files:
        log.error("No speech files found in %s", speech_dir)
        return
    if not noise_files:
        log.error("No noise files found in %s", noise_dir)
        return

    log.info("Speech files: %d", len(speech_files))
    log.info("Noise files : %d", len(noise_files))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    segment_samples = int(cfg.SAMPLE_RATE * 4.0)  # 4 seconds

    for i in range(n_pairs):
        try:
            speech_path = rng.choice(speech_files)
            noise_path = rng.choice(noise_files)

            near_rir = get_near_field_rir(rir_dir, rng=rng)
            far_rir = get_far_field_rir(rir_dir, rng=rng)
            snr_db = rng.uniform(*cfg.SNR_RANGE_DB)

            noisy, clean = generate_synthetic_pair(
                speech_path, noise_path, near_rir, far_rir,
                snr_db, segment_samples, rng=rng,
            )

            np.savez_compressed(
                out_path / f"{i:06d}.npz",
                noisy=noisy,
                clean=clean,
                snr_db=snr_db,
            )

            if (i + 1) % 500 == 0 or i == 0:
                hours = (i + 1) * 4.0 / 3600
                log.info("  %d / %d pairs generated (%.1f hours)", i + 1, n_pairs, hours)

        except Exception as exc:
            log.warning("  Pair %d failed: %s — skipping", i, exc)

    total_hours = n_pairs * 4.0 / 3600
    log.info("Dataset complete: %d pairs (%.1f hours) → %s", n_pairs, total_hours, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--output", type=str, default=cfg.DATASET_DIR)
    parser.add_argument("--speech-dir", type=str, default=cfg.LIBRISPEECH_DIR)
    parser.add_argument("--noise-dir", type=str, default=cfg.FREESOUND_DIR)
    parser.add_argument("--rir-dir", type=str, default=cfg.RIR_DIR)
    args = parser.parse_args()
    generate_full_dataset(args.output, args.n_pairs, args.speech_dir, args.noise_dir, args.rir_dir)

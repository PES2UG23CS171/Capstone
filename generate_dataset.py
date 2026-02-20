#!/usr/bin/env python3
"""
=============================================================================
Synthetic Audio Dataset Generator
Project: Real-Time AI-Powered Audio Filter for Transient Noise Suppression
Methodology: "Scalable Audio Synthesis Using Room Impulse Responses"
=============================================================================

Generates (noisy_input, clean_target) pairs by:
  1. Convolving clean speech   with a NEAR-FIELD  Room Impulse Response (RIR)
  2. Convolving transient noise with a FAR-FIELD   RIR
  3. Mixing at a random SNR ∈ [-5, +20] dB
  4. Normalising to prevent clipping

Author : Dhrus (Capstone 2026)
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

# Optional: pyroomacoustics for synthetic RIR generation
try:
    import pyroomacoustics as pra

    PRA_AVAILABLE = True
except ImportError:
    PRA_AVAILABLE = False

# ---------------------------------------------------------------------------
#  CONFIGURATION  –  edit paths / parameters here
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Central configuration – all tuneable knobs in one place."""

    # ── Input Paths ──────────────────────────────────────────────────────
    # Path to LibriSpeech root (e.g. .../LibriSpeech/train-clean-100)
    speech_dir: str = "./data/LibriSpeech/train-clean-100"

    # Path to folder of transient‐noise .wav / .flac files (FreeSound)
    noise_dir: str = "./data/FreeSound/transient_noises"

    # Path to folder of RIR .wav files (OpenAIR or similar).
    # If empty **and** pyroomacoustics is installed, synthetic RIRs are used.
    rir_dir: str = "./data/RIRs"

    # ── Output ───────────────────────────────────────────────────────────
    output_dir: str = "./dataset"

    # ── Scale ────────────────────────────────────────────────────────────
    total_samples: int = 10_000
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # ── Audio parameters ─────────────────────────────────────────────────
    target_sr: int = 48_000          # Standardise everything to 48 kHz
    segment_duration: float = 4.0    # seconds per sample
    segment_samples: int = field(init=False)

    # ── SNR range (dB) ───────────────────────────────────────────────────
    snr_min: float = -5.0
    snr_max: float = 20.0

    # ── Synthetic RIR defaults (pyroomacoustics) ─────────────────────────
    room_dim_range: Tuple[Tuple[float, float], ...] = (
        (4.0, 10.0),   # x  metres
        (4.0, 8.0),    # y
        (2.5, 4.0),    # z
    )
    rt60_range: Tuple[float, float] = (0.15, 0.9)    # seconds
    nearfield_dist: Tuple[float, float] = (0.3, 1.0)  # metres from mic
    farfield_dist: Tuple[float, float] = (2.0, 6.0)

    # ── Misc ─────────────────────────────────────────────────────────────
    seed: int = 42
    num_workers: int = 1           # for future multiprocessing extension
    output_format: str = "wav"     # wav | flac
    output_subtype: str = "PCM_16" # PCM_16 | FLOAT
    peak_norm_target: float = 0.95 # peak-normalise final mix to this
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.segment_samples = int(self.target_sr * self.segment_duration)
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"


# ---------------------------------------------------------------------------
#  LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dataset_gen")


# ---------------------------------------------------------------------------
#  AUDIO I/O UTILITIES
# ---------------------------------------------------------------------------

def load_audio(path: str | Path, target_sr: int) -> np.ndarray:
    """Load an audio file, resample to *target_sr*, return mono float64."""
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    # Convert to mono by averaging channels
    data = data.mean(axis=1)

    if sr != target_sr:
        # Resample using linear interpolation (fast, good enough for dataset gen)
        try:
            import soxr
            data = soxr.resample(data, sr, target_sr, quality="HQ")
        except ImportError:
            # Fall back to scipy
            from scipy.signal import resample_poly
            gcd = math.gcd(target_sr, sr)
            data = resample_poly(data, target_sr // gcd, sr // gcd)

    return data


def save_audio(path: str | Path, data: np.ndarray, sr: int,
               subtype: str = "PCM_16") -> None:
    """Save a 1-D float array as an audio file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype=subtype)


# ---------------------------------------------------------------------------
#  FILE DISCOVERY
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


def discover_audio_files(root: str | Path) -> List[Path]:
    """Recursively find all audio files under *root*."""
    root = Path(root)
    if not root.exists():
        log.error("Directory does not exist: %s", root)
        return []
    files = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()
    )
    return files


# ---------------------------------------------------------------------------
#  ROOM IMPULSE RESPONSE (RIR) HANDLING
# ---------------------------------------------------------------------------

class RIRProvider:
    """Provides Room Impulse Responses – either from disk or synthesised."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.file_rirs: List[Path] = []
        self.use_synthetic = False

        rir_path = Path(cfg.rir_dir)
        if rir_path.exists():
            self.file_rirs = discover_audio_files(rir_path)

        if len(self.file_rirs) >= 2:
            log.info("Found %d RIR files on disk.", len(self.file_rirs))
        elif PRA_AVAILABLE:
            log.info("Using pyroomacoustics for synthetic RIR generation.")
            self.use_synthetic = True
        else:
            raise RuntimeError(
                "No RIR files found and pyroomacoustics is not installed.\n"
                "Either populate the RIR directory or:  pip install pyroomacoustics"
            )

    # ── Disk-based RIR ───────────────────────────────────────────────────

    def _load_random_rir_file(self, rng: random.Random) -> np.ndarray:
        path = rng.choice(self.file_rirs)
        return load_audio(path, self.cfg.target_sr)

    # ── Synthetic RIR ────────────────────────────────────────────────────

    def _make_synthetic_pair(
        self, rng: random.Random
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a (near-field, far-field) RIR pair in the same room."""
        dim_ranges = self.cfg.room_dim_range
        room_dim = [rng.uniform(*r) for r in dim_ranges]

        rt60 = rng.uniform(*self.cfg.rt60_range)

        # pyroomacoustics: e_absorption and max_order from RT60
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        if max_order > 50:
            max_order = 50  # cap for speed

        room = pra.ShoeBox(
            room_dim,
            fs=self.cfg.target_sr,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )

        # Mic in centre-ish of room
        mic_pos = [d / 2 + rng.uniform(-0.5, 0.5) for d in room_dim]
        room.add_microphone(mic_pos)

        # Near-field source
        nf_dist = rng.uniform(*self.cfg.nearfield_dist)
        nf_angle = rng.uniform(0, 2 * math.pi)
        nf_pos = [
            mic_pos[0] + nf_dist * math.cos(nf_angle),
            mic_pos[1] + nf_dist * math.sin(nf_angle),
            mic_pos[2] + rng.uniform(-0.3, 0.3),
        ]
        # Clamp inside room
        nf_pos = [max(0.1, min(p, d - 0.1)) for p, d in zip(nf_pos, room_dim)]

        # Far-field source
        ff_dist = rng.uniform(*self.cfg.farfield_dist)
        ff_angle = rng.uniform(0, 2 * math.pi)
        ff_pos = [
            mic_pos[0] + ff_dist * math.cos(ff_angle),
            mic_pos[1] + ff_dist * math.sin(ff_angle),
            mic_pos[2] + rng.uniform(-0.5, 0.5),
        ]
        ff_pos = [max(0.1, min(p, d - 0.1)) for p, d in zip(ff_pos, room_dim)]

        # Add dummy signals so pyroomacoustics computes RIRs
        dummy = np.array([1.0])
        room.add_source(nf_pos, signal=dummy)
        room.add_source(ff_pos, signal=dummy)

        room.compute_rir()

        rir_near = room.rir[0][0]  # mic 0, source 0
        rir_far = room.rir[0][1]   # mic 0, source 1

        return rir_near.astype(np.float64), rir_far.astype(np.float64)

    # ── Public API ───────────────────────────────────────────────────────

    def get_rir_pair(
        self, rng: random.Random
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (near_field_rir, far_field_rir)."""
        if self.use_synthetic:
            return self._make_synthetic_pair(rng)
        else:
            # Pick two distinct files at random
            rir_near = self._load_random_rir_file(rng)
            rir_far = self._load_random_rir_file(rng)
            return rir_near, rir_far


# ---------------------------------------------------------------------------
#  DSP UTILITIES
# ---------------------------------------------------------------------------

def rms(x: np.ndarray) -> float:
    """Root-mean-square of signal *x*."""
    val = np.sqrt(np.mean(x ** 2))
    return float(val) if val > 0 else 1e-12


def convolve_rir(signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve *signal* with *rir* using FFT, return same-length result."""
    out = fftconvolve(signal, rir, mode="full")[: len(signal)]
    return out


def random_segment(audio: np.ndarray, length: int,
                   rng: random.Random) -> np.ndarray:
    """Extract a random segment of *length* samples, zero-pad if short."""
    if len(audio) >= length:
        start = rng.randint(0, len(audio) - length)
        return audio[start: start + length].copy()
    else:
        # Zero-pad short audio
        padded = np.zeros(length, dtype=audio.dtype)
        start = rng.randint(0, length - len(audio))
        padded[start: start + len(audio)] = audio
        return padded


def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mix *clean* and *noise* at the given SNR (dB).

    Returns (noisy_mix, clean_target).
    The clean_target is the convolved-clean signal (the model's learning target).
    """
    clean_rms = rms(clean)
    noise_rms = rms(noise)

    # desired_noise_rms = clean_rms / 10^(snr_db / 20)
    desired_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))

    if noise_rms > 0:
        gain = desired_noise_rms / noise_rms
    else:
        gain = 0.0

    scaled_noise = noise * gain
    noisy_mix = clean + scaled_noise
    return noisy_mix, clean


def peak_normalise(x: np.ndarray, target: float = 0.95) -> np.ndarray:
    """Scale so that max|x| == target. Prevents clipping."""
    peak = np.max(np.abs(x))
    if peak < 1e-8:
        return x
    return x * (target / peak)


def apply_same_normalisation(
    noisy: np.ndarray, clean: np.ndarray, target: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalise *noisy* and *clean* by the **same** gain factor so that the
    noisy signal peaks at *target*.  This keeps the relative level intact
    for the model.
    """
    peak = np.max(np.abs(noisy))
    if peak < 1e-8:
        return noisy, clean
    gain = target / peak
    return noisy * gain, clean * gain


# ---------------------------------------------------------------------------
#  DATASET BUILDER
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """Orchestrates the full dataset-generation pipeline."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        np.random.seed(cfg.seed)

        log.info("Discovering audio files …")
        self.speech_files = discover_audio_files(cfg.speech_dir)
        self.noise_files = discover_audio_files(cfg.noise_dir)

        if not self.speech_files:
            raise FileNotFoundError(
                f"No speech files found under {cfg.speech_dir}"
            )
        if not self.noise_files:
            raise FileNotFoundError(
                f"No noise files found under {cfg.noise_dir}"
            )

        log.info("  Speech files : %d", len(self.speech_files))
        log.info("  Noise  files : %d", len(self.noise_files))

        self.rir_provider = RIRProvider(cfg)

    # ── Single-sample generation ─────────────────────────────────────────

    def _generate_one(self, idx: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate one (noisy, clean) pair.

        Returns (noisy, clean, metadata_dict).
        """
        seg_len = self.cfg.segment_samples

        # 1. Pick random speech & noise files
        speech_path = self.rng.choice(self.speech_files)
        noise_path = self.rng.choice(self.noise_files)

        speech_raw = load_audio(speech_path, self.cfg.target_sr)
        noise_raw = load_audio(noise_path, self.cfg.target_sr)

        # 2. Extract fixed-length segments
        speech_seg = random_segment(speech_raw, seg_len, self.rng)
        noise_seg = random_segment(noise_raw, seg_len, self.rng)

        # 3. Get an RIR pair (near‐field for speech, far‐field for noise)
        rir_near, rir_far = self.rir_provider.get_rir_pair(self.rng)

        # 4. Spatial convolution  → acoustic realism
        speech_conv = convolve_rir(speech_seg, rir_near)
        noise_conv = convolve_rir(noise_seg, rir_far)

        # 5. Random SNR
        snr_db = self.rng.uniform(self.cfg.snr_min, self.cfg.snr_max)

        # 6. Mix
        noisy_mix, clean_target = mix_at_snr(speech_conv, noise_conv, snr_db)

        # 7. Joint peak normalisation (same gain for both so model target stays valid)
        noisy_mix, clean_target = apply_same_normalisation(
            noisy_mix, clean_target, target=self.cfg.peak_norm_target
        )

        metadata = {
            "index": idx,
            "speech_file": str(speech_path),
            "noise_file": str(noise_path),
            "snr_db": round(snr_db, 2),
            "rir_type": "synthetic" if self.rir_provider.use_synthetic else "file",
        }

        return noisy_mix, clean_target, metadata

    # ── Full pipeline ────────────────────────────────────────────────────

    def build(self) -> None:
        cfg = self.cfg
        total = cfg.total_samples
        n_train = int(total * cfg.train_ratio)
        n_val = int(total * cfg.val_ratio)
        n_test = total - n_train - n_val

        split_map: List[Tuple[str, int]] = []
        split_map += [("train", i) for i in range(n_train)]
        split_map += [("val", i) for i in range(n_val)]
        split_map += [("test", i) for i in range(n_test)]

        # Shuffle so that speech/noise combos are not clustered by split
        self.rng.shuffle(split_map)

        log.info(
            "Generating %d samples  (train=%d, val=%d, test=%d) …",
            total, n_train, n_val, n_test,
        )

        all_metadata: dict = {"train": [], "val": [], "test": []}
        counters = {"train": 0, "val": 0, "test": 0}

        for global_idx, (split, _) in enumerate(split_map):
            try:
                noisy, clean, meta = self._generate_one(global_idx)
            except Exception as exc:
                log.warning(
                    "Sample %d failed (%s). Skipping.", global_idx, exc
                )
                continue

            local_idx = counters[split]
            counters[split] += 1

            fname = f"{local_idx:06d}.{cfg.output_format}"

            noisy_path = Path(cfg.output_dir) / split / "noisy" / fname
            clean_path = Path(cfg.output_dir) / split / "clean" / fname

            save_audio(noisy_path, noisy, cfg.target_sr, cfg.output_subtype)
            save_audio(clean_path, clean, cfg.target_sr, cfg.output_subtype)

            meta["noisy_file"] = str(noisy_path)
            meta["clean_file"] = str(clean_path)
            all_metadata[split].append(meta)

            if (global_idx + 1) % 100 == 0 or (global_idx + 1) == total:
                log.info(
                    "  Progress: %d / %d  (%.1f%%)",
                    global_idx + 1, total, 100.0 * (global_idx + 1) / total,
                )

        # Save metadata JSON
        meta_path = Path(cfg.output_dir) / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "config": asdict(cfg),
                    "splits": {
                        k: {"count": len(v), "samples": v}
                        for k, v in all_metadata.items()
                    },
                },
                f,
                indent=2,
                default=str,
            )
        log.info("Metadata saved → %s", meta_path)
        log.info(
            "Done!  train=%d  val=%d  test=%d",
            counters["train"], counters["val"], counters["test"],
        )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Generate synthetic noisy/clean audio pairs for transient-noise suppression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--speech-dir", type=str, default=Config.speech_dir,
                        help="Path to LibriSpeech root folder.")
    parser.add_argument("--noise-dir", type=str, default=Config.noise_dir,
                        help="Path to FreeSound transient-noise folder.")
    parser.add_argument("--rir-dir", type=str, default=Config.rir_dir,
                        help="Path to RIR wav files (or empty for synthetic).")
    parser.add_argument("--output-dir", type=str, default=Config.output_dir,
                        help="Output dataset root.")
    parser.add_argument("--total-samples", type=int,
                        default=Config.total_samples,
                        help="Total number of (noisy, clean) pairs.")
    parser.add_argument("--segment-duration", type=float,
                        default=Config.segment_duration,
                        help="Duration of each sample in seconds.")
    parser.add_argument("--target-sr", type=int, default=Config.target_sr,
                        help="Target sample rate (Hz).")
    parser.add_argument("--snr-min", type=float, default=Config.snr_min,
                        help="Minimum SNR in dB.")
    parser.add_argument("--snr-max", type=float, default=Config.snr_max,
                        help="Maximum SNR in dB.")
    parser.add_argument("--seed", type=int, default=Config.seed,
                        help="Random seed for reproducibility.")
    parser.add_argument("--output-format", type=str, default=Config.output_format,
                        choices=["wav", "flac"],
                        help="Output audio format.")
    parser.add_argument("--peak-norm", type=float,
                        default=Config.peak_norm_target,
                        help="Peak-normalisation target (0–1).")

    args = parser.parse_args()

    cfg = Config(
        speech_dir=args.speech_dir,
        noise_dir=args.noise_dir,
        rir_dir=args.rir_dir,
        output_dir=args.output_dir,
        total_samples=args.total_samples,
        segment_duration=args.segment_duration,
        target_sr=args.target_sr,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        seed=args.seed,
        output_format=args.output_format,
        peak_norm_target=args.peak_norm,
    )
    return cfg


# ---------------------------------------------------------------------------
#  ENTRYPOINT
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = parse_args()
    log.setLevel(cfg.log_level)

    log.info("=" * 60)
    log.info("  Synthetic Dataset Generator – Transient Noise Suppression")
    log.info("=" * 60)
    log.info("Config:")
    for k, v in asdict(cfg).items():
        log.info("  %-22s = %s", k, v)
    log.info("-" * 60)

    builder = DatasetBuilder(cfg)
    builder.build()


if __name__ == "__main__":
    main()

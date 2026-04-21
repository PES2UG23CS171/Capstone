"""
Dataset Loader
==============
PyTorch Dataset for loading paired (noisy, clean) ``.wav`` files produced
by ``generate_dataset.py``.

Expected directory layout::

    dataset/
    ├── train/
    │   ├── noisy/  000000.wav, 000001.wav, …
    │   └── clean/  000000.wav, 000001.wav, …
    ├── val/
    │   ├── noisy/
    │   └── clean/
    └── test/
        ├── noisy/
        └── clean/
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

import config as cfg


class TransientNoiseDataset(Dataset):
    """PyTorch Dataset for transient noise suppression training.

    Parameters
    ----------
    dataset_dir : str
        Root path containing ``train/``, ``val/``, ``test/`` sub-dirs.
    split : str
        One of ``'train'``, ``'val'``, ``'test'``.
    context_window : int
        Audio window size to crop for each sample.
    augment : bool
        Enable data augmentation (train only).
    cache_in_memory : bool
        If True, load all audio into RAM on first access for faster
        subsequent epochs.  Uses ~6 GB for 8K samples @ 48 kHz × 4 s.
    """

    def __init__(
        self,
        dataset_dir: str = cfg.DATASET_DIR,
        split: str = "train",
        context_window: int = cfg.CONTEXT_WINDOW_SAMPLES,
        augment: bool = True,
        cache_in_memory: bool = False,
    ) -> None:
        super().__init__()
        self.context_window = context_window
        self.augment = augment and (split == "train")
        self.cache_in_memory = cache_in_memory

        # Discover paired .wav files
        noisy_dir = Path(dataset_dir) / split / "noisy"
        clean_dir = Path(dataset_dir) / split / "clean"

        if not noisy_dir.exists():
            raise FileNotFoundError(f"Noisy directory not found: {noisy_dir}")
        if not clean_dir.exists():
            raise FileNotFoundError(f"Clean directory not found: {clean_dir}")

        # Sort to ensure alignment between noisy and clean
        noisy_files = sorted(noisy_dir.glob("*.wav"))
        clean_files = sorted(clean_dir.glob("*.wav"))

        if not noisy_files:
            raise FileNotFoundError(f"No .wav files found in {noisy_dir}")

        # Verify pairing by filename
        noisy_names = {f.name for f in noisy_files}
        clean_names = {f.name for f in clean_files}
        common = sorted(noisy_names & clean_names)

        if not common:
            raise FileNotFoundError(
                "No matching filenames between noisy/ and clean/ directories"
            )

        self.noisy_files = [noisy_dir / name for name in common]
        self.clean_files = [clean_dir / name for name in common]

        # Optional in-memory cache
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.noisy_files)

    def _load_audio(self, path: Path) -> np.ndarray:
        """Load a .wav file as float32 mono."""
        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load or retrieve from cache
        if self.cache_in_memory and idx in self._cache:
            noisy, clean = self._cache[idx]
        else:
            noisy = self._load_audio(self.noisy_files[idx])
            clean = self._load_audio(self.clean_files[idx])
            if self.cache_in_memory:
                self._cache[idx] = (noisy.copy(), clean.copy())

        # Random crop to context_window
        T = len(noisy)
        if T > self.context_window:
            start = random.randint(0, T - self.context_window)
            noisy = noisy[start:start + self.context_window]
            clean = clean[start:start + self.context_window]
        elif T < self.context_window:
            pad_n = np.zeros(self.context_window, dtype=np.float32)
            pad_n[:T] = noisy
            noisy = pad_n
            pad_c = np.zeros(self.context_window, dtype=np.float32)
            pad_c[:T] = clean
            clean = pad_c

        # Data augmentation (train only)
        if self.augment:
            # Random polarity inversion
            if random.random() < 0.5:
                noisy = -noisy
                clean = -clean

            # Random gain ±3 dB
            gain_db = random.uniform(-3, 3)
            gain = 10.0 ** (gain_db / 20.0)
            noisy = noisy * gain
            clean = clean * gain

        return torch.from_numpy(noisy), torch.from_numpy(clean)

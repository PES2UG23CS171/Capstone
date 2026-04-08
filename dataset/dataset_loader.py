"""
Dataset Loader
==============
PyTorch Dataset for loading synthetic (noisy, clean) audio pairs from
``.npz`` files produced by ``generate_dataset.py``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import config as cfg


class TransientNoiseDataset(Dataset):
    """PyTorch Dataset for transient noise suppression training.

    Parameters
    ----------
    dataset_dir : str
        Path to directory containing ``.npz`` files.
    split : str
        ``'train'`` or ``'val'``.
    val_ratio : float
        Fraction of files reserved for validation.
    context_window : int
        Audio window size to crop for each sample.
    augment : bool
        Enable data augmentation (train only).
    """

    def __init__(
        self,
        dataset_dir: str = cfg.DATASET_DIR,
        split: str = "train",
        val_ratio: float = 0.1,
        context_window: int = cfg.CONTEXT_WINDOW_SAMPLES,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.context_window = context_window
        self.augment = augment and (split == "train")

        # Discover .npz files
        data_path = Path(dataset_dir)
        all_files = sorted(data_path.glob("*.npz"))

        if not all_files:
            raise FileNotFoundError(f"No .npz files found in {dataset_dir}")

        # Deterministic train/val split
        rng = random.Random(42)
        indices = list(range(len(all_files)))
        rng.shuffle(indices)

        n_val = max(1, int(len(all_files) * val_ratio))
        if split == "val":
            selected = indices[:n_val]
        else:
            selected = indices[n_val:]

        self.files = [all_files[i] for i in selected]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(str(self.files[idx]))
        noisy = data["noisy"].astype(np.float32)
        clean = data["clean"].astype(np.float32)

        # Random crop to context_window
        T = len(noisy)
        if T > self.context_window:
            start = random.randint(0, T - self.context_window)
            noisy = noisy[start:start + self.context_window]
            clean = clean[start:start + self.context_window]
        elif T < self.context_window:
            pad = np.zeros(self.context_window, dtype=np.float32)
            pad[:T] = noisy
            noisy = pad
            pad2 = np.zeros(self.context_window, dtype=np.float32)
            pad2[:T] = clean
            clean = pad2

        # Data augmentation (train only)
        if self.augment:
            # Random polarity inversion
            if random.random() < 0.5:
                noisy = -noisy
                clean = -clean

            # Random gain ±3 dB
            gain_db = random.uniform(-3, 3)
            gain = 10.0 ** (gain_db / 20.0)
            noisy *= gain
            clean *= gain

        return torch.from_numpy(noisy), torch.from_numpy(clean)

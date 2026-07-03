"""
Vendored EfficientAT mel front-end (eval-only).

The EfficientAT mn10_as AudioSet model expects log-mel features produced by its
``AugmentMelSTFT`` at **32 kHz**.  Feeding any other mel (wrong sample rate,
generic torchaudio MelSpectrogram, etc.) pushes the model out of distribution
and it saturates.  This module reproduces ``AugmentMelSTFT``'s *eval* path
exactly so the runtime classifier computes the same features the model was
trained on — without depending on the EfficientAT package at inference time.

Source: github.com/fschmid56/EfficientAT (MIT), ``models/preprocess.py``,
eval path only (no SpecAugment, fixed fmin/fmax, no preemphasis randomisation).

torch + torchaudio are imported lazily so non-EfficientAT pipelines don't pay
for them.
"""

from __future__ import annotations

from typing import Optional


class EfficientATMel:
    """Eval-only AugmentMelSTFT.  Call ``mel(samples_32k)`` → (1, n_mels, frames)."""

    def __init__(self, n_mels: int = 128, sr: int = 32000, win_length: int = 800,
                 hopsize: int = 320, n_fft: int = 1024,
                 fmin: float = 0.0, fmax: Optional[float] = None,
                 fmax_aug_range: int = 2000) -> None:
        import torch
        import torchaudio
        self._torch = torch
        self.n_mels = n_mels
        self.sr = sr
        self.win_length = win_length
        self.hopsize = hopsize
        self.n_fft = n_fft
        self.fmin = fmin
        if fmax is None:
            # Matches AugmentMelSTFT: fmax = sr//2 - fmax_aug_range//2
            fmax = sr // 2 - fmax_aug_range // 2
        self.fmax = fmax

        self._window = torch.hann_window(win_length, periodic=False)
        self._preemph = torch.as_tensor([[[-0.97, 1.0]]])
        # Precompute the (eval, fixed fmin/fmax) Kaldi mel basis once.
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels, self.n_fft, self.sr,
            self.fmin, self.fmax,
            vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0,
        )
        self._mel_basis = torch.nn.functional.pad(mel_basis, (0, 1), mode="constant", value=0)

    def __call__(self, samples_32k):
        """``samples_32k``: 1-D numpy array or torch tensor at 32 kHz.
        Returns a torch tensor of shape (1, n_mels, frames)."""
        torch = self._torch
        x = samples_32k
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.to(torch.float32).reshape(1, -1)                  # (1, T)
        # Preemphasis (conv1d with [-0.97, 1]).
        x = torch.nn.functional.conv1d(x.unsqueeze(1), self._preemph).squeeze(1)
        spec = torch.stft(
            x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
            center=True, normalized=False, window=self._window, return_complex=True,
        )
        # Power = |z|² = re² + im² (numerically identical to the old
        # return_complex=False → (...,2)→sum path, but future-proof: torch is
        # deprecating return_complex=False).
        spec = spec.real ** 2 + spec.imag ** 2                  # power, (1, F, frames)
        mel = torch.matmul(self._mel_basis, spec)              # (1, n_mels, frames)
        mel = (mel + 1e-5).log()
        mel = (mel + 4.5) / 5.0                                 # AugmentMelSTFT fast-norm
        return mel

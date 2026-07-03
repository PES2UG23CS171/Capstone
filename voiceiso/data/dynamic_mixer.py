"""
Dynamic mixer — generates (clean, noisy) pairs on the fly.

Each draw: random speech clip + random noise clip(s) + optional RIR convolution,
mixed at a random SNR.  Because mixing happens at sample time (not pre-baked),
every epoch sees new combinations → far better generalisation than a static set.

This is used both for (a) training data when fine-tuning, and (b) building
benchmark sets at controlled SNRs.  It is a torch ``IterableDataset`` when torch
is present, and also exposes a plain ``draw()`` for benchmarking without torch.
"""

from __future__ import annotations

import random
from math import gcd
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

from voiceiso.data.corpora import CorpusRegistry


# Path-substring aliases used to bucket noise files by voiceiso class for
# per-class benchmarking.  Matching is case-insensitive against the full file
# path, so it works whether corpora label by sub-folder (MUSAN/music/…) or by
# filename (…/keyboard_typing_03.wav).  Multiple aliases collapse to one class.
_CLASS_ALIASES: dict[str, tuple[str, ...]] = {
    "fan":              ("fan", "mechanical_fan"),
    "hvac":             ("hvac", "air_condition", "air conditioning", "aircon",
                         "ventilation"),
    "traffic":          ("traffic", "car", "vehicle", "engine", "truck",
                         "motorcycle", "road"),
    "keyboard":         ("keyboard", "typing", "keystroke", "typewriter"),
    "dog_bark":         ("dog", "bark", "yip", "howl"),
    "music":            ("music", "guitar", "piano", "instrument", "song", "/mus"),
    "television":       ("television", "/tv", "radio"),
    "competing_speech": ("speech", "talk", "conversation", "babble", "vocal"),
    # Extra classes the heuristic/EfficientAT classifier also know about:
    "wind":             ("wind", "rustl"),
    "door_slam":        ("door", "slam"),
    "mouse_click":      ("mouse", "click"),
}


def _load(path: Path, sr: int) -> np.ndarray:
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if file_sr != sr:
        from scipy.signal import resample_poly
        g = gcd(sr, file_sr)
        data = resample_poly(data, sr // g, file_sr // g).astype("float32")
    return data


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


class DynamicMixer:
    def __init__(self, data_root: str = "data", sr: int = 48_000,
                 segment_s: float = 4.0, snr_range: Tuple[float, float] = (-5.0, 20.0),
                 use_rir: bool = True, seed: int = 0) -> None:
        if sf is None:
            raise ImportError("soundfile is required for DynamicMixer")
        self.sr = sr
        self.n = int(segment_s * sr)
        self.snr_range = snr_range
        self.use_rir = use_rir
        self.rng = random.Random(seed)

        reg = CorpusRegistry(Path(data_root))
        self.speech: List[Path] = [f for c in reg.speech().values() for f in c.files]
        self.noise: List[Path] = [f for c in reg.noise().values() for f in c.files]
        self.rirs: List[Path] = [f for c in reg.rirs().values() for f in c.files]

    def available(self) -> bool:
        return bool(self.speech and self.noise)

    # ── per-class noise selection (for per-class benchmarking) ───────────
    def noise_for_class(self, noise_class: str) -> List[Path]:
        """Return the subset of noise files whose path matches ``noise_class``."""
        aliases = _CLASS_ALIASES.get(noise_class)
        if not aliases:
            return []
        out = []
        for p in self.noise:
            ps = str(p).lower()
            if any(a in ps for a in aliases):
                out.append(p)
        return out

    def available_classes(self) -> dict[str, int]:
        """Map each known noise class → count of matching files present locally."""
        return {c: len(self.noise_for_class(c)) for c in _CLASS_ALIASES}

    def _crop(self, x: np.ndarray) -> np.ndarray:
        if len(x) >= self.n:
            s = self.rng.randint(0, len(x) - self.n)
            return x[s:s + self.n].copy()
        out = np.zeros(self.n, dtype="float32")
        out[: len(x)] = x
        return out

    def draw(self, noise_class: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (reference, noisy) of length ``segment_s``.

        The reference is the noise-free *target* the enhancer is expected to
        recover.  When RIR convolution is on, that target is the REVERBERANT
        speech (``target = clean * rir``), because DFN3 is a denoiser, not a
        dereverberator — scoring its output against the DRY clean would penalise
        reverberation it was never asked to remove (a perfect enhancer would
        score hugely negative SI-SDR).  So the reference matches the noisy
        signal's reverberation; only the additive noise differs.

        ``noise_class`` (optional) restricts the noise draw to files matching
        that class (per :data:`_CLASS_ALIASES`).  Falls back to the full noise
        pool if no files match the class.
        """
        clean = self._crop(_load(self.rng.choice(self.speech), self.sr))
        pool = self.noise_for_class(noise_class) if noise_class else self.noise
        if not pool:
            pool = self.noise
        noise = self._crop(_load(self.rng.choice(pool), self.sr))

        target = clean
        if self.use_rir and self.rirs:
            from scipy.signal import fftconvolve
            rir = _load(self.rng.choice(self.rirs), self.sr)
            wet = fftconvolve(clean, rir)[: self.n].astype("float32")
            target = wet  # reverberant speech is the realistic mic signal

        snr = self.rng.uniform(*self.snr_range)
        g = _rms(target) / (_rms(noise) * 10 ** (snr / 20.0))
        noisy = (target + g * noise).astype("float32")
        # Reference is the (possibly reverberant) target — NOT the dry clean.
        peak = max(np.abs(noisy).max(), np.abs(target).max(), 1e-6)
        if peak > 0.98:
            target = target * (0.98 / peak); noisy = noisy * (0.98 / peak)
        return target.astype("float32"), noisy.astype("float32")

    def build_benchmark_set(self, n_pairs: int = 20,
                            noise_class: Optional[str] = None
                            ) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [self.draw(noise_class=noise_class) for _ in range(n_pairs)]

"""
Corpus registry for the redesigned dataset.

The current capstone trains on ~10k *static* synthetic pairs — far too small and
too homogeneous to approach commercial robustness.  Commercial systems train on
**hundreds of hours, thousands of speakers, thousands of noise types**, mixed
*dynamically* (a fresh SNR/RIR every epoch so the model never memorises a pair).

Recommended open corpora (place under ``data/`` and point the registry here):

  Speech
    * DNS-Challenge 5 (ICASSP 2023) read-speech  — ~750 h, multilingual
    * LibriSpeech / LibriTTS                      — clean read speech
    * VCTK                                        — 110 speakers, accents
  Noise
    * DNS-Challenge noise set (AudioSet-derived)  — broad real noise
    * MUSAN (music/speech/noise)                  — music + babble
    * DEMAND                                      — 18 real environments
    * WHAM!                                       — real ambient (restaurant/…)
    * FSD50K                                      — 200 labelled sound classes
                                                    (drives noise-classifier labels)
  Reverberation
    * OpenSLR SLR28 RIR set / SoundSpaces / pyroomacoustics-generated RIRs

Targets: ≥ 500 h speech, ≥ 2 000 speakers, ≥ 1 000 noise clips across the
classifier's classes, RT60 0.1–1.0 s.  Expected gain over the current 10k static
set: large — better generalisation to unseen speakers/noises and far fewer
artifacts (the single biggest data lever after adopting DFN3).

This module only *indexes* whatever is present locally; it downloads nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

_AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")


@dataclass
class Corpus:
    name: str
    root: Path
    files: List[Path] = field(default_factory=list)

    def scan(self) -> "Corpus":
        if self.root.exists():
            self.files = [p for p in self.root.rglob("*") if p.suffix.lower() in _AUDIO_EXT]
        return self


@dataclass
class CorpusRegistry:
    data_root: Path

    # Expected sub-directory names (override as needed).
    speech_dirs: tuple = ("DNS5/clean", "LibriSpeech", "LibriTTS", "VCTK", "dataset/train/clean")
    noise_dirs: tuple = ("DNS5/noise", "MUSAN", "DEMAND", "WHAM", "FSD50K")
    rir_dirs: tuple = ("RIRS", "SLR28", "rirs")

    def _collect(self, names) -> Dict[str, Corpus]:
        out: Dict[str, Corpus] = {}
        for n in names:
            c = Corpus(n, self.data_root / n).scan()
            if c.files:
                out[n] = c
        return out

    def speech(self) -> Dict[str, Corpus]:
        return self._collect(self.speech_dirs)

    def noise(self) -> Dict[str, Corpus]:
        return self._collect(self.noise_dirs)

    def rirs(self) -> Dict[str, Corpus]:
        return self._collect(self.rir_dirs)

    def summary(self) -> str:
        sp = self.speech(); ns = self.noise(); rr = self.rirs()
        def tot(d): return sum(len(c.files) for c in d.values())
        return (f"speech: {tot(sp)} files across {list(sp)}\n"
                f"noise : {tot(ns)} files across {list(ns)}\n"
                f"rirs  : {tot(rr)} files across {list(rr)}")

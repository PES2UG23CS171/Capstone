"""
StreamingPipeline — wires all stages into the commercial-style chain.

V2 order::

    mic block
      → Preprocessing            (DC/rumble removal, SNR estimate)
      → AEC                       (echo cancel + DTD + RES coherence flags)
      → VAD                       (speech probability + hangover)
      → SpeakerEmbedder           (target-speaker cosine similarity)
      → NoiseClassifier           (what kind of noise?)
      → CompetingSpeech           (consume sim → mid-band cut hint)
      → DynamicController         (suppression + per-band gains)
      → SpeechPreservation        (consonant cues + fused speech_conf)
      → Enhancement (DFN3)        (the heavy lifting + over-suppression rollback)
      → MultiBandGainModulator    (per-band gain applied AFTER DFN3)
      → TinyGRUPostFilter         (learned residual cleanup — passthrough w/o ckpt)
      → PostFilter                (residual gate + class-shaped comfort noise)
      → output block

Process one block at a time via :meth:`process_block`.  Stages keep their own
streaming state, so feeding consecutive blocks is real-time-safe.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages.aec import AEC
from voiceiso.stages.base import FrameContext, Stage
from voiceiso.stages.competing_speech import CompetingSpeech
from voiceiso.stages.controller import DynamicController
from voiceiso.stages.enhancement import Enhancement
from voiceiso.stages.multiband import MultiBandGainModulator
from voiceiso.stages.noise_classifier import HeuristicNoiseClassifier
from voiceiso.stages.postfilter import PostFilter
from voiceiso.stages.preprocessing import Preprocessing
from voiceiso.stages.speaker_embedder import SpeakerEmbedder
from voiceiso.stages.speech_preservation import SpeechPreservation
from voiceiso.stages.tiny_postfilter import TinyGRUPostFilter
from voiceiso.stages.vad import VAD


class StreamingPipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None, enh_threads: int = 4) -> None:
        self.cfg = cfg or PipelineConfig()
        self.preprocessing = Preprocessing(self.cfg)
        self.aec = AEC(self.cfg)
        self.vad = VAD(self.cfg)
        self.speaker_embedder = SpeakerEmbedder(self.cfg)
        self.classifier = HeuristicNoiseClassifier(self.cfg)
        self.competing = CompetingSpeech(self.cfg)
        self.controller = DynamicController(self.cfg)
        self.preservation = SpeechPreservation(self.cfg)
        self.enhancement = Enhancement(self.cfg, threads=enh_threads)
        self.multiband = MultiBandGainModulator(self.cfg)
        self.tiny_postfilter = TinyGRUPostFilter(self.cfg)
        self.postfilter = PostFilter(self.cfg)

        self.stages: List[Stage] = [
            self.preprocessing,
            self.aec,
            self.vad,
            self.speaker_embedder,
            self.classifier,
            self.competing,
            self.controller,
            self.preservation,
            self.enhancement,
            self.multiband,
            self.tiny_postfilter,
            self.postfilter,
        ]

    @property
    def backend_summary(self) -> dict:
        return {
            "vad": self.vad.backend,
            "enhancement": self.enhancement.backend,
            "speaker_embedder": self.speaker_embedder.backend,
            "tiny_postfilter": self.tiny_postfilter.backend,
            "aec_enabled": self.aec.enabled,
            "algorithmic_latency_ms": round(self.cfg.algorithmic_latency_ms, 1),
        }

    def reset(self) -> None:
        for s in self.stages:
            s.reset()

    def process_block(
        self, block: np.ndarray, reference: Optional[np.ndarray] = None
    ) -> FrameContext:
        ctx = FrameContext(
            audio=np.ascontiguousarray(block, dtype=np.float32),
            sample_rate=self.cfg.sample_rate,
            reference=None if reference is None else np.asarray(reference, dtype=np.float32),
        )
        for stage in self.stages:
            ctx = stage(ctx)
        return ctx

    def process_signal(self, x: np.ndarray, block: Optional[int] = None) -> np.ndarray:
        """Offline helper: stream a whole signal through in blocks.

        Default block size matches the live-streaming block (20 ms) so offline
        results are representative of real-time behaviour.
        """
        block = block or self.cfg.hop * 2            # 20 ms default (2 × 10 ms hop)
        out = np.zeros(len(x), dtype=np.float32)
        for s in range(0, len(x), block):
            seg = x[s : s + block]
            ctx = self.process_block(seg)
            out[s : s + len(seg)] = ctx.audio[: len(seg)]
        return out

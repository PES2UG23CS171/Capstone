"""
StreamingPipeline — wires all stages into the commercial-style chain.

V2 order::

    mic block
      → Preprocessing            (DC/rumble removal, SNR estimate)
      → AEC                       (echo cancel + DTD + RES coherence flags)
      → VAD                       (adaptive threshold + adaptive hangover)
      → SpeakerEmbedder           (target-speaker cosine similarity)
      → NoiseClassifier           (what kind of noise?)
      → CompetingSpeech           (consume sim → mid-band cut hint)
      → DynamicController         (suppression + per-band gains)
      → SpeechPreservation        (consonants + pitch voicing + sticky speech_conf)
      → Enhancement (DFN3)        (heavy lifting + multi-cue per-band rollback)
      → TargetSpeakerMask         (VoiceFilter-Lite — passthrough w/o ckpt+enroll)
      → MultiBandGainModulator    (per-band gain applied AFTER DFN3)
      → TinyGRUPostFilter         (learned residual cleanup — passthrough w/o ckpt)
      → PostFilter                (residual gate + class-shaped comfort noise)
      → output block

Process one block at a time via :meth:`process_block`.  Stages keep their own
streaming state, so feeding consecutive blocks is real-time-safe.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from voiceiso.config import PipelineConfig
from voiceiso.stages.aec import AEC
from voiceiso.stages.base import FrameContext, Stage
from voiceiso.stages.competing_speech import CompetingSpeech
from voiceiso.stages.controller import DynamicController
from voiceiso.stages.enhancement import Enhancement
from voiceiso.stages.multiband import MultiBandGainModulator
from voiceiso.stages.noise_classifier import (
    EfficientATNoiseClassifier,
    HeuristicNoiseClassifier,
)
from voiceiso.stages.postfilter import PostFilter
from voiceiso.stages.preprocessing import Preprocessing
from voiceiso.stages.speaker_embedder import SpeakerEmbedder
from voiceiso.stages.speech_preservation import SpeechPreservation
from voiceiso.stages.target_speaker_mask import TargetSpeakerMask
from voiceiso.stages.tiny_postfilter import TinyGRUPostFilter
from voiceiso.stages.vad import VAD


class StreamingPipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None, enh_threads: int = 4) -> None:
        self.cfg = cfg or PipelineConfig()
        self.preprocessing = Preprocessing(self.cfg)
        self.aec = AEC(self.cfg)
        self.vad = VAD(self.cfg)
        self.speaker_embedder = SpeakerEmbedder(self.cfg)
        # Primary classifier: learned EfficientAT-S when a checkpoint is
        # available and onnxruntime is installed.  Fall back to the
        # heuristic classifier transparently — same downstream contract.
        # Catch ``Exception`` (not just RuntimeError) because
        # ``onnxruntime`` raises its own custom exception classes
        # (InvalidProtobuf, InvalidGraph, NoSuchFile, …) that inherit
        # from Exception directly, not RuntimeError.  A corrupted or
        # schema-mismatched ONNX file would otherwise crash
        # StreamingPipeline construction.
        self.classifier_fallback_reason: Optional[str] = None
        try:
            self.classifier: Stage = EfficientATNoiseClassifier(self.cfg)
            self.classifier_backend = "efficientat"
            logger.info(
                "noise classifier: EfficientAT (learned) — %s",
                self.cfg.noise_classifier_model_path,
            )
        except Exception as exc:
            self.classifier = HeuristicNoiseClassifier(self.cfg)
            self.classifier_backend = "heuristic"
            self.classifier_fallback_reason = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "noise classifier: FALLING BACK to heuristic — the learned "
                "EfficientAT path is unavailable (%s). Set "
                "cfg.noise_classifier_model_path to a valid head ONNX and "
                "install onnxruntime to enable it.",
                self.classifier_fallback_reason,
            )
        self.competing = CompetingSpeech(self.cfg)
        self.controller = DynamicController(self.cfg)
        self.preservation = SpeechPreservation(self.cfg)
        self.enhancement = Enhancement(self.cfg, threads=enh_threads)
        self.target_speaker_mask = TargetSpeakerMask(self.cfg)
        self.multiband = MultiBandGainModulator(self.cfg)
        self.tiny_postfilter = TinyGRUPostFilter(self.cfg)
        self.postfilter = PostFilter(self.cfg)

        self._nan_warned = False
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
            self.target_speaker_mask,
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
            "target_speaker_mask": self.target_speaker_mask.backend,
            "tiny_postfilter": self.tiny_postfilter.backend,
            "noise_classifier": self.classifier_backend,
            "noise_classifier_fallback": self.classifier_fallback_reason,
            "aec_enabled": self.aec.enabled,
            "algorithmic_latency_ms": round(self.cfg.algorithmic_latency_ms, 1),
        }

    def reset(self) -> None:
        for s in self.stages:
            s.reset()

    def warmup(self, seconds: float = 4.6) -> None:
        """Prime every model before real-time audio starts.

        First-call costs (torch graph autotune, ONNX session optimisation,
        classifier first inference at the 4 s window fill) otherwise land on
        the live stream: the DSP worker stalls 1–2 s, the callback plays
        zeros, and the output queue parks at full depth — permanently
        ratcheting mouth-to-ear latency.  Run BEFORE ``stream.start()``.

        Feeds ``seconds`` of silence through the full chain (warms VAD +
        classifier ONNX, fills the classifier window so its first inference
        happens here, not live) and then warms the DFN3 path directly (the
        silence blocks bypass it by design).  Deliberately does NOT reset
        afterwards — the warmed buffers/state are the point.
        """
        block = int(round(self.cfg.sample_rate * 0.1))
        zeros = np.zeros(block, dtype=np.float32)
        for _ in range(int(np.ceil(seconds / 0.1))):
            self.process_block(zeros)
        warm = getattr(self.enhancement, "warmup", None)
        if callable(warm):
            warm()

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
        # Feedback hook: the Preprocessing stage uses the previous frame's
        # noise_class to pick its HP cutoff (default 25 Hz, or 80 Hz under
        # wind).  Pipeline writes back here so the next block sees the hint.
        self.preprocessing.set_prev_noise_class(ctx.noise_class)
        # NaN/Inf guard: np.clip passes NaN through, so a single numerical
        # edge case anywhere in the chain would otherwise reach the speakers.
        if not np.all(np.isfinite(ctx.audio)):
            if not self._nan_warned:
                self._nan_warned = True
                logger.warning("pipeline emitted non-finite samples — sanitised "
                               "(this indicates a stage bug; logged once)")
            ctx.audio = np.nan_to_num(ctx.audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return ctx

    def process_signal(self, x: np.ndarray, block: Optional[int] = None) -> np.ndarray:
        """Offline helper: stream a whole signal through in blocks.

        Default block size is 100 ms — the SAME operating point as the live
        paths (LiveStream, desktop app) and the benchmark, so offline results
        are representative of real-time behaviour.  (20 ms blocks give DFN3
        so little per-call context that SI-SDRi goes negative — never the
        default.)
        """
        block = block or int(round(self.cfg.sample_rate * 0.1))   # 100 ms
        out = np.zeros(len(x), dtype=np.float32)
        for s in range(0, len(x), block):
            seg = x[s : s + block]
            ctx = self.process_block(seg)
            out[s : s + len(seg)] = ctx.audio[: len(seg)]
        return out

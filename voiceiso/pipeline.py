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
        self._nan_streak = 0
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
            "noise_classifier_head": (
                __import__("os").path.basename(self.cfg.noise_classifier_model_path)
                if self.cfg.noise_classifier_model_path else None
            ),
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
        classifier ONNX and triggers the classifier's first inference here,
        not live) and then warms the DFN3 path directly (the silence blocks
        bypass it by design).

        Afterwards the DSP state that the silence POISONS is cleared while
        the warmed sessions/models are kept: with 4.6 s of digital zeros in
        the preprocessing noise-floor window, the first seconds of real audio
        measure an absurd ~60 dB SNR (noise floor = silence), the controller
        holds suppression near zero, and the session audibly opens
        under-suppressed for ~5 s (fan rooms sound untouched).  Clearing the
        preprocessing / classifier / controller state costs nothing — those
        resets touch numpy state only, never the loaded models — and the
        classifier re-labels real audio from ~1 s via the partial-buffer
        tile-pad path instead of arguing with 3 s of buffered silence.
        """
        block = int(round(self.cfg.sample_rate * 0.1))
        zeros = np.zeros(block, dtype=np.float32)
        for _ in range(int(np.ceil(seconds / 0.1))):
            self.process_block(zeros)
        warm = getattr(self.enhancement, "warmup", None)
        if callable(warm):
            warm()
        # Clear silence-poisoned DSP state; models stay warm (these resets
        # are numpy-only — Enhancement.reset() is NOT called so its torch
        # first-call warm-up is preserved, and its history rebuilds in 80 ms).
        self.preprocessing.reset()
        self.classifier.reset()
        self.controller.reset()

    def process_block(
        self, block: np.ndarray, reference: Optional[np.ndarray] = None
    ) -> FrameContext:
        audio = np.ascontiguousarray(block, dtype=np.float32)
        # INPUT-side finiteness guard.  A single NaN/Inf sample entering the
        # chain poisons every stage's recursive state (IIR zi, EWMAs, delay
        # lines, GRU context, classifier ring buffer) — measured: one bad
        # sample → permanent digital-zero output with NO exception, which the
        # live containment (dry-passthrough on exception) can never catch.
        # Sanitising the input makes poisoning impossible from outside.
        if not np.all(np.isfinite(audio)):
            if not self._nan_warned:
                self._nan_warned = True
                logger.warning("non-finite INPUT samples sanitised (bad capture "
                               "device / WAV?); logged once")
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        ctx = FrameContext(
            audio=audio,
            sample_rate=self.cfg.sample_rate,
            reference=None if reference is None else np.asarray(reference, dtype=np.float32),
        )
        for stage in self.stages:
            ctx = stage(ctx)
        # Feedback hook: the Preprocessing stage uses the previous frame's
        # noise_class to pick its HP cutoff (default 25 Hz, or 80 Hz under
        # wind).  Pipeline writes back here so the next block sees the hint.
        self.preprocessing.set_prev_noise_class(ctx.noise_class)
        # OUTPUT-side guard: np.clip passes NaN through, so a numerical edge
        # case born INSIDE a stage would otherwise reach the speakers.
        if not np.all(np.isfinite(ctx.audio)):
            if not self._nan_warned:
                self._nan_warned = True
                logger.warning("pipeline emitted non-finite samples — sanitised "
                               "(this indicates a stage bug; logged once)")
            ctx.audio = np.nan_to_num(ctx.audio, nan=0.0, posinf=1.0, neginf=-1.0)
            # Self-healing: internally-born NaN can still poison persistent
            # state (no exception → containment can't fire, and the live path
            # never calls reset()).  Ten consecutive poisoned blocks (1 s) →
            # reset the DSP state once.  reset() costs ~0.01 ms (no model
            # reload), so the cure is a one-second hiccup instead of
            # permanent silence until app restart.
            self._nan_streak += 1
            if self._nan_streak >= 10:
                logger.error("pipeline output non-finite for %d consecutive "
                             "blocks — resetting DSP state to self-heal",
                             self._nan_streak)
                self.reset()
                self._nan_streak = 0
        else:
            self._nan_streak = 0
        return ctx

    @property
    def output_delay_samples(self) -> int:
        """Constant delay of the output timeline relative to the input, summed
        over the stages that declare one: the enhancement stage's deferred
        DFN3 tail (``dfn_tail_ms``, 960 samples by default) and the always-on
        multi-band FIR group delay (15 samples).  The live paths simply carry
        it (it is part of ``algorithmic_latency_ms``); the offline helper
        :meth:`process_signal` removes it."""
        return int(sum(int(getattr(st, "output_delay_samples", 0)) for st in self.stages))

    def process_signal(self, x: np.ndarray, block: Optional[int] = None,
                       flush: bool = True) -> np.ndarray:
        """Offline helper: stream a whole signal through in blocks.

        Default block size is 100 ms — the SAME operating point as the live
        paths (LiveStream, desktop app) and the benchmark, so offline results
        are representative of real-time behaviour.  (20 ms blocks give DFN3
        so little per-call context that SI-SDRi goes negative — never the
        default.)

        With ``flush=True`` (default) the result is SAMPLE-ALIGNED with ``x``
        and has exactly ``len(x)`` samples: the constant output delay
        (:attr:`output_delay_samples`) is removed, and the samples the stages
        were still holding at end-of-file are recovered by pushing zero blocks
        through the (still stateful) chain until the delay has been flushed.
        This is the same block-by-block processing the live paths run — no
        splice, no reset, the stages' state carries straight into the flush —
        so there is no discontinuity at the final block.  **Documented
        caveat:** the last ``dfn_tail_ms`` of the file are thereby enhanced
        against FABRICATED (zero) right context, which is exactly the
        condition the deferred tail exists to avoid everywhere else; those
        final 20 ms have the quality of the pre-tail design.  Live streams
        never do this, and their timing is unchanged by this helper.
        Signals shorter than one block, or shorter than the delay, are
        handled (the flush loop runs until the delay is covered).

        ``flush=False`` returns the raw delayed stream instead: the first
        ``output_delay_samples`` samples are the stages' zero history and the
        input's last ``output_delay_samples`` samples are not emitted.
        """
        block = block or int(round(self.cfg.sample_rate * 0.1))   # 100 ms
        x = np.asarray(x, dtype=np.float32)
        n = len(x)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        chunks = []
        for s in range(0, n, block):
            seg = x[s: s + block]
            ctx = self.process_block(seg)
            chunks.append(np.asarray(ctx.audio[: len(seg)], dtype=np.float32).copy())
        y = np.concatenate(chunks)[:n]
        d = self.output_delay_samples
        if not flush or d <= 0:
            return y
        zeros = np.zeros(block, dtype=np.float32)
        flushed = []
        got = 0
        while got < d:
            z = np.asarray(self.process_block(zeros).audio[:block], dtype=np.float32).copy()
            flushed.append(z)
            got += len(z)
        y = np.concatenate([y] + flushed)
        return y[d: d + n]

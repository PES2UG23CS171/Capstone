"""
SpeakerEmbedder — target-speaker recognition (ECAPA-TDNN, ONNX, INT8).

When a target speaker has been **enrolled** (10 s of clean speech recorded
once; their 192-dim x-vector stored on disk), this stage computes a fresh
embedding every ~100 ms from the running mic audio and writes the cosine
similarity to the enrolled vector into ``ctx.target_speaker_sim``.

Downstream stages — primarily the controller and the multi-band modulator —
use the similarity to decide whether the current voice is the *target* (sim
near 1.0 → preserve fully) or a *competing speaker* (sim near 0 → cut the
mid-band aggressively, the band where speech information lives).

Cost
----
* Model: ECAPA-TDNN small (SpeechBrain / 3D-Speaker), exported to ONNX, then
  ``onnxruntime.quantization.quantize_dynamic`` to INT8 weights.
* Parameters: ~6.2M | Memory: ~14 MB (INT8) | Inference: ~4 ms / 200 ms input.
* Latency: 0 added — runs in parallel with DFN3 on a separate ONNX session.

Fallback
--------
This stage is **passthrough** in three cases:

1. ``onnxruntime`` is not installed.
2. ``cfg.speaker_model_path`` is unset (no ECAPA ONNX available).
3. No enrolled vector is loaded (``cfg.speaker_enroll_path`` missing or empty).

In all passthrough cases ``ctx.target_speaker_sim`` is set to **1.0**, i.e.
"treat every voice as the target" — the feature is gracefully disabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage

try:
    import onnxruntime as ort  # type: ignore[import]
    _HAS_ORT = True
except Exception:  # pragma: no cover
    _HAS_ORT = False


def _resample_to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == 16000:
        return x.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr_in, 16000)
    return resample_poly(x, 16000 // g, sr_in // g).astype(np.float32)


class SpeakerEmbedder(Stage):
    name = "speaker_embedder"

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.window_samples_16k = int(16000 * cfg.speaker_window_ms / 1000.0)
        # Cadence tracked in milliseconds of input audio (not process() calls)
        # so it's robust to varying caller block sizes (20 ms live vs 100 ms bench).
        self.update_ms = float(cfg.speaker_update_ms)
        self.backend = "passthrough"

        self._session: Optional["ort.InferenceSession"] = None
        self._enrolled: Optional[np.ndarray] = None
        self._buf16k = np.zeros(0, dtype=np.float32)
        self._ms_since_update = 0.0
        self._last_sim = 1.0          # default: treat as target speaker

        if _HAS_ORT and cfg.speaker_model_path and Path(cfg.speaker_model_path).exists():
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1   # speaker model is cheap; 1 thread = lowest latency
                self._session = ort.InferenceSession(
                    cfg.speaker_model_path, sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self.backend = "ecapa_onnx"
            except Exception:  # pragma: no cover
                self._session = None

        if cfg.speaker_enroll_path and Path(cfg.speaker_enroll_path).exists():
            try:
                v = np.load(cfg.speaker_enroll_path).astype(np.float32).reshape(-1)
                n = np.linalg.norm(v)
                if n > 1e-6:
                    self._enrolled = v / n
            except Exception:  # pragma: no cover
                self._enrolled = None

    def reset(self) -> None:
        self._buf16k = np.zeros(0, dtype=np.float32)
        self._ms_since_update = 0.0
        self._last_sim = 1.0

    # ── enrollment API (used by the desktop app's "Calibrate" button) ────
    def enroll_from_audio(self, audio: np.ndarray, sr_in: int) -> bool:
        """Compute and store the enrolled x-vector for this audio (10 s+).

        Returns True on success, False if the model isn't available."""
        if self._session is None:
            return False
        x = _resample_to_16k(audio, sr_in)
        emb = self._infer(x)
        n = np.linalg.norm(emb)
        if n < 1e-6:
            return False
        self._enrolled = emb / n
        if self.cfg.speaker_enroll_path:
            try:
                np.save(self.cfg.speaker_enroll_path, self._enrolled)
            except Exception:  # pragma: no cover
                pass
        return True

    def _infer(self, audio_16k: np.ndarray) -> np.ndarray:
        """Run the ECAPA ONNX session and return a flat embedding vector."""
        x = audio_16k.reshape(1, -1).astype(np.float32)
        out = self._session.run(None, {self._session.get_inputs()[0].name: x})
        return np.asarray(out[0]).reshape(-1).astype(np.float32)

    def process(self, ctx: FrameContext) -> FrameContext:
        # Passthrough cases.
        if self._session is None or self._enrolled is None:
            ctx.target_speaker_sim = 1.0
            return ctx

        # Only buffer VOICED audio that's NOT echo-contaminated and NOT
        # during AEC calibration (where ctx.audio is the raw echo-laden mic
        # while echo_conf hasn't yet been computed).  Three gates:
        #   1. VAD-positive (real voice present)
        #   2. echo_conf below threshold (not leaked far-end speech)
        #   3. AEC not calibrating (raw mic flowing through)
        calibrating = float(ctx.meta.get("aec_calibrating", 0.0)) >= 0.5
        echo_clean = ctx.echo_conf < self.cfg.echo_conf_threshold
        voice_present = ctx.is_speech or ctx.vad_prob >= 0.4
        block_ms = 1000.0 * len(ctx.audio) / max(self.sr, 1)
        if voice_present and echo_clean and not calibrating:
            self._buf16k = np.concatenate([self._buf16k, _resample_to_16k(ctx.audio, self.sr)])
            if len(self._buf16k) > self.window_samples_16k * 2:
                self._buf16k = self._buf16k[-self.window_samples_16k:]
        self._ms_since_update += block_ms

        if (self._ms_since_update >= self.update_ms and
                len(self._buf16k) >= self.window_samples_16k):
            self._ms_since_update = 0.0
            try:
                emb = self._infer(self._buf16k[-self.window_samples_16k:])
                n = np.linalg.norm(emb)
                if n > 1e-6:
                    emb = emb / n
                    self._last_sim = float(np.dot(emb, self._enrolled))
            except Exception:  # pragma: no cover
                # On any inference error, fall back to passthrough rather than
                # propagating a bad similarity into the controller.
                self._last_sim = 1.0

        ctx.target_speaker_sim = float(self._last_sim)
        ctx.meta["spk_sim"] = ctx.target_speaker_sim
        return ctx

"""
Voice Activity Detection stage.

Primary: **Silero VAD** (silero-vad), an ONNX/torch model ~1 MB, ~1 ms/frame on
CPU, far more robust than energy or GMM VADs (handles music/noise without
false-triggering).  Runs at 16 kHz on 512-sample (32 ms) windows; we resample
the 48 kHz frame down for the VAD only.

Fallback: an adaptive energy/SNR gate if silero-vad is unavailable, so the
pipeline always has *some* speech probability.

Placement: VAD sits *after* preprocessing/AEC and *before* the controller, so
its confidence can steer suppression aggressiveness.  Output:
  * ``ctx.vad_prob``  — raw P(speech) ∈ [0, 1]  (confidence)
  * ``ctx.is_speech`` — thresholded + hangover-smoothed boolean

Confidence is used downstream:  high confidence → gentle suppression (protect
speech);  low confidence → aggressive suppression (kill noise in gaps).
"""

from __future__ import annotations

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage

try:
    import torch
    from silero_vad import load_silero_vad
    _HAS_SILERO = True
except Exception:  # pragma: no cover
    _HAS_SILERO = False


def _resample_poly(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


class VAD(Stage):
    name = "vad"

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.vsr = cfg.vad_sample_rate
        self.win = cfg.vad_window
        self.threshold = cfg.vad_speech_threshold
        # Hangover length in SAMPLES at the pipeline sample rate.  We decrement
        # by the actual block length each call, so the realised hangover is in
        # milliseconds regardless of the caller's block size.  (V1 stored
        # hop-count and decremented once per call, which doubled the realised
        # hangover whenever the block was longer than one hop — the default.)
        self._hangover_samples_total = int(cfg.sample_rate * cfg.vad_hangover_ms / 1000.0)
        self._hang_samples = 0
        self._buf = np.zeros(0, dtype=np.float32)   # accumulates 16 kHz samples
        self._last_prob = 0.0
        self.backend = "none"

        if _HAS_SILERO:
            torch.set_num_threads(1)                # VAD is tiny; 1 thread is lowest-latency
            self._model = load_silero_vad(onnx=True)
            self.backend = "silero"
        else:
            self._model = None
            self.backend = "energy"
            self._noise = 1e-4

    def reset(self) -> None:
        self._hang_samples = 0
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0
        if self.backend == "silero" and hasattr(self._model, "reset_states"):
            self._model.reset_states()

    # ── backends ─────────────────────────────────────────────────────────
    def _silero_prob(self, frame48: np.ndarray) -> float:
        # Accumulate resampled audio and run the model on full 512-sample windows.
        self._buf = np.concatenate([self._buf, _resample_poly(frame48, self.sr, self.vsr)])
        prob = self._last_prob
        while len(self._buf) >= self.win:
            chunk = self._buf[: self.win]
            self._buf = self._buf[self.win :]
            with torch.no_grad():
                prob = float(self._model(torch.from_numpy(chunk).float(), self.vsr).item())
        self._last_prob = prob
        return prob

    def _energy_prob(self, frame48: np.ndarray) -> float:
        p = float(np.mean(frame48 * frame48) + 1e-12)
        self._noise = min(self._noise * 1.02, max(self._noise, p)) if p > self._noise else \
            0.95 * self._noise + 0.05 * p
        ratio = 10.0 * np.log10(p / max(self._noise, 1e-10))
        return float(np.clip((ratio - 3.0) / 12.0, 0.0, 1.0))

    def process(self, ctx: FrameContext) -> FrameContext:
        prob = self._silero_prob(ctx.audio) if self.backend == "silero" else self._energy_prob(ctx.audio)
        ctx.vad_prob = prob
        block_samples = len(ctx.audio)

        if prob >= self.threshold:
            self._hang_samples = self._hangover_samples_total
            ctx.is_speech = True
        else:
            if self._hang_samples > 0:
                # Decrement by ACTUAL block length so the realised hangover is
                # always vad_hangover_ms regardless of the caller's block size.
                self._hang_samples = max(0, self._hang_samples - block_samples)
                ctx.is_speech = True       # hangover: don't chop word tails
            else:
                ctx.is_speech = False
        ctx.meta["vad_backend"] = 1.0 if self.backend == "silero" else 0.0
        return ctx

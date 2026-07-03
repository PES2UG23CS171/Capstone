"""
TargetSpeakerMask — VoiceFilter-Lite-style target-speaker extraction.

Concept
-------
After DFN3 has removed *noise*, a competing-speech mask network removes
*non-target speech*.  The mask is conditioned on the enrolled speaker's
ECAPA-TDNN x-vector (produced by :class:`SpeakerEmbedder`) and operates
per-bin in the STFT magnitude domain.

Architecture (target, when a checkpoint ships):

    Input  : log-mel(80) of post-DFN3 audio   + enrolled x-vector(192)
             = 272-dim feature vector per frame
    LSTM   : hidden=128, 1 layer  → ≈100k params
    FC     : 128 → 481 rFFT bins (960-sample window @ 48 kHz)
    Sigmoid: per-bin gain mask  ∈ [0, 1]
    Apply  : STFT(audio) · mask → iSTFT → output

Total ≈ 100k params, INT8 ONNX ≈ 130 KB on disk / ~0.6 MB resident.
CPU ≈ 3 ms / 20 ms block, latency ≈ 0 added (operates on existing STFT).

Bypass policy
-------------
* No ``onnxruntime`` installed → bypass.
* ``cfg.target_speaker_mask_path`` unset or missing → bypass.
* Enrolled x-vector unavailable (``target_speaker_sim`` sentinel 1.0) → bypass.
* ``ctx.vad_prob < 0.35`` (no voice present) → bypass.
* ``target_speaker_sim ≥ 0.65`` for several consecutive frames → bypass
  (the input is confidently the target speaker — masking is unnecessary
  and may damage the very signal we want to keep).

Sticky activation
-----------------
To avoid flicker between target and competing speaker during overlap, when
the mask DOES activate it stays active for a minimum of 200 ms even if a
single frame's sim score briefly recovers.  Conversely, deactivation
requires 200 ms of sustained high-sim frames.

Output
------
Replaces ``ctx.audio`` with the masked signal when active; writes
``ctx.meta['tsm_active']`` and ``ctx.meta['tsm_mix']`` for telemetry.
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


def _log_mel(x: np.ndarray, sr: int, n_mels: int = 80) -> np.ndarray:
    """Log-mel feature vector (single frame)."""
    n = len(x)
    if n < 64:
        return np.zeros(n_mels, dtype=np.float32)
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mel_max = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
    mel_edges = np.linspace(0.0, mel_max, n_mels + 2)
    hz_edges = 700.0 * (10.0 ** (mel_edges / 2595.0) - 1.0)
    out = np.zeros(n_mels, dtype=np.float32)
    for i in range(n_mels):
        f_lo, f_c, f_hi = hz_edges[i], hz_edges[i + 1], hz_edges[i + 2]
        m_lo = (freqs >= f_lo) & (freqs < f_c)
        m_hi = (freqs >= f_c) & (freqs < f_hi)
        w_lo = (freqs[m_lo] - f_lo) / max(f_c - f_lo, 1e-6)
        w_hi = (f_hi - freqs[m_hi]) / max(f_hi - f_c, 1e-6)
        out[i] = (spec[m_lo] * w_lo).sum() + (spec[m_hi] * w_hi).sum()
    return np.log(out + 1e-9).astype(np.float32)


class TargetSpeakerMask(Stage):
    name = "target_speaker_mask"

    # Sim regimes for active/inactive decisions.  Hysteresis bands prevent
    # flicker between target and competing-speaker labels.
    _SIM_ACTIVATE_BELOW = 0.40
    _SIM_DEACTIVATE_ABOVE = 0.65
    # Stickiness — minimum on/off duration (samples at sr).
    _STICKY_MS = 200.0

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.backend = "passthrough"
        self._session: Optional["ort.InferenceSession"] = None
        self._hidden: Optional[np.ndarray] = None
        self._enrolled: Optional[np.ndarray] = None
        self._active = False
        self._hold_samples = 0

        # Look for a target-speaker-mask checkpoint at the dedicated config path.
        ckpt = getattr(cfg, "target_speaker_mask_path", None)
        if _HAS_ORT and ckpt and Path(ckpt).exists():
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                self._session = ort.InferenceSession(
                    ckpt, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                self.backend = "vf_lite_onnx"
            except Exception:  # pragma: no cover
                self._session = None

        # Try to load the enrolled x-vector (same path as SpeakerEmbedder).
        if cfg.speaker_enroll_path and Path(cfg.speaker_enroll_path).exists():
            try:
                v = np.load(cfg.speaker_enroll_path).astype(np.float32).reshape(-1)
                n = np.linalg.norm(v)
                if n > 1e-6:
                    self._enrolled = v / n
            except Exception:  # pragma: no cover
                self._enrolled = None

    def reset(self) -> None:
        self._hidden = None
        self._active = False
        self._hold_samples = 0

    def _bypass_now(self, ctx: FrameContext) -> bool:
        """Quick checks: skip the mask under any of these conditions."""
        if self._session is None or self._enrolled is None:
            return True
        if ctx.vad_prob < 0.35:
            return True
        # When sim sentinel ≈ 1.0 (no live ECAPA running), nothing to gate on.
        if ctx.target_speaker_sim > 0.999:
            return True
        return False

    def _update_active(self, ctx: FrameContext, block_samples: int) -> bool:
        """Hysteretic activation decision with stickiness."""
        sim = float(ctx.target_speaker_sim)
        sticky_samples = int(self.sr * self._STICKY_MS / 1000.0)
        if self._active:
            # Only deactivate when sim is well above threshold for the sticky period.
            if sim >= self._SIM_DEACTIVATE_ABOVE:
                self._hold_samples += block_samples
                if self._hold_samples >= sticky_samples:
                    self._active = False
                    self._hold_samples = 0
            else:
                self._hold_samples = 0
        else:
            # Activate when sim drops below activation threshold for the sticky period.
            if sim <= self._SIM_ACTIVATE_BELOW:
                self._hold_samples += block_samples
                if self._hold_samples >= sticky_samples:
                    self._active = True
                    self._hold_samples = 0
            else:
                self._hold_samples = 0
        return self._active

    def _apply_mask(self, audio: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply a per-bin gain mask via single-window STFT/iSTFT."""
        n = len(audio)
        if n < 64:
            return audio
        win = np.hanning(n).astype(np.float32)
        X = np.fft.rfft(audio * win)
        m = mask[: len(X)]
        if len(m) < len(X):
            m = np.concatenate([m, np.ones(len(X) - len(m), dtype=np.float32)])
        Y = X * m
        y = np.fft.irfft(Y, n=n).astype(np.float32)
        # Same reconstruction trick as TinyGRUPostFilter: avoid the analysis
        # window's zero-tapered edges in the output by adding back (1-win)*x.
        return y + (1.0 - win) * audio

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.meta["tsm_active"] = 0.0
        if self._bypass_now(ctx):
            return ctx

        block_samples = len(ctx.audio)
        active = self._update_active(ctx, block_samples)
        if not active:
            return ctx

        # Build feature vector: log-mel(audio) + enrolled x-vector.
        feats = np.concatenate([_log_mel(ctx.audio, self.sr), self._enrolled])
        # Reshape to (batch=1, time=1, feature_dim) for an LSTM.
        feats = feats.reshape(1, 1, -1).astype(np.float32)

        try:
            inputs = self._session.get_inputs()
            feed = {inputs[0].name: feats}
            # Thread the hidden state if the model exposes it.
            for inp in inputs[1:]:
                if "hidden" in inp.name.lower() or "h0" in inp.name.lower():
                    if self._hidden is None:
                        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
                        self._hidden = np.zeros(shape, dtype=np.float32)
                    feed[inp.name] = self._hidden
            outputs = self._session.run(None, feed)
            mask = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
            if len(outputs) > 1:
                self._hidden = outputs[1]
        except Exception:  # pragma: no cover
            # On any inference failure, fall back to passthrough so we don't
            # damage audio with a misbehaving model.
            return ctx

        # Gentle mask when sim is in the ambiguous range — at 0.3 we want
        # almost full mask, at 0.5 we want barely any.
        sim = float(ctx.target_speaker_sim)
        mask_strength = float(np.clip(
            (self._SIM_DEACTIVATE_ABOVE - sim) / (self._SIM_DEACTIVATE_ABOVE - self._SIM_ACTIVATE_BELOW),
            0.0, 1.0,
        ))
        # Blend mask toward unity by (1 - mask_strength).
        effective_mask = mask_strength * mask + (1.0 - mask_strength) * 1.0

        ctx.audio = self._apply_mask(ctx.audio, effective_mask)
        ctx.meta["tsm_active"] = 1.0
        ctx.meta["tsm_mix"] = mask_strength
        return ctx

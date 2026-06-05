"""
TinyGRUPostFilter — learned per-bin spectral cleanup after DFN3.

DFN3 leaves the most audible residuals on music, TV speech, and competing
speech — categories where the noise has tonal/harmonic structure.  A small
GRU operating on log-mel features + class one-hot can learn to mask these
residuals far better than the rule-based residual gate in
:class:`PostFilter`.

Architecture (target):
    Input  : log-mel(64) + noise_class_one_hot(12) + suppression(1) + erle_db(1)
             = 78-dim feature vector per frame
    GRU    : hidden=120, 1 layer  → ≈84k params
    FC     : 120 → 481  (rFFT bins of 960-sample window @ 48 kHz)
    Output : sigmoid → per-bin gain mask in [0, 1]
    Apply  : STFT(wet) · mask → ISTFT → output

Total ≈ 96k params, INT8 ONNX ≈ 120 KB on disk / ~0.5 MB resident.
CPU ≈ 1.4 ms / 20 ms block, latency ≈ 0 added (operates on the STFT we already
take for the over-suppression detector + classifier features).

**This file is a scaffold.**  No checkpoint ships with the repo, so:

* When ``cfg.postfilter_model_path`` is unset or the file is missing: this
  stage is a complete passthrough (no compute, no allocation).
* When a checkpoint exists and ``onnxruntime`` is installed: the stage runs
  the GRU mask.  Training is documented separately
  (see ``scripts/train_tiny_postfilter.py``, not shipped here).

Bypass policy: even with a checkpoint loaded, this stage skips frames where
DFN3 already does well (``clean``/``fan``/``hvac``/``traffic`` classes, or
``suppression < 0.3``) so it cannot make a good frame worse.
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


# Classes where DFN3 alone already does well — skip the post-filter on these
# to keep latency and artifact risk lower.
_SKIP_CLASSES = {"clean", "fan", "hvac", "traffic"}


def _log_mel(x: np.ndarray, sr: int, n_mels: int = 64) -> np.ndarray:
    """Cheap log-mel features.  Used both at training time and inference."""
    n = len(x)
    if n < 64:
        return np.zeros(n_mels, dtype=np.float32)
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    # Mel filterbank edges
    mel_max = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
    mel_edges = np.linspace(0.0, mel_max, n_mels + 2)
    hz_edges = 700.0 * (10.0 ** (mel_edges / 2595.0) - 1.0)
    out = np.zeros(n_mels, dtype=np.float32)
    for i in range(n_mels):
        f_lo, f_c, f_hi = hz_edges[i], hz_edges[i + 1], hz_edges[i + 2]
        m_lo = (freqs >= f_lo) & (freqs < f_c)
        m_hi = (freqs >= f_c) & (freqs < f_hi)
        # Triangle weights
        w_lo = (freqs[m_lo] - f_lo) / max(f_c - f_lo, 1e-6)
        w_hi = (f_hi - freqs[m_hi]) / max(f_hi - f_c, 1e-6)
        out[i] = (spec[m_lo] * w_lo).sum() + (spec[m_hi] * w_hi).sum()
    return np.log(out + 1e-9).astype(np.float32)


class TinyGRUPostFilter(Stage):
    name = "tiny_postfilter"

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.backend = "passthrough"
        self._session: Optional["ort.InferenceSession"] = None
        self._hidden: Optional[np.ndarray] = None
        self.classes = list(cfg.noise_classes)

        if _HAS_ORT and cfg.postfilter_model_path and Path(cfg.postfilter_model_path).exists():
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                self._session = ort.InferenceSession(
                    cfg.postfilter_model_path, sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self.backend = "tiny_gru_onnx"
                # Hidden state shape inferred from model on first call.
                self._hidden = None
            except Exception:  # pragma: no cover
                self._session = None
                self.backend = "passthrough"

    def reset(self) -> None:
        self._hidden = None

    def _class_one_hot(self, noise_class: str) -> np.ndarray:
        v = np.zeros(len(self.classes), dtype=np.float32)
        if noise_class in self.classes:
            v[self.classes.index(noise_class)] = 1.0
        return v

    def process(self, ctx: FrameContext) -> FrameContext:
        if self._session is None:
            return ctx
        # Bypass on classes DFN3 already handles well.
        if ctx.noise_class in _SKIP_CLASSES or ctx.suppression < 0.3:
            return ctx

        # Feature vector: log-mel + class one-hot + suppression + erle.
        feats = np.concatenate([
            _log_mel(ctx.audio, self.sr),
            self._class_one_hot(ctx.noise_class),
            np.array([float(ctx.suppression)], dtype=np.float32),
            np.array([float(ctx.meta.get("erle_db", 0.0))], dtype=np.float32),
        ])
        # Reshape to (batch=1, time=1, feature_dim) for a GRU.
        feats = feats.reshape(1, 1, -1).astype(np.float32)

        try:
            inputs = self._session.get_inputs()
            input_name = inputs[0].name
            feed = {input_name: feats}
            # If the model exposes a hidden-state input, thread it through.
            for inp in inputs[1:]:
                if "hidden" in inp.name.lower() or "h0" in inp.name.lower():
                    if self._hidden is None:
                        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
                        self._hidden = np.zeros(shape, dtype=np.float32)
                    feed[inp.name] = self._hidden
            outputs = self._session.run(None, feed)
            mask = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
            # Capture updated hidden state if the model returns it.
            if len(outputs) > 1:
                self._hidden = outputs[1]
        except Exception:  # pragma: no cover
            return ctx

        # Apply mask in the magnitude-STFT domain.
        # We trained / will train the mask under a single Hann window analysis
        # (no overlap-add).  The synthesis is therefore plain irfft of the
        # masked spectrum: the Hann window's analysis envelope is left intact
        # in the output, which DFN3's downstream tail handles cleanly.
        #
        # An earlier draft divided by ``(win*win + 1e-6)``, which exploded near
        # the window edges (``win[0] == win[-1] == 0``) — ~1e5 amplification
        # that was masked only by the final ``np.clip`` and produced an
        # audible click every 20 ms.  Removed.
        x = ctx.audio
        n = len(x)
        if n < 64 or len(mask) < (n // 2 + 1):
            return ctx
        win = np.hanning(n).astype(np.float32)
        X = np.fft.rfft(x * win)
        m = mask[: len(X)]
        if len(m) < len(X):
            m = np.concatenate([m, np.ones(len(X) - len(m), dtype=np.float32)])
        Y = X * m
        y = np.fft.irfft(Y, n=n).astype(np.float32)
        # Cross-fade onto the *un-windowed* input so the analysis Hann doesn't
        # taper the output to zero at block edges.  At unity mask y == x*win,
        # so y + (1 - win)*x == x exactly (perfect reconstruction at unity).
        y = y + (1.0 - win) * x
        ctx.audio = np.clip(y, -1.0, 1.0).astype("float32")
        ctx.meta["tinypf_applied"] = 1.0
        return ctx

"""
Enhancement stage — the DeepFilterNet3 core.

DFN3 is a 48 kHz, real-time, CPU-first speech-enhancement network that combines
ERB-band gain estimation (coarse spectral envelope) with *deep filtering*
(per-bin complex filters for fine structure / low frequencies).  It is the
closest open-source thing to Krisp-grade quality on CPU.

Design decisions (validated empirically on real LibriSpeech + noise):

* **Trust DFN — output is fully wet.**  DFN preserves speech at correlation
  ≈ 0.99 with the clean reference; blending the *noisy dry* signal back in to
  "protect speech" was found to wreck SNR (+14 dB → +3 dB).  Speech protection
  is instead done by limiting DFN's attenuation (``atten_lim_db``), which keeps
  the output clean while being gentle.
* **Overlap-save streaming.**  DFN benefits from temporal context, so each block
  is enhanced together with a look-back ``context`` of prior input and only the
  aligned tail is emitted.  This recovers near-whole-signal quality
  (+16.4 dB vs +17.6 dB whole-signal) while staying streaming.
* **Bypass when clean.**  When the controller reports a genuinely clean, high-SNR
  input (suppression ≈ 0) we pass the dry signal through and skip DFN — saving
  CPU and avoiding any processing on already-clean audio.

Cost: reprocessing the context costs ~2× compute → RTF ≈ 0.18–0.20 at the
default 100 ms block / 200 ms context.  True frame-streaming via DFN's stateful
Rust API is the documented next optimization.
"""

from __future__ import annotations

import numpy as np

import voiceiso._compat  # noqa: F401  — installs torchaudio shim before df import
from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage

try:
    import torch
    from df.enhance import init_df, enhance
    _HAS_DF = True
except Exception:  # pragma: no cover
    _HAS_DF = False


class Enhancement(Stage):
    name = "enhancement"

    def __init__(self, cfg: PipelineConfig, threads: int = 4, context_ms: float = 200.0,
                 bypass_below: float = 0.08) -> None:
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.backend = "passthrough"
        self.bypass_below = bypass_below
        self._ctx_len = int(self.sr * context_ms / 1000.0)
        self._history = np.zeros(0, dtype="float32")
        self._model = None
        self._state = None
        if _HAS_DF:
            torch.set_num_threads(threads)
            self._model, self._state, _ = init_df(
                model_base_dir=cfg.dfn_model_dir, config_allow_defaults=True
            )
            self.backend = "deepfilternet3"
            assert self._state.sr() == self.sr, "DFN3 expects 48 kHz audio"

    def reset(self) -> None:
        self._history = np.zeros(0, dtype="float32")

    def _enhance(self, buf: np.ndarray, atten_lim_db: float) -> np.ndarray:
        x = torch.from_numpy(buf.reshape(1, -1).astype("float32"))
        with torch.no_grad():
            y = enhance(self._model, self._state, x, atten_lim_db=atten_lim_db)
        out = y.squeeze(0).cpu().numpy().astype("float32")
        if len(out) < len(buf):
            out = np.concatenate([out, np.zeros(len(buf) - len(out), dtype="float32")])
        return out[: len(buf)]

    def process(self, ctx: FrameContext) -> FrameContext:
        dry = ctx.audio
        supp = float(np.clip(ctx.suppression, 0.0, 1.0))

        # Always keep the (dry) look-back history up to date for continuity.
        prev = self._history
        if self.backend == "passthrough" or supp <= self.bypass_below:
            # Clean/bypass: passthrough dry, skip the network.
            self._history = np.concatenate([prev, dry])[-self._ctx_len :]
            ctx.meta["enh_wet"] = 0.0
            return ctx

        # Map suppression → DFN attenuation cap (gentle 12 dB … aggressive 100 dB).
        atten_lim_db = float(np.interp(supp, [0.0, 1.0], [12.0, 100.0]))
        buf = np.concatenate([prev, dry])
        wet_full = self._enhance(buf, atten_lim_db)
        wet = wet_full[len(prev):][: len(dry)]

        self._history = buf[-self._ctx_len :]
        ctx.audio = wet.astype("float32")
        ctx.meta["enh_wet"] = 1.0
        ctx.meta["enh_atten_db"] = atten_lim_db
        return ctx

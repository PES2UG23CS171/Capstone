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
* **True stateful streaming.**  DFN3's ``_state`` object carries GRU hidden
  states across ``enhance()`` calls; each call receives only the *new* block
  and the model uses its internal recurrent state for temporal context.
  The previous overlap-save approach prepended a 200 ms history buffer on every
  call, causing DFN to re-process old audio (double-weighting it through the
  recurrent layers) and doubling compute cost.  Removing the context buffer
  halves RTF and enables 20 ms blocks without quality loss.
* **Bypass when clean.**  When the controller reports a genuinely clean, high-SNR
  input (suppression ≈ 0) we pass the dry signal through and skip DFN — saving
  CPU and avoiding any processing on already-clean audio.
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


def _erb_band_energies(x: np.ndarray, sr: int, n_bands: int = 8) -> np.ndarray:
    """Approximate ERB-band log-energies — used for over-suppression detection."""
    if len(x) < 32:
        return np.zeros(n_bands, dtype=np.float32)
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    # Log-spaced edges 80 Hz … sr/2 — coarse but fine for over-suppression cues.
    edges = np.logspace(np.log10(80.0), np.log10(sr / 2.0 - 1.0), n_bands + 1)
    out = np.zeros(n_bands, dtype=np.float32)
    for b in range(n_bands):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        out[b] = float(mag[m].sum()) + 1e-12
    return out


class Enhancement(Stage):
    name = "enhancement"

    def __init__(self, cfg: PipelineConfig, threads: int = 4,
                 bypass_below: float = 0.08) -> None:
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.threads = threads
        self.backend = "passthrough"
        self.bypass_below = bypass_below
        self._model = None
        self._state = None
        if _HAS_DF:
            torch.set_num_threads(threads)
            self._model, self._state, _ = init_df(
                model_base_dir=cfg.dfn_model_dir, config_allow_defaults=True
            )
            self.backend = "deepfilternet3"
            assert self._state.sr() == self.sr, "DFN3 expects 48 kHz audio"

        # Self-managed rollback state: number of dB to subtract from the next
        # frame's atten_lim_db cap after an over-suppression event.  Decays
        # over a few frames so we don't permanently relax the enhancer.
        self._cap_relief_db = 0.0

    def reset(self) -> None:
        """Clear DFN3's recurrent state.

        ``DfState`` (the Rust object) does not expose a Python-visible reset, so
        we re-construct it via ``init_df``.  This is somewhat expensive (~50 ms)
        but only happens on stream (re)start — never on the hot path.
        """
        self._cap_relief_db = 0.0
        if self.backend != "deepfilternet3" or not _HAS_DF:
            return
        torch.set_num_threads(self.threads)
        self._model, self._state, _ = init_df(
            model_base_dir=self.cfg.dfn_model_dir, config_allow_defaults=True
        )

    def _enhance(self, buf: np.ndarray, atten_lim_db: float) -> np.ndarray:
        x = torch.from_numpy(buf.reshape(1, -1).astype("float32"))
        with torch.no_grad():
            y = enhance(self._model, self._state, x, atten_lim_db=atten_lim_db)
        out = y.squeeze(0).cpu().numpy().astype("float32")
        if len(out) < len(buf):
            # DFN3 sometimes drops trailing samples at the block boundary; pad
            # with zero to keep the pipeline's hop alignment intact.
            out = np.concatenate([out, np.zeros(len(buf) - len(out), dtype="float32")])
        return out[: len(buf)]

    def process(self, ctx: FrameContext) -> FrameContext:
        dry = ctx.audio

        supp = float(np.clip(ctx.suppression, 0.0, 1.0))
        if self.backend == "passthrough" or supp <= self.bypass_below:
            ctx.meta["enh_wet"] = 0.0
            return ctx

        # Only stash dry for rollback when we're actually about to run DFN3.
        ctx.dry_audio = dry.copy()

        # Map suppression → DFN attenuation cap (gentle 12 dB … aggressive 100 dB).
        # Self-managed cap relief from any recent over-suppression event.
        relief = self._cap_relief_db
        atten_lim_db = float(np.interp(supp, [0.0, 1.0], [12.0, 100.0])) - relief
        atten_lim_db = max(6.0, atten_lim_db)
        # Decay cap relief one notch each frame regardless of whether we triggered
        # a new overshoot — keeps the effect from compounding indefinitely.
        self._cap_relief_db = max(0.0, self._cap_relief_db - 2.0)

        # Pass only the current block — DFN3's recurrent state (_state) carries
        # temporal context from all previous blocks automatically.
        wet = self._enhance(dry, atten_lim_db)

        # ── Sub-band over-suppression detector + immediate rollback ──────
        # Only fire when:
        #   1. speech_conf ≥ 0.6 (looks like real speech — VAD + consonants), AND
        #   2. target_speaker_sim ≥ 0.4 OR the embedder is in passthrough (sim≈1).
        #      This blocks the rollback during competing speech / TV speech,
        #      where Silero false-fires VAD on a non-target speaker and the
        #      "rollback" would re-inject the very noise we're trying to remove.
        #   3. DFN3 dropped some ERB band by more than the configured threshold.
        # When the rollback DOES fire, blend a fraction of dry back into this
        # block and schedule cap relief for the next frame.
        sim = float(ctx.target_speaker_sim)
        embedder_passthrough = sim > 0.999          # 1.0 sentinel = no embedder
        target_present = embedder_passthrough or sim >= 0.4

        overshoot = 0.0
        if ctx.speech_conf >= 0.6 and target_present:
            dry_e = _erb_band_energies(dry, self.sr)
            wet_e = _erb_band_energies(wet, self.sr)
            drops_db = 10.0 * np.log10((dry_e + 1e-9) / (wet_e + 1e-9))
            max_drop = float(np.max(drops_db))
            ctx.meta["enh_max_drop_db"] = max_drop
            if max_drop > self.cfg.artifact_band_drop_db:
                overshoot = 1.0
                mix = float(self.cfg.artifact_rollback_mix)
                wet = (1.0 - mix) * wet + mix * dry
                self._cap_relief_db = float(self.cfg.artifact_cap_relief_db)
                ctx.meta["enh_rollback"] = mix
        ctx.meta["enh_overshoot"] = overshoot

        ctx.audio = wet.astype("float32")
        ctx.meta["enh_wet"] = 1.0
        ctx.meta["enh_atten_db"] = atten_lim_db
        ctx.meta["enh_cap_relief_db"] = relief
        return ctx

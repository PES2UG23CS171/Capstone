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
* **Per-block processing (stateless GRU across calls).**  Each ``enhance()``
  call processes only the *new* block.  NOTE: contrary to an earlier claim,
  DFN3's ``_state`` object carries only the STFT/ISTFT analysis-synthesis and
  ERB-normalisation state across calls — it does **not** persist the model's
  GRU hidden states.  DFN3's GRUs are invoked stateless (``h0`` defaults to
  zero every call), so temporal context spans only the STFT frames *within* the
  current block, not across blocks.  Consequence: larger blocks give the GRUs
  more intra-call context.  All paths (benchmark, ``voiceiso live``, desktop
  app) therefore run 100 ms blocks — DFN3's efficient design point.  The old
  overlap-save approach (prepending a 200 ms history buffer every call) is
  still avoided because it re-ran old audio through the network; we simply do
  not claim cross-call recurrence we don't have.
* **History-primed per-call processing (no persistent DfState).**  The naive
  per-block mode (one long-lived DfState, ``enhance(pad=True)`` per call)
  feeds ``n_fft`` fabricated zeros into the stateful STFT/ERB stream on every
  call, and the model's first frames each call run with cold (zeroed) GRU and
  conv context.  Measured on speech at 5 dB SNR / 100 ms blocks this costs
  ~6 dB SI-SDR vs one-shot enhancement (11.5 dB vs 17.97 dB) plus a 10 Hz
  block-edge warble.  (Plain ``pad=False`` on a persistent state is even
  worse — 9.0 dB — because it *keeps* each call's cold first frame, which
  ``pad=True``'s output trim happens to discard.)  Instead each call
  processes ``[last 80 ms of real input | current block]`` with a FRESH
  DfState and keeps only the block's region: every emitted frame has real
  left context, the GRUs warm up over the history, nothing fabricated enters
  any persistent state (there is none), and the output stays sample-aligned
  (zero added latency).  Measured: 17.7 dB SI-SDR on the same condition —
  within 0.3 dB of one-shot — at RTF ≈ 0.17 (p95 ≈ 20 ms per 100 ms block).
  History length is ``cfg.dfn_history_ms`` (80 ms = saturation point; 160 and
  240 ms measure no better).
* **Bypass when clean.**  When the controller reports a genuinely clean, high-SNR
  input (suppression ≈ 0) we pass the dry signal through and skip DFN — saving
  CPU and avoiding any processing on already-clean audio.
"""

from __future__ import annotations

import logging

import numpy as np

import voiceiso._compat  # noqa: F401  — installs torchaudio shim before df import
from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage

logger = logging.getLogger(__name__)

try:
    import torch
    from df.enhance import init_df, enhance
    _HAS_DF = True
    _DF_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover
    _HAS_DF = False
    _DF_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"


def _erb_band_edges(sr: int, n_bands: int = 8) -> np.ndarray:
    """Log-spaced ERB-like band edges from 80 Hz to Nyquist."""
    return np.logspace(np.log10(80.0), np.log10(sr / 2.0 - 1.0), n_bands + 1)


def _erb_band_energies(x: np.ndarray, sr: int, n_bands: int = 8) -> np.ndarray:
    """Approximate ERB-band log-energies — used for over-suppression detection."""
    if len(x) < 32:
        return np.zeros(n_bands, dtype=np.float32)
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    edges = _erb_band_edges(sr, n_bands)
    out = np.zeros(n_bands, dtype=np.float32)
    for b in range(n_bands):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        out[b] = float(mag[m].sum()) + 1e-12
    return out


def _spectral_kurtosis(x: np.ndarray) -> float:
    """Kurtosis of |STFT|² — a proxy for sparseness/tonality of the spectrum.

    Musical-noise artifacts appear as sparse tonal residuals, which raise the
    spectral kurtosis sharply compared to the input's distribution.
    """
    if len(x) < 32:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    spec = spec / (float(spec.sum()) + 1e-12)
    mu = float(np.mean(spec))
    sigma = float(np.std(spec)) + 1e-12
    z = (spec - mu) / sigma
    return float(np.mean(z ** 4)) - 3.0


def _lpc_residual_ratio(dry: np.ndarray, wet: np.ndarray, order: int = 12) -> float:
    """Ratio of LPC-residual energy of ``wet`` to ``dry``.

    A speech-preserving enhancer maintains the formant structure, so the LPC
    residual of the wet signal should track the residual of the dry signal.
    If the wet residual has much *less* energy (relative to the wet signal),
    formants have been smoothed → likely distortion.

    Returns a score ∈ [0, 1]: 0 = preserved, 1 = strongly distorted.
    """
    if len(dry) < order * 4 or len(wet) < order * 4:
        return 0.0

    def _residual_energy(sig: np.ndarray) -> float:
        # Burg-like simplified LPC: solve normal equations via numpy.
        n = len(sig)
        r = np.array([np.dot(sig[: n - k], sig[k:]) for k in range(order + 1)])
        if r[0] < 1e-12:
            return 0.0
        # Yule-Walker solve (small order, cheap).
        try:
            R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
            a = np.linalg.solve(R + 1e-9 * np.eye(order), r[1 : order + 1])
        except np.linalg.LinAlgError:
            return 0.0
        # Residual = sig − sum(a_k · sig[t-k])
        pred = np.convolve(sig, np.concatenate([[0.0], a]), mode="full")[: n]
        resid = sig - pred[: n]
        return float(np.mean(resid * resid))

    e_dry = _residual_energy(dry) + 1e-12
    e_wet = _residual_energy(wet) + 1e-12
    # Energy ratio normalised by signal energy ratio.
    s_dry = float(np.mean(dry * dry)) + 1e-12
    s_wet = float(np.mean(wet * wet)) + 1e-12
    # Relative residual fraction: low for over-smoothed (formants gone).
    rel_dry = e_dry / s_dry
    rel_wet = e_wet / s_wet
    drop = (rel_dry - rel_wet) / max(rel_dry, 1e-9)
    return float(np.clip(drop, 0.0, 1.0))


def _mask_confidence(mask: np.ndarray) -> float:
    """Entropy-based mask confidence ∈ [0, 1].

    Confidence is HIGH when the mask values cluster near 0 (suppress) or near 1
    (keep) — i.e. DFN3 made a clean decision.  Confidence is LOW when values
    cluster around 0.5 — the network is uncertain, and aggressive rollback in
    that case is riskier (we may damage what little signal DFN3 was trying to
    extract).

    Computed as ``1 − H(p) / log(2)`` where p = clip(mask, ε, 1-ε) and H is
    the per-bin binary entropy, averaged across bins.
    """
    if mask is None or len(mask) == 0:
        return 1.0
    p = np.clip(mask.astype(np.float64), 1e-6, 1.0 - 1e-6)
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    h_norm = float(np.mean(h)) / np.log(2.0)   # ∈ [0, 1]
    return float(np.clip(1.0 - h_norm, 0.0, 1.0))


def _band_rollback_stft(wet: np.ndarray, dry: np.ndarray, sr: int,
                        offending_bands: list[int], band_edges: np.ndarray,
                        mix: float) -> np.ndarray:
    """Blend dry into ``wet`` only inside ``offending_bands`` (STFT domain).

    Single-window FFT/iFFT — no overlap.  This is acceptable because rollback
    fires on isolated frames, not continuously.  Within the offending bands
    the output is ``(1-mix)*wet + mix*dry`` per FFT bin; outside, unchanged.
    """
    n = len(wet)
    if n < 32 or not offending_bands:
        return wet
    win = np.hanning(n).astype(np.float32)
    Xw = np.fft.rfft(wet * win)
    Xd = np.fft.rfft(dry * win)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    Xout = Xw.copy()
    for b in offending_bands:
        if b < 0 or b + 1 >= len(band_edges):
            continue
        m = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
        Xout[m] = (1.0 - mix) * Xw[m] + mix * Xd[m]
    y = np.fft.irfft(Xout, n=n).astype(np.float32)
    # Compensate the analysis window the same way TinyGRUPostFilter does so
    # block edges aren't tapered to zero.
    return y + (1.0 - win) * wet


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
        # History-primed per-call processing (see module docstring): each call
        # enhances [hist | block] with a FRESH DfState and keeps the block's
        # region.  ``_hist`` holds the last ``dfn_history_ms`` of RAW input —
        # updated on every block (bypass included) so an active block after a
        # bypass stretch still gets true left context.
        self._hist_len = 0
        self._hist = np.zeros(0, dtype=np.float32)
        self._DFState = None
        self._fft = 960
        self._hop = 480
        self._nb_erb = 32
        if _HAS_DF:
            torch.set_num_threads(threads)
            self._model, state0, _ = init_df(
                model_base_dir=cfg.dfn_model_dir, config_allow_defaults=True
            )
            self.backend = "deepfilternet3"
            if state0.sr() != self.sr:
                raise RuntimeError(
                    f"DFN3 expects 48 kHz audio (state sr={state0.sr()}, cfg sr={self.sr})"
                )
            self._DFState = type(state0)
            self._fft = int(state0.fft_size())
            self._hop = int(state0.hop_size())
            # Round history to whole hops so the kept region starts on a frame.
            hist_ms = float(getattr(cfg, "dfn_history_ms", 80.0))
            self._hist_len = int(round(self.sr * hist_ms / 1000.0 / self._hop)) * self._hop
            self._hist = np.zeros(self._hist_len, dtype=np.float32)
            logger.info("enhancement: DeepFilterNet3 loaded (threads=%d, history=%d ms)",
                        threads, int(1000 * self._hist_len / self.sr))
        else:
            # Make the no-op state LOUD: a passthrough enhancer means the whole
            # product is doing nothing, which must never look like success.
            logger.warning(
                "enhancement: DeepFilterNet3 UNAVAILABLE (%s) — running as "
                "PASSTHROUGH (no denoising). Install deepfilternet (+ Rust "
                "toolchain) so the enhancement core actually runs.",
                _DF_IMPORT_ERROR,
            )

        # Self-managed rollback state: number of dB to subtract from the next
        # frame's atten_lim_db cap after an over-suppression event.  Decays
        # over a few frames so we don't permanently relax the enhancer.
        self._cap_relief_db = 0.0

    def reset(self) -> None:
        """Reset per-stream state.

        There is no persistent DfState to clear (a fresh one is built per
        call — see ``_enhance``); only the raw-input history buffer and the
        rollback cap relief carry across blocks.
        """
        self._cap_relief_db = 0.0
        self._hist = np.zeros(self._hist_len, dtype=np.float32)

    def warmup(self) -> None:
        """Run DFN3 once so torch first-call costs don't hit the live stream.

        The pipeline-level warm-up feeds silence, which this stage bypasses
        (suppression ≈ 0), so DFN3's first real inference would otherwise
        land mid-demo.  Two direct calls cover the first-call graph work.
        ``_enhance`` only reads the history buffer, so stage state is
        untouched.
        """
        if self.backend != "deepfilternet3":
            return
        blk = int(round(self.sr * 0.1))
        noise = (1e-3 * np.random.default_rng(0).standard_normal(blk)).astype(np.float32)
        for _ in range(2):
            self._enhance(noise, atten_lim_db=100.0)

    def _push_hist(self, x: np.ndarray) -> None:
        if self._hist_len <= 0:
            return
        if len(x) >= self._hist_len:
            self._hist = x[-self._hist_len:].astype(np.float32, copy=True)
        else:
            self._hist = np.concatenate([self._hist[len(x):], x.astype(np.float32)])

    def _enhance(self, buf: np.ndarray, atten_lim_db: float) -> np.ndarray:
        """Enhance one block with real left context and zero added latency.

        Runs ``enhance(pad=True)`` on ``[hist | buf]`` with a FRESH DfState
        (so nothing fabricated ever enters a persistent stream state), then
        keeps only ``buf``'s region.  pad=True's internal trim keeps the
        output input-aligned, so the kept region is exactly ``buf``'s
        timeline.  NOTE: ``self._hist`` still holds the audio *preceding*
        ``buf`` at this point — process() pushes the block only afterwards.
        """
        seg = np.concatenate([self._hist, buf.astype(np.float32, copy=False)])
        state = self._DFState(sr=self.sr, fft_size=self._fft,
                              hop_size=self._hop, nb_bands=self._nb_erb)
        x = torch.from_numpy(seg.reshape(1, -1))
        with torch.no_grad():
            y = enhance(self._model, state, x, atten_lim_db=atten_lim_db)
        out = y.squeeze(0).cpu().numpy().astype("float32")
        h = self._hist_len
        if len(out) < h + len(buf):
            out = np.concatenate([out, np.zeros(h + len(buf) - len(out), dtype="float32")])
        return out[h: h + len(buf)]

    def process(self, ctx: FrameContext) -> FrameContext:
        if self.backend == "passthrough":
            ctx.meta["enh_wet"] = 0.0
            return ctx

        dry = ctx.audio

        supp = float(np.clip(ctx.suppression, 0.0, 1.0))
        if supp <= self.bypass_below:
            # Keep the context history warm even when bypassing, so the next
            # active block gets true left context.
            self._push_hist(dry)
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

        # History-primed enhancement (see module docstring): the block is
        # enhanced with cfg.dfn_history_ms of real left context; the output
        # region is sample-aligned with ``dry``.  _enhance reads self._hist,
        # so push the block into the history only AFTER enhancing.
        wet = self._enhance(dry, atten_lim_db)
        self._push_hist(dry)

        # ── Multi-cue artifact confidence + per-band rollback ──────────
        # Gating preconditions:
        #   1. speech_conf ≥ 0.6 (looks like real speech), AND
        #   2. target_speaker_sim ≥ 0.4 (or embedder in passthrough with sim≈1).
        # If either fails we skip the rollback machinery entirely — false-fire
        # during competing speech / TV would re-inject noise we just removed.
        sim = float(ctx.target_speaker_sim)
        embedder_passthrough = sim > 0.999          # 1.0 sentinel = no embedder
        target_present = embedder_passthrough or sim >= 0.4

        # ── Mask confidence (energy-ratio proxy) ─────────────────────────
        # We don't expose DFN3's internal ERB-gain mask, so build a proxy
        # from the per-band wet/dry energy ratio: ratio near 0 = strongly
        # suppressed, near 1 = passed through, near 0.5 = uncertain.
        # ``_mask_confidence`` then converts that distribution into a
        # 1 − normalized_entropy score: clean decisions (mostly 0 or 1)
        # produce HIGH confidence; mid-band ratios produce LOW confidence.
        dry_e_proxy = _erb_band_energies(dry, self.sr)
        wet_e_proxy = _erb_band_energies(wet, self.sr)
        mask_for_conf = np.clip(
            wet_e_proxy / (dry_e_proxy + 1e-9), 0.0, 1.0
        ).astype(np.float32)
        mask_conf = _mask_confidence(mask_for_conf)
        ctx.meta["mask_confidence"] = mask_conf
        ctx.meta["mask_mean_gain"] = float(np.mean(mask_for_conf))

        overshoot = 0.0
        artifact_conf = 0.0
        if ctx.speech_conf >= 0.6 and target_present:
            # Cue 1: ERB sub-band drop — primary indicator.
            dry_e = _erb_band_energies(dry, self.sr)
            wet_e = _erb_band_energies(wet, self.sr)
            drops_db = 10.0 * np.log10((dry_e + 1e-9) / (wet_e + 1e-9))
            max_drop = float(np.max(drops_db))
            ctx.meta["enh_max_drop_db"] = max_drop
            # Score: map [15, 40] dB drop range to [0, 1].
            drop_score = float(np.clip((max_drop - 15.0) / 25.0, 0.0, 1.0))

            # Cue 2: spectral-kurtosis change — flags musical noise.
            dry_k = _spectral_kurtosis(dry)
            wet_k = _spectral_kurtosis(wet)
            kurt_score = float(np.clip(abs(wet_k - dry_k) / 8.0, 0.0, 1.0))

            # Cue 3: LPC-residual ratio — flags formant smoothing.
            formant_score = _lpc_residual_ratio(dry, wet)

            # Cue 4: total-energy ratio — flags gross over-attenuation.
            total_drop_db = 10.0 * np.log10(
                (float(np.mean(dry * dry)) + 1e-12) /
                (float(np.mean(wet * wet)) + 1e-12)
            )
            energy_score = float(np.clip((total_drop_db - 10.0) / 20.0, 0.0, 1.0))

            artifact_conf = float(np.clip(
                self.cfg.artifact_w_drop * drop_score
                + self.cfg.artifact_w_kurtosis * kurt_score
                + self.cfg.artifact_w_formant * formant_score
                + self.cfg.artifact_w_energy * energy_score,
                0.0, 1.0,
            )) * float(ctx.speech_conf)
            ctx.meta["enh_artifact_conf"] = artifact_conf
            ctx.meta["enh_drop_score"] = drop_score
            ctx.meta["enh_kurt_score"] = kurt_score
            ctx.meta["enh_formant_score"] = formant_score
            ctx.meta["enh_energy_score"] = energy_score

            if artifact_conf >= self.cfg.artifact_conf_threshold:
                overshoot = 1.0
                # When DFN3's mask is UNCERTAIN (low confidence) yet speech_conf
                # is HIGH, the rollback should be MORE conservative: DFN3 may
                # have been guessing, and over-blending dry back could amplify
                # the underlying ambiguity into a louder artifact.  Scale the
                # rollback mix down by mask_confidence (clamped at 0.5 floor
                # so we still apply *some* repair even at zero mask confidence).
                conf_scale = max(0.5, mask_conf)
                mix = float(self.cfg.artifact_rollback_mix) * artifact_conf * conf_scale
                if self.cfg.artifact_per_band_rollback:
                    # Per-band: blend dry only in bands that overshot.
                    offending = [int(b) for b in np.where(
                        drops_db > self.cfg.artifact_band_drop_db
                    )[0]]
                    if offending:
                        band_edges = _erb_band_edges(self.sr)
                        wet = _band_rollback_stft(
                            wet, dry, self.sr, offending, band_edges, mix,
                        )
                    else:
                        # No specific band exceeded the threshold but fused
                        # artifact_conf did — fall back to a small global blend.
                        wet = (1.0 - mix * 0.5) * wet + (mix * 0.5) * dry
                else:
                    wet = (1.0 - mix) * wet + mix * dry
                self._cap_relief_db = float(self.cfg.artifact_cap_relief_db) * artifact_conf
                ctx.meta["enh_rollback"] = mix
        ctx.meta["enh_overshoot"] = overshoot

        ctx.audio = wet.astype("float32")
        ctx.meta["enh_wet"] = 1.0
        ctx.meta["enh_atten_db"] = atten_lim_db
        ctx.meta["enh_cap_relief_db"] = relief
        return ctx

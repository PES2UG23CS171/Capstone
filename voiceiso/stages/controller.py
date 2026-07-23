"""
Dynamic Suppression Controller — V2 (multi-band, multi-confidence).

V1 → V2 changes
~~~~~~~~~~~~~~~
* **Smooth SNR base.**  A continuous ``np.interp`` over SNR replaces the V1
  3-tier ``if/elif`` (which had audible steps at 10 dB and 20 dB boundaries).
* **VAD probability, not just is_speech.**  ``vad_prob`` continuously scales
  the speech-protection floor, so 0.55-confidence frames are treated less
  protectively than 0.99-confidence ones.
* **Per-band gains.**  Outputs ``ctx.band_gain = (g_lo, g_md, g_hi)`` — the
  controller's first multi-band capability.  Driven by:
    – noise class (fan → cut LF; keyboard → cut HF; …)
    – target-speaker similarity (competing speech → cut mid)
    – per-band residual-echo coherence from the AEC (echo leakage → cut that band)
* **Over-suppression rollback** is now handled inside the Enhancement stage
  itself (it has direct access to both dry and wet, and to its own DFN call)
  so the rollback applies on the very same frame the artifact is detected.
  This controller does not need to coordinate it.
* **Per-class release times** for transients (unchanged from V1.5).
* **Leaky clean-run counter.**  A single bad frame no longer fully resets the
  400 ms sustained-clean hold.

Inputs:  ``ctx.vad_prob``, ``ctx.speech_conf``, ``ctx.snr_db``, ``ctx.noise_class``,
         ``ctx.noise_probs``, ``ctx.target_speaker_sim``, ``ctx.res_band``,
         ``ctx.meta['competing_cut']``.

Outputs: ``ctx.suppression`` (→ DFN atten cap),
         ``ctx.band_gain`` (→ MultiBandGainModulator),
         ``ctx.postfilter_strength`` (→ PostFilter).
"""

from __future__ import annotations

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage

# Per-class suppression modifier (1.0 = base; <1.0 = softer; relative to SNR base).
_CLASS_SUPP_MODIFIER = {
    "clean": 0.0,
    "fan": 1.0, "hvac": 1.0,
    "traffic": 0.9, "wind": 0.9,
    "music": 1.0, "television": 1.0,
    "competing_speech": 0.95,
    "keyboard": 0.7,
    "mouse_click": 0.65,
    "dog_bark": 0.65,
    "door_slam": 0.6,
}

_CLASS_POSTFILTER = {
    "clean": 0.0, "fan": 0.4, "hvac": 0.4, "traffic": 0.6, "wind": 0.7,
    "keyboard": 0.5, "mouse_click": 0.5, "dog_bark": 0.6, "door_slam": 0.6,
    "music": 0.8, "television": 0.8, "competing_speech": 0.9,
}

_CLASS_RELEASE_MS = {
    "keyboard": 50.0, "mouse_click": 50.0,
    "dog_bark": 60.0, "door_slam": 60.0,
}

# Per-band gain hints by noise class.  Multiplicative, applied AFTER DFN3.
# (lo, md, hi) — 1.0 = no extra attenuation in that band.
_CLASS_BAND_GAIN = {
    "clean":           (1.00, 1.00, 1.00),
    "fan":             (0.50, 1.00, 1.00),   # extra LF cleanup
    "hvac":            (0.55, 1.00, 1.00),
    "wind":            (0.30, 1.00, 1.00),
    "traffic":         (0.70, 1.00, 0.90),
    "keyboard":        (1.00, 1.00, 0.40),   # extra HF cleanup
    "mouse_click":     (1.00, 1.00, 0.50),
    "dog_bark":        (1.00, 0.85, 0.85),
    "door_slam":       (0.70, 1.00, 1.00),
    "music":           (1.00, 0.80, 0.90),
    "television":      (1.00, 0.75, 0.90),
    "competing_speech":(1.00, 0.40, 1.00),   # cut mid (speech band)
}


class DynamicController(Stage):
    name = "controller"

    def __init__(self, cfg: PipelineConfig, clean_snr_db: float = 22.0,
                 clean_hold_ms: float = 400.0) -> None:
        self.cfg = cfg
        self._supp = cfg.supp_max
        self._clean_snr = clean_snr_db
        # Smoothing/hold are tracked in MILLISECONDS of audio, derived from the
        # actual block duration each call — NOT a fixed per-call assumption.
        # (Previously attack/release/hold used cfg.hop_ms but ran once per block,
        # so the real time-constants were off by the block/hop ratio — 10x at the
        # 100 ms live block.)
        self._clean_hold_ms = clean_hold_ms
        self._clean_ms = 0.0          # ms of sustained confidently-clean audio
        # Slow room-level average (dB) for transient detection.  In a quiet
        # room the SNR estimator reads sparse impulses (claps, snaps, door
        # slams) as "signal over a silent floor" → high SNR → suppression
        # relaxes → each successive impulse is LESS suppressed (measured:
        # clap #1 −52 dB, #3 only −23 dB).  A block whose level jumps far
        # above this average with no voice activity gets a suppression KICK
        # so DFN3 engages at full strength on every impulse.
        self._rms_slow_db = -70.0
        # Smoothed outputs (H3): band gains and post-filter strength get the
        # same attack/release treatment as _supp — previously they stepped
        # straight to the class table's values on a classifier flip (e.g.
        # clean→keyboard = −8 dB on g_hi in one block edge).
        self._band = np.ones(3, dtype=np.float64)
        self._pf = 0.0

    def reset(self) -> None:
        self._supp = self.cfg.supp_max
        self._clean_ms = 0.0
        self._band = np.ones(3, dtype=np.float64)
        self._pf = 0.0
        self._rms_slow_db = -70.0

    @staticmethod
    def _smooth_snr_base(snr_db: float) -> float:
        """Continuous SNR-to-base mapping (replaces V1's 3-tier step function).

            SNR ≤  0 dB → base = 1.00 (heavy noise → full DFN)
            SNR ≈ 10 dB → base = 0.80
            SNR ≈ 20 dB → base = 0.50 (residuals only)
            SNR ≥ 30 dB → base = 0.20
        """
        return float(np.interp(snr_db,
                               [0.0, 10.0, 20.0, 30.0],
                               [1.00, 0.80, 0.50, 0.20]))

    def _compute_target(self, ctx: FrameContext) -> float:
        """Return the desired suppression ∈ [supp_min, supp_max]."""
        if self._clean_ms >= self._clean_hold_ms:
            return self.cfg.supp_min          # sustained clean → bypass

        base = self._smooth_snr_base(ctx.snr_db)

        # Classifier confidence: low confidence → don't over-commit.
        top_prob = ctx.noise_conf if ctx.noise_conf > 0.0 else (
            max(ctx.noise_probs.values()) if ctx.noise_probs else 0.5
        )
        conf_scale = float(np.interp(top_prob, [0.3, 0.9], [0.7, 1.0]))

        modifier = _CLASS_SUPP_MODIFIER.get(ctx.noise_class, 0.85)

        # Speech-protection floor: when vad_prob is high, never go above 0.92
        # (so DFN's atten cap never reaches the most aggressive end of the range
        # while we are confident speech is active).
        speech_cap = float(np.interp(ctx.vad_prob, [0.4, 0.9], [1.0, 0.92]))

        # Whisper protection: whispered speech is unvoiced, HF-dominant, and
        # quiet — DFN3's default attenuation will treat it as noise.  Halve
        # the suppression to keep whispers intelligible.
        whisper = float(ctx.meta.get("whisper_score", 0.0))
        whisper_scale = 1.0 - 0.5 * whisper            # 1.0 → 0.5 as whisper → 1.0

        return float(np.clip(base * modifier * conf_scale * whisper_scale,
                             self.cfg.supp_min,
                             min(self.cfg.supp_max, speech_cap)))

    def _compute_band_gains(self, ctx: FrameContext) -> tuple[float, float, float]:
        """Per-band gain modulation applied AFTER DFN3."""
        g_lo, g_md, g_hi = _CLASS_BAND_GAIN.get(ctx.noise_class, (1.0, 1.0, 1.0))

        # Competing speech: deeper mid-band cut when the embedder has a real
        # similarity score and it's well below threshold.
        cut = float(ctx.meta.get("competing_cut", 0.0))
        if cut > 0.0:
            # Interpolate g_md from current toward 0.10 by cut strength.
            g_md = float(g_md * (1.0 - cut) + 0.10 * cut)

        # Residual-echo cleanup: bands with high coherence get extra attenuation.
        if ctx.dt_active or any(c > 0.3 for c in ctx.res_band):
            res_lo, res_md, res_hi = ctx.res_band
            g_lo *= float(np.interp(res_lo, [0.3, 0.7], [1.0, 0.4]))
            g_md *= float(np.interp(res_md, [0.3, 0.7], [1.0, 0.5]))
            g_hi *= float(np.interp(res_hi, [0.3, 0.7], [1.0, 0.5]))

        # Fused echo-confidence: when AEC + NLES still report that the frame
        # is echo-contaminated (echo_conf high), pull all three bands toward
        # 0.7 of their current value — a uniform extra ~3 dB suppression
        # on top of any per-band response above.  This catches the case
        # where echo is spectrally diffuse and no single band's RES coherence
        # exceeds 0.3.
        if ctx.echo_conf > self.cfg.echo_conf_threshold:
            ec_scale = float(np.interp(ctx.echo_conf, [0.5, 1.0], [1.0, 0.7]))
            g_lo *= ec_scale
            g_md *= ec_scale
            g_hi *= ec_scale

        # Speech protection in LF / HF.  Voice fundamentals live at 80–250 Hz;
        # fricatives at >4 kHz.  Never cut these >6 dB (g >= 0.5) during speech.
        if ctx.vad_prob >= 0.6:
            g_lo = max(g_lo, 0.5)
            g_hi = max(g_hi, 0.5)

        return (
            float(np.clip(g_lo, 0.05, 1.0)),
            float(np.clip(g_md, 0.05, 1.0)),
            float(np.clip(g_hi, 0.05, 1.0)),
        )

    def process(self, ctx: FrameContext) -> FrameContext:
        cfg = self.cfg

        # Real block duration (ms) — drives every time constant so behaviour is
        # identical regardless of the caller's block size.
        dt_ms = 1000.0 * len(ctx.audio) / max(cfg.sample_rate, 1)

        # ── Transient kick ───────────────────────────────────────────────
        # Impulse (clap/snap/slam) detection: block level far above the slow
        # room average with no voice activity.  Without this, sparse impulses
        # in a quiet room read as high SNR and suppression RELAXES with each
        # one.  A kick forces DFN3 on at high strength for the block; a false
        # positive on a word onset is fail-safe (the block just gets normal
        # denoising — DFN3 preserves speech).
        rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(ctx.audio ** 2))) + 1e-12)
        transient_kick = (
            (rms_db - self._rms_slow_db) >= 15.0
            and not ctx.is_speech          # VAD verdict incl. hangover: never
                                           # kick during (or right after) the
                                           # user's own speech — a mid-speech
                                           # kick maxes DFN3 on a block that
                                           # CONTAINS speech and audibly
                                           # mangles it (claps don't trip
                                           # Silero: measured vad≤0.03)
            and rms_db > -50.0
            and self._rms_slow_db < -45.0  # QUIET-room arm only: in steady
                                           # noise (rain/fan/white) suppression
                                           # is already high and phrase onsets
                                           # over the noise would false-fire
                                           # the jump detector before VAD
                                           # catches up (measured: audible
                                           # onset distortion over rain)
            and ctx.snr_db > 20.0          # …and the cold-start hole: the slow
                                           # average starts at −70 after every
                                           # warm-up, so a loud room reads
                                           # "quiet" for ~2 s.  An impulse over
                                           # true silence reads high SNR
                                           # (signal over silent floor); steady
                                           # noise reads low SNR — this gate is
                                           # valid from block 0.
        )
        if not transient_kick:
            # Track the room level only from non-impulse blocks (τ ≈ 3 s) so a
            # burst of claps can't drag the average up and mask itself.
            a_rms = 1.0 - np.exp(-dt_ms / 3000.0)
            self._rms_slow_db += a_rms * (rms_db - self._rms_slow_db)
        ctx.meta["transient_kick"] = float(transient_kick)

        # ── Leaky clean-hold timer (in ms) ───────────────────────────────
        confidently_clean = (
            (not ctx.is_speech)
            and ctx.snr_db >= self._clean_snr
            and ctx.noise_class == "clean"
            and not transient_kick
        )
        if confidently_clean:
            self._clean_ms += dt_ms
        elif transient_kick:
            self._clean_ms = 0.0        # an impulse means the room is NOT clean
        else:
            # Decay 2× faster than it builds — a single flicker doesn't kill the
            # hold, but a real onset clears it quickly.
            self._clean_ms = max(0.0, self._clean_ms - 2.0 * dt_ms)

        target = self._compute_target(ctx)
        if transient_kick:
            target = max(target, 0.85)

        # ── Attack / per-class release smoothing (time-constant in ms) ───
        release_ms = _CLASS_RELEASE_MS.get(ctx.noise_class, cfg.supp_release_ms)
        a_attack = 1.0 - np.exp(-dt_ms / max(cfg.supp_attack_ms, 1e-3))
        a_release = 1.0 - np.exp(-dt_ms / max(release_ms, 1e-3))
        a = a_attack if target > self._supp else a_release
        self._supp += a * (target - self._supp)
        ctx.suppression = float(np.clip(self._supp, cfg.supp_min, cfg.supp_max))

        # ── Post-filter aggressiveness (attack/release smoothed) ─────────
        pf = _CLASS_POSTFILTER.get(ctx.noise_class, 0.5)
        snr_scale = float(np.interp(ctx.snr_db, [0.0, 25.0], [1.0, 0.3]))
        pf_target = float(np.clip(pf * snr_scale, 0.0, 1.0))
        a_pf = a_attack if pf_target > self._pf else a_release
        self._pf += a_pf * (pf_target - self._pf)
        ctx.postfilter_strength = float(np.clip(self._pf, 0.0, 1.0))

        # ── Per-band gains (attack/release smoothed) ─────────────────────
        # Gain DOWN (more cut) follows the attack constant; gain UP (recover
        # toward unity) follows the release constant, so a classifier flip
        # glides instead of stepping.  The MultiBandGainModulator additionally
        # ramps per-sample inside each block.
        band_target = np.asarray(self._compute_band_gains(ctx), dtype=np.float64)
        a_band = np.where(band_target < self._band, a_attack, a_release)
        self._band += a_band * (band_target - self._band)
        ctx.band_gain = (float(self._band[0]), float(self._band[1]), float(self._band[2]))

        ctx.meta["ctrl_target"] = target
        ctx.meta["band_gain_target"] = tuple(float(g) for g in band_target)
        ctx.meta["clean_ms"] = self._clean_ms
        return ctx

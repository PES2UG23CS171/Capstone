"""
Unified configuration for the voiceiso real-time pipeline.

Frame geometry is chosen to match DeepFilterNet3's native 48 kHz / 10 ms-hop
operation so the enhancement core runs at its design point.

Latency accounting (honest):

    STFT framing (``algorithmic_latency_ms``): hop (10 ms) + window overhang
    (10 ms) + optional lookahead ≈ 20 ms.  This is NOT the end-to-end number.

    The live paths process 100 ms blocks on a worker thread (DFN3's GRUs are
    stateless across calls, so per-call context ~100 ms is the quality design
    point).  End-to-end mouth-to-ear = 100 ms block fill + one queue hop
    (≤ 100 ms) + device I/O ≈ **200–300 ms**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class PipelineConfig:
    # ── Core sample rate / framing ───────────────────────────────────────
    sample_rate: int = 48_000          # DFN3 native rate
    hop_ms: float = 10.0               # 480 samples @ 48 kHz
    win_ms: float = 20.0               # 960 samples @ 48 kHz (STFT analysis)
    lookahead_ms: float = 0.0          # extra future context (raises latency)
    channels: int = 1

    # ── VAD (Silero runs at 16 kHz) ──────────────────────────────────────
    vad_sample_rate: int = 16_000
    vad_window: int = 512              # Silero v5/v6 frame size @ 16 kHz (32 ms)
    vad_speech_threshold: float = 0.5  # base P(speech) threshold (used if adaptation off)
    vad_hangover_ms: float = 200.0     # default hangover (vowel-ending case)
    # Adaptive threshold (track running percentiles of vad_prob over ~30 s).
    vad_adaptive: bool = True
    vad_adapt_alpha: float = 1.0 / 1500.0  # per-call EWMA α (≈150 s @ 100 ms blocks)
    vad_threshold_min: float = 0.35
    vad_threshold_max: float = 0.65
    # Adaptive hangover: short when speech ended on a fricative (HF-dominant
    # spectrum at offset → voice tail is brief).  Long for vowel endings.
    vad_hangover_fricative_ms: float = 80.0
    vad_fricative_hf_ratio: float = 0.45     # HF-energy fraction threshold

    # ── Noise classification ─────────────────────────────────────────────
    noise_classes: Tuple[str, ...] = (
        "keyboard", "mouse_click", "dog_bark", "door_slam", "fan",
        "hvac", "traffic", "wind", "music", "television", "competing_speech",
        "clean",
    )
    noise_clf_window_ms: float = 320.0  # log-mel context for classification

    # ── Dynamic suppression controller bounds ────────────────────────────
    supp_min: float = 0.0              # never suppress below this (dry passthrough)
    supp_max: float = 1.0              # fully wet (max suppression)
    supp_speech_floor: float = 0.55    # cap suppression while speech is active
    supp_attack_ms: float = 30.0       # how fast aggressiveness ramps up
    supp_release_ms: float = 150.0     # how fast it ramps down (avoid pumping)

    # ── Post-filter / comfort noise ──────────────────────────────────────
    # Ceiling for injected comfort noise.  -65 dBFS was audible as a constant
    # "static air hiss" on headphones once real room noise was actually
    # suppressed (measured: fan-room output floor sat exactly at the CN
    # ceiling).  -78 dBFS keeps the dead-air masking function while sitting
    # below audibility at normal playback gain.
    comfort_noise_db: float = -78.0
    postfilter_floor_db: float = -18.0 # max additional residual attenuation

    # ── AEC ──────────────────────────────────────────────────────────────
    aec_enabled: bool = False          # requires a reference (loopback) signal
    aec_filter_ms: float = 200.0       # adaptive filter length (tail coverage)
    aec_dtd_threshold: float = 0.5     # Geigel double-talk detector threshold
    aec_dt_freeze_ms: float = 1000.0   # how long to freeze NLMS after double-talk
    aec_step_size: float = 0.15        # NLMS µ; halved from V1 (was 0.3) for stability
    # ── GCC-PHAT one-shot delay calibration ──────────────────────────────
    aec_gcc_phat: bool = True          # estimate loopback delay once at session start
    aec_gcc_max_delay_ms: float = 100.0  # plausible-range clamp for the delay estimate
    aec_gcc_peak_threshold: float = 0.2  # min normalised peak to accept; else fall back to 0
    # ── Nonlinear echo suppression (NLES, per-bin Wiener) ────────────────
    nles_enabled: bool = True
    nles_beta: float = 1.5             # over-estimation factor in Wiener formula
    nles_min_gain_db: float = -20.0    # floor — never attenuate a bin by more than this
    # ── Fused echo-confidence ────────────────────────────────────────────
    echo_conf_threshold: float = 0.5   # consumers fire when echo_conf > this

    # ── Multi-band controller (V2) ───────────────────────────────────────
    # Band split frequencies for post-DFN per-band gain modulation.
    band_low_hz: float = 300.0         # < this = low band
    band_high_hz: float = 4000.0       # > this = high band; between = mid band
    # Linear-phase FIR length.  31 taps → group delay 15 samples ≈ 0.31 ms at
    # 48 kHz — short enough that bench SI-SDR is preserved (delay-sensitive
    # metric), still long enough to give meaningful band separation for the
    # coarse 3-band split.  Earlier 65-tap design (0.67 ms) tanked SI-SDR by
    # ~25 dB on every block boundary the controller switched gain modes.
    band_fir_taps: int = 31

    # ── Speaker (target-speaker extraction) ──────────────────────────────
    speaker_model_path: Optional[str] = None     # ECAPA-TDNN ONNX (None = disabled)
    speaker_enroll_path: Optional[str] = None    # path to enrolled 192-dim x-vector
    speaker_sim_threshold: float = 0.25          # below = strong competing-speech cut
    speaker_window_ms: float = 200.0             # ECAPA analysis window
    speaker_update_ms: float = 100.0             # rerun ECAPA at this cadence

    # ── Post-filter (learned tiny GRU) ───────────────────────────────────
    postfilter_model_path: Optional[str] = None  # tiny GRU ONNX (None = bypass)

    # ── Target-speaker extraction (VoiceFilter-Lite-style) ───────────────
    target_speaker_mask_path: Optional[str] = None  # mask network ONNX (None = bypass)

    # ── Benchmarking ─────────────────────────────────────────────────────
    # Microsoft DNSMOS P.835 primary model (sig_bak_ovr.onnx).  Non-intrusive
    # SIG/BAK/OVRL prediction.  None → DNSMOS metrics are skipped.
    dnsmos_model_path: Optional[str] = None

    # ── Speech-quality protection (over-suppression rollback) ────────────
    artifact_band_drop_db: float = 25.0  # sub-band drop that flags over-suppression
    artifact_rollback_mix: float = 0.30  # dry-blend fraction during rollback
    artifact_cap_relief_db: float = 12.0 # how much to relax next-frame atten cap
    # Multi-cue artifact-confidence fusion weights (sum ~ 1.0).
    artifact_w_drop: float = 0.40        # ERB sub-band drop
    artifact_w_kurtosis: float = 0.30    # spectral-kurtosis change (musical noise)
    artifact_w_formant: float = 0.20     # LPC formant preservation
    artifact_w_energy: float = 0.10      # total-energy ratio
    artifact_conf_threshold: float = 0.5  # fused artifact conf above this fires rollback
    # Per-band rollback: blend dry only in the offending ERB band(s) — narrower
    # noise re-injection than V1's whole-spectrum dry-blend.
    artifact_per_band_rollback: bool = True

    # ── Speech preservation ──────────────────────────────────────────────
    # Sticky speech_conf — current frame can't drop below ``stickiness`` × prior.
    speech_conf_stickiness: float = 0.7
    # Pitch detector range (Hz) — covers adult speakers male & female.
    pitch_lo_hz: float = 70.0
    pitch_hi_hz: float = 400.0

    # ── Noise classification (learned) ───────────────────────────────────
    # When set + onnxruntime available, EfficientAT-S replaces the heuristic
    # noise classifier as the primary path.  Heuristic remains the fallback.
    # Default None → ``__post_init__`` auto-resolves the shipped native-12 head
    # (checkpoints/efficientat_head12.onnx) relative to the repo root, so the
    # learned classifier is the PRIMARY path out of the box (CWD-independent).
    noise_classifier_model_path: Optional[str] = None
    # The shipped head ONNX (efficientat_head12.onnx) was trained AND exported
    # with a STATIC 4.0 s / 400-frame mel input (head_12.pt window_s = 4.0).
    # The runtime window MUST match the model's frame count or every inference
    # raises InvalidArgument.  EfficientATNoiseClassifier additionally reads the
    # model's required frame count from the ONNX input shape and pads/crops to
    # it, so this value is the buffering target, not a silent failure point.
    noise_classifier_window_s: float = 4.0     # mn10_as needs ~4 s context
    # Rerun cadence (ms of input audio between inferences).  The 4 s-window
    # backbone costs ~11 ms/run on CPU; 500 ms cadence amortises that to ~2 %
    # of one core while the noise class changes far slower than that anyway.
    noise_classifier_hop_ms: float = 500.0     # rerun cadence
    # Minimum buffered audio before the FIRST inference.  A partial buffer
    # (≥ this, < window) is tile-repeated to the 4 s window — mirroring the
    # training pad policy for short clips — so the classifier can label the
    # first seconds of a session / short clips instead of being forced to
    # "clean" until the full window fills.
    noise_classifier_min_infer_s: float = 1.0
    noise_classifier_smooth_alpha: float = 0.5  # EWMA over per-class posteriors
    # A noise/speech target must exceed this smoothed posterior to win over
    # "clean".  AudioSet posteriors for a correct coarse class on a short clip
    # are modest (~0.2–0.4), so a low threshold is right; below it → clean.
    noise_classifier_clean_threshold: float = 0.15

    # ── Adaptive HP cutoff under wind ────────────────────────────────────
    # When previous-frame noise_class == 'wind', Preprocessing switches to this
    # higher cutoff to attenuate the broadband LF rumble that defeats the
    # default 25 Hz cutoff.  1st-order filter — only ~6 dB at 80 Hz fundamental.
    wind_hp_cutoff_hz: float = 80.0

    # ── Comfort-noise boost during echo-only suppression ────────────────
    # When echo_conf > echo_conf_threshold AND vad_prob is low, raise the
    # comfort-noise level by this many dB so the DT-suppressed gap doesn't
    # sound like a dropped call.  Capped above comfort_noise_db.
    cn_echo_boost_db: float = 6.0

    # ── Live-stream queue / latency mode ─────────────────────────────────
    # Trade-off:  small queue + latency='low' → minimum end-to-end delay but
    #             intolerant of CPU bursts (xrun → silent drop).
    #             Larger queue + latency='high' → more delay but smoother.
    # Defaults pick a middle ground: 4-deep queue (400 ms of slack at 100 ms
    # blocks) with latency='low'.  V1 used 16-deep + 'high' which silently
    # ballooned latency to 1.6 s under load; V2-α used 2-deep + 'low' which
    # was too tight on stock laptops.  The callback drains the OUTPUT queue
    # to the freshest block, so queue depth adds slack, not steady latency.
    live_queue_maxsize: int = 4
    live_latency_mode: str = "low"

    # ── Enhancement (DFN3) streaming context ─────────────────────────────
    # Real left-context re-enhanced with each block (history-primed per-call
    # processing — see stages/enhancement.py module docstring).  80 ms is the
    # measured saturation point: 17.7 dB SI-SDR on speech @ 5 dB SNR / 100 ms
    # blocks (one-shot bound 18.0 dB; naive per-block stateful mode 11.5 dB),
    # at RTF ≈ 0.17.  160/240 ms measure no better.
    dfn_history_ms: float = 80.0

    # ── Paths ────────────────────────────────────────────────────────────
    dfn_model_dir: Optional[str] = None  # None → DFN3 default bundled model
    data_root: str = "data"
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self) -> None:
        # Auto-wire the shipped native-12 EfficientAT head when the caller did
        # not pin a path, resolving it relative to the repo root so it works no
        # matter what the current working directory is.  If the checkpoint is
        # absent the path stays None and the pipeline falls back (loudly) to the
        # heuristic classifier — see StreamingPipeline.
        if self.noise_classifier_model_path is None:
            from pathlib import Path
            import os as _os
            repo_root = Path(__file__).resolve().parent.parent
            ck = Path(self.checkpoint_dir)
            if not ck.is_absolute():
                ck = repo_root / ck
            # Explicit head override (e.g. VOICEISO_HEAD=checkpoints/
            # efficientat_head12_v4.onnx enables the opt-in TV-rejection head,
            # which trades −0.02 FSD50K macro-F1 for 0.54→0.86 loudspeaker-
            # speech rejection — see ARCHITECTURE.md §10).
            env_head = _os.environ.get("VOICEISO_HEAD")
            if env_head:
                cand = Path(env_head)
                if not cand.is_absolute():
                    cand = repo_root / cand
                if cand.exists():
                    self.noise_classifier_model_path = str(cand)
            # Preference order: v3 (correct protocol + rolling-window-robust
            # training) > v2 (correct protocol, first-4s crops) > v1 (legacy
            # eval-pool training).  v4 (TV-rejection) is opt-in only, via
            # VOICEISO_HEAD — it failed the pre-registered no-regression
            # deploy gate on FSD50K eval (−0.02 macro-F1) despite its
            # decisive loudspeaker-rejection gain.
            for fname in (() if self.noise_classifier_model_path else
                          ("efficientat_head12_v3.onnx",
                           "efficientat_head12_v2.onnx",
                           "efficientat_head12.onnx")):
                candidate = ck / fname
                if candidate.exists():
                    self.noise_classifier_model_path = str(candidate)
                    break

        # Same auto-wiring for the DNSMOS P.835 model so a dropped-in
        # checkpoints/sig_bak_ovr.onnx is used (and reproducible) without flags.
        if self.dnsmos_model_path is None:
            from pathlib import Path
            repo_root = Path(__file__).resolve().parent.parent
            ck = Path(self.checkpoint_dir)
            if not ck.is_absolute():
                ck = repo_root / ck
            dns = ck / "sig_bak_ovr.onnx"
            if dns.exists():
                self.dnsmos_model_path = str(dns)

    # ── Derived (samples) ─────────────────────────────────────────────────
    @property
    def hop(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def win(self) -> int:
        return int(round(self.sample_rate * self.win_ms / 1000.0))

    @property
    def lookahead(self) -> int:
        return int(round(self.sample_rate * self.lookahead_ms / 1000.0))

    @property
    def algorithmic_latency_ms(self) -> float:
        return self.hop_ms + (self.win_ms - self.hop_ms) + self.lookahead_ms

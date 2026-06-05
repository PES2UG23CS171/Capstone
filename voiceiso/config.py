"""
Unified configuration for the voiceiso real-time pipeline.

Frame geometry is chosen to match DeepFilterNet3's native 48 kHz / 10 ms-hop
operation so the enhancement core runs at its design point, while keeping
algorithmic latency low enough for conferencing.

Latency budget (algorithmic, excluding device I/O)::

    frame hop (10 ms) + STFT window overhang (10 ms) + optional lookahead (0–10 ms)
    ≈ 20–30 ms   — comparable to Krisp's "low" mode.
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
    vad_speech_threshold: float = 0.5  # P(speech) above → speech
    vad_hangover_ms: float = 200.0     # keep "speech" state this long after offset

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
    comfort_noise_db: float = -65.0    # injected to mask dead-silence artifacts
    postfilter_floor_db: float = -18.0 # max additional residual attenuation

    # ── AEC ──────────────────────────────────────────────────────────────
    aec_enabled: bool = False          # requires a reference (loopback) signal
    aec_filter_ms: float = 200.0       # adaptive filter length (tail coverage)
    aec_dtd_threshold: float = 0.5     # Geigel double-talk detector threshold
    aec_dt_freeze_ms: float = 1000.0   # how long to freeze NLMS after double-talk
    aec_step_size: float = 0.15        # NLMS µ; halved from V1 (was 0.3) for stability

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

    # ── Speech-quality protection (over-suppression rollback) ────────────
    artifact_band_drop_db: float = 25.0  # sub-band drop that flags over-suppression
    artifact_rollback_mix: float = 0.30  # dry-blend fraction during rollback
    artifact_cap_relief_db: float = 12.0 # how much to relax next-frame atten cap

    # ── Live-stream queue / latency mode ─────────────────────────────────
    # Trade-off:  small queue + latency='low' → minimum end-to-end delay but
    #             intolerant of CPU bursts (xrun → silent drop).
    #             Larger queue + latency='high' → more delay but smoother.
    # Defaults pick a middle ground: 4-deep queue (~80 ms of slack at 20 ms
    # blocks) with latency='low'.  V1 used 16-deep + 'high' which silently
    # ballooned latency to 1.6 s under load; V2-α used 2-deep + 'low' which
    # was too tight on stock laptops.
    live_queue_maxsize: int = 4
    live_latency_mode: str = "low"

    # ── Paths ────────────────────────────────────────────────────────────
    dfn_model_dir: Optional[str] = None  # None → DFN3 default bundled model
    data_root: str = "data"
    checkpoint_dir: str = "checkpoints"

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

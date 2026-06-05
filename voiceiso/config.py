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

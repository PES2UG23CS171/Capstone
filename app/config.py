"""
Runtime configuration for the audio-filter application.

Shared between the GUI process and the audio-engine child process via
serialisable dataclass instances passed through multiprocessing Queues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppConfig:
    """Immutable-ish runtime knobs sent to the audio engine at startup."""

    # ── Audio stream parameters ──────────────────────────────────────────
    sample_rate: int = 48_000
    # 20 ms blocks: stateful DFN3 streaming — each call processes only new audio,
    # GRU state carries temporal context, no context-buffer overhead.
    # End-to-end latency ≈ 20–25 ms (vs 100 ms with the old overlap-save design).
    block_size: int = 960           # frames per callback (20 ms @ 48 kHz)
    channels: int = 1               # mono processing
    dtype: str = "float32"          # PCM format inside NumPy arrays

    # ── Device indices (None → system default) ───────────────────────────
    input_device: Optional[int] = None
    output_device: Optional[int] = None

    # ── Processing defaults ──────────────────────────────────────────────
    suppression_enabled: bool = True
    suppression_strength: float = 1.0   # 0.0 = bypass … 1.0 = full denoise
    output_gain_db: float = 0.0         # post-processing gain

    # ── ONNX model path (Phase 2) ────────────────────────────────────────
    model_path: Optional[str] = None

    # ── Status polling interval (seconds) ────────────────────────────────
    status_interval: float = 0.05       # 50 ms → ~20 fps meter updates

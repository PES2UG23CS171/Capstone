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
    # 100 ms blocks: DFN3's efficient design point.  DFN3 has NO cross-call GRU
    # memory, so per-call temporal context = the STFT frames in the block; at
    # 20 ms (≈1–2 frames) measured SI-SDRi is negative, at 100 ms it is +4 dB on
    # the worst-case white-noise condition.  The engine runs the chain on a
    # worker thread (not the audio callback), so a 100 ms block is xrun-safe.
    # End-to-end latency ≈ block + one queue hop + compute (~200 ms).
    block_size: int = 4800          # frames per callback (100 ms @ 48 kHz)
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

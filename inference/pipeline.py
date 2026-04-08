"""
Real-Time Processing Pipeline
==============================
Wires all 4 layers together: ring buffer (L1) → quantized model (L2) →
DeepFIR (L3) → Mamba SSM (L4) via the ONNX inference runner.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np

import config as cfg
from audio.ring_buffer import RingBuffer
from inference.onnx_runner import ONNXInferenceRunner


class LatencyTracker:
    """Rolling window of processing-time measurements."""

    def __init__(self, max_samples: int = 1000) -> None:
        self._times: deque = deque(maxlen=max_samples)

    def record(self, elapsed_ms: float) -> None:
        self._times.append(elapsed_ms)

    @property
    def avg_ms(self) -> float:
        return sum(self._times) / len(self._times) if self._times else 0.0

    @property
    def max_ms(self) -> float:
        return max(self._times) if self._times else 0.0

    @property
    def rtf(self) -> float:
        """Real-Time Factor (processing time / audio duration)."""
        if not self._times:
            return 0.0
        chunk_duration_ms = cfg.CONTEXT_WINDOW_SAMPLES / cfg.SAMPLE_RATE * 1000
        return self.avg_ms / chunk_duration_ms


class RealTimePipeline:
    """Real-time inference pipeline gluing all processing layers.

    Parameters
    ----------
    onnx_runner : ONNXInferenceRunner
        The ONNX model runner.
    ring_buffer : RingBuffer
        Shared ring buffer for audio context.
    bypass_mode : bool
        If True, pass audio through unmodified (for A/B comparison).
    suppression_level : float
        Wet/dry mix: 0.0 = fully bypassed, 1.0 = fully suppressed.
    """

    def __init__(
        self,
        onnx_runner: ONNXInferenceRunner,
        ring_buffer: RingBuffer,
        bypass_mode: bool = False,
        suppression_level: float = 1.0,
    ) -> None:
        self.runner = onnx_runner
        self.buffer = ring_buffer
        self.bypass_mode = bypass_mode
        self.suppression_level = suppression_level
        self.latency_tracker = LatencyTracker()

    def process_sample(self, sample: float) -> float:
        """Process one audio sample through the full pipeline.

        Steps:
        1. Write sample to ring buffer (Layer 1)
        2. Read context window from ring buffer
        3. Run ONNX inference (Layers 2+3+4 fused)
        4. Return the centre sample of the output (overlap-add)
        5. Track latency
        """
        if self.bypass_mode:
            return sample

        # 1. Write to ring buffer
        self.buffer.write(np.array([sample], dtype=np.float32))

        # 2. Read context window
        context = self.buffer.read_context()

        # 3. Run inference
        t0 = time.perf_counter()
        clean = self.runner.run(context)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 4. Extract centre sample (overlap-add)
        centre_idx = cfg.CONTEXT_WINDOW_SAMPLES // 2
        clean_sample = float(clean[0, centre_idx])

        # 5. Track latency
        self.latency_tracker.record(elapsed_ms)

        # Apply suppression level (wet/dry mix)
        if self.suppression_level < 1.0:
            clean_sample = (
                self.suppression_level * clean_sample +
                (1.0 - self.suppression_level) * sample
            )

        return clean_sample

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process a chunk of samples through the pipeline.

        More efficient than per-sample processing — runs inference once
        on the full context and returns the filtered chunk.
        """
        if self.bypass_mode:
            return chunk.copy()

        n = len(chunk)

        # Write chunk to ring buffer
        self.buffer.write(chunk)

        # Read context and run inference
        context = self.buffer.read_context()

        t0 = time.perf_counter()
        clean = self.runner.run(context)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.latency_tracker.record(elapsed_ms)

        # Extract the last n samples from the output
        clean_chunk = clean[0, -n:]

        # Wet/dry mix
        if self.suppression_level < 1.0:
            clean_chunk = (
                self.suppression_level * clean_chunk +
                (1.0 - self.suppression_level) * chunk
            )

        return clean_chunk.astype(np.float32)

    def get_stats(self) -> dict:
        """Return current pipeline statistics."""
        return {
            "avg_latency_ms": round(self.latency_tracker.avg_ms, 3),
            "max_latency_ms": round(self.latency_tracker.max_ms, 3),
            "rtf": round(self.latency_tracker.rtf, 6),
            "buffer_fill_pct": round(
                self.buffer.fill_level / self.buffer.capacity * 100, 1
            ),
            "bypass_mode": self.bypass_mode,
            "suppression_level": self.suppression_level,
        }

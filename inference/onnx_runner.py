"""
ONNX Runtime Inference Runner
==============================
Wraps ``onnxruntime.InferenceSession`` for real-time CPU inference with
pre-allocated buffers and low-latency execution settings.
"""

from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

import config as cfg

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class ONNXInferenceRunner:
    """ONNX Runtime inference engine for real-time audio processing.

    Parameters
    ----------
    onnx_path : str
        Path to the ``.onnx`` model file.
    context_window : int
        Expected input length.
    """

    def __init__(
        self,
        onnx_path: str = cfg.ONNX_MODEL_PATH,
        context_window: int = cfg.CONTEXT_WINDOW_SAMPLES,
    ) -> None:
        if ort is None:
            raise ImportError(
                "onnxruntime is required.  Install with: pip install onnxruntime"
            )

        self.context_window = context_window

        # Session options for low-latency CPU inference
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.inter_op_num_threads = max(1, os.cpu_count() - 1)
        sess_options.intra_op_num_threads = max(1, os.cpu_count() - 1)
        sess_options.enable_mem_pattern = True

        self._session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=["CPUExecutionProvider"],
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # Pre-allocated buffers
        self._input_buf = np.zeros((1, context_window), dtype=np.float32)
        self._latency_history: deque = deque(maxlen=100)

    def run(self, noisy_audio: np.ndarray) -> np.ndarray:
        """Run inference on a single audio chunk.

        Parameters
        ----------
        noisy_audio : ndarray, shape ``(context_window,)`` or ``(1, context_window)``

        Returns
        -------
        clean_audio : ndarray, shape ``(1, context_window)``
        """
        # Fill pre-allocated buffer
        if noisy_audio.ndim == 1:
            self._input_buf[0, :] = noisy_audio
        else:
            self._input_buf[:] = noisy_audio

        t0 = time.perf_counter()
        result = self._session.run(
            [self._output_name],
            {self._input_name: self._input_buf},
        )
        elapsed = time.perf_counter() - t0
        self._latency_history.append(elapsed * 1000)  # ms

        return result[0]

    def get_inference_latency_ms(self) -> float:
        """Average of last 100 inference times in milliseconds."""
        if not self._latency_history:
            return 0.0
        return sum(self._latency_history) / len(self._latency_history)

    def warmup(self, n_runs: int = 50) -> None:
        """Run dummy inference to pre-JIT any lazy compilation."""
        dummy = np.random.randn(1, self.context_window).astype(np.float32)
        for _ in range(n_runs):
            self.run(dummy)
        self._latency_history.clear()  # Reset after warmup

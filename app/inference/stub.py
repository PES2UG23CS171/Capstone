"""
Stubbed inference engine — Phase 1 passthrough.

In Phase 2 this module will load the Mamba + DeepFIR model exported to ONNX
and run INT8-quantised inference via ONNX Runtime.  For now it simply passes
audio through unchanged (or applies a trivial gain-based "strength" knob so
the GUI slider has an audible effect for integration testing).
"""

from __future__ import annotations

import numpy as np


class StubDenoiser:
    """Drop-in replacement for the future ONNXDenoiser.

    Public API contract (same as the real model):
        __init__(model_path, sample_rate, block_size)
        process(block: np.ndarray) -> np.ndarray
    """

    def __init__(
        self,
        model_path: str | None = None,
        sample_rate: int = 48_000,
        block_size: int = 1024,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._strength: float = 1.0   # 0 = bypass, 1 = full processing

    # ── public API ───────────────────────────────────────────────────────

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value: float) -> None:
        self._strength = float(np.clip(value, 0.0, 1.0))

    def process(self, block: np.ndarray) -> np.ndarray:
        """Process a single audio block.

        Parameters
        ----------
        block : np.ndarray, shape (frames,) or (frames, channels)
            Raw PCM float32 input from the microphone.

        Returns
        -------
        np.ndarray
            Denoised PCM float32 output (same shape as *block*).

        Phase 1 behaviour
        -----------------
        Pure passthrough — the audio is returned unchanged.
        When ``strength < 1.0`` the output is a crossfade between the
        original and the "denoised" version (which is also the original,
        so the effect is inaudible).  This exercises the wet/dry mix path
        that Phase 2 will rely on.
        """
        # ---- Phase 2: replace this block with ONNX inference ----
        denoised = block.copy()
        # ---------------------------------------------------------

        # Wet / dry crossfade (useful once the real model is in place)
        if self._strength < 1.0:
            return (1.0 - self._strength) * block + self._strength * denoised

        return denoised


class ONNXDenoiser:
    """Placeholder for Phase 2 — will wrap ``onnxruntime.InferenceSession``.

    Importing this class without onnxruntime installed will raise
    ``ImportError`` at construction time so it is safe to reference in
    type annotations without crashing Phase 1 code.
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 48_000,
        block_size: int = 1024,
    ) -> None:
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNXDenoiser.  "
                "Install it with:  pip install onnxruntime"
            ) from exc

        self.sample_rate = sample_rate
        self.block_size = block_size
        self._strength: float = 1.0
        # TODO: load session, allocate IO binding, warm up
        raise NotImplementedError("ONNXDenoiser is a Phase 2 target.")

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value: float) -> None:
        self._strength = float(np.clip(value, 0.0, 1.0))

    def process(self, block: np.ndarray) -> np.ndarray:
        raise NotImplementedError

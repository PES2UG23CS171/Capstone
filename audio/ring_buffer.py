"""
Ring Buffer — Layer 1
=====================
Lock-free (single-producer / single-consumer) circular buffer backed by a
NumPy float32 array.  Eliminates batch-buffering delay by providing
sample-by-sample read access to the most recent audio context.

Design Rationale
----------------
A ring buffer decouples the audio callback (producer) from the processing
thread (consumer) without requiring a mutex on the hot path.  Writes use a
``threading.Lock`` only for pointer updates; reads snapshot the write pointer
and then index into the immutable region behind it.
"""

from __future__ import annotations

import threading

import numpy as np

import config as cfg


class RingBuffer:
    """Circular buffer for real-time audio samples.

    Parameters
    ----------
    capacity : int
        Maximum number of float32 samples the buffer can hold.
        Defaults to ``SAMPLE_RATE * RING_BUFFER_SECONDS``.
    """

    def __init__(self, capacity: int | None = None) -> None:
        if capacity is None:
            capacity = int(cfg.SAMPLE_RATE * cfg.RING_BUFFER_SECONDS)

        self._capacity = capacity
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._write_pos: int = 0          # absolute write position (monotonic)
        self._lock = threading.Lock()     # guards _write_pos updates only

    # ── Write (producer) ─────────────────────────────────────────────────

    def write(self, samples: np.ndarray) -> None:
        """Append *samples* (1-D float32) to the buffer.

        Handles wraparound transparently.  If *samples* is longer than the
        capacity the oldest data is silently overwritten — by design.
        """
        samples = np.asarray(samples, dtype=np.float32).ravel()
        n = len(samples)

        with self._lock:
            start = self._write_pos % self._capacity
            end = start + n

            if end <= self._capacity:
                self._buf[start:end] = samples
            else:
                first = self._capacity - start
                self._buf[start:] = samples[:first]
                self._buf[: n - first] = samples[first:]

            self._write_pos += n

    # ── Read (consumer — lock-free) ──────────────────────────────────────

    def read(self, n: int) -> np.ndarray:
        """Return the last *n* samples without consuming them.

        If fewer than *n* samples have been written, the leading portion is
        zero-padded.
        """
        wp = self._write_pos  # snapshot (atomic int read under GIL)
        available = min(n, min(wp, self._capacity))

        out = np.zeros(n, dtype=np.float32)
        if available == 0:
            return out

        end = wp % self._capacity
        start = end - available
        if start >= 0:
            out[n - available:] = self._buf[start:end]
        else:
            # Wraparound
            first_len = -start  # samples before index 0 → end of array
            out[n - available: n - available + first_len] = self._buf[self._capacity + start:]
            out[n - available + first_len:] = self._buf[:end]

        return out

    def read_context(self, window: int | None = None) -> np.ndarray:
        """Return the most recent *window* samples (default: ``CONTEXT_WINDOW_SAMPLES``)."""
        if window is None:
            window = cfg.CONTEXT_WINDOW_SAMPLES
        return self.read(window)

    # ── Diagnostics ──────────────────────────────────────────────────────

    def get_latency_ms(self, sample_rate: int | None = None) -> float:
        """Return the current buffering delay in milliseconds.

        For a ring buffer the effective latency is one sample (the sample
        that was just written is immediately readable), so this returns
        approximately ``1 / sample_rate * 1000``.
        """
        sr = sample_rate or cfg.SAMPLE_RATE
        return 1.0 / sr * 1000.0

    @property
    def fill_level(self) -> int:
        """Number of valid samples in the buffer (capped at capacity)."""
        return min(self._write_pos, self._capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __repr__(self) -> str:
        return (
            f"RingBuffer(capacity={self._capacity}, "
            f"fill={self.fill_level}, "
            f"write_pos={self._write_pos})"
        )

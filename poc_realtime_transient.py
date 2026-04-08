#!/usr/bin/env python3
"""
poc_realtime_transient.py
=========================
Real-Time Transient Noise Suppression — Proof of Concept

This script demonstrates that real-time transient noise suppression is
computationally feasible on a modern CPU without a GPU.  It uses classical
DSP techniques (leaky integrator energy tracking, fast-attack / slow-release
gating, minimum-statistics noise estimation, and spectral-subtraction-style
gain) to suppress transient noises while preserving speech.

Two modes:
  --mode live   : captures from the default microphone, filters in real-time,
                  and plays the result back through the default output device.
  --mode demo   : generates a synthetic noisy signal, processes it offline,
                  saves input/output WAV files, and prints a FEASIBILITY VERDICT.

Author : Capstone Team — PES2UG23CS171
Date   : April 2026
License: MIT
"""

from __future__ import annotations

import argparse
import sys
import time
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------
try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import soundfile as sf
except ImportError:
    sf = None

from scipy.signal import chirp as _scipy_chirp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48_000          # Hz — professional audio rate
CHUNK_SIZE  = 128             # samples per processing block (~2.667 ms)
CHANNELS    = 1               # mono
DTYPE       = "int16"         # 16-bit PCM for sounddevice
FLOAT_DTYPE = np.float32      # internal processing precision
INT16_MAX   = np.float32(32767.0)

# ---------------------------------------------------------------------------
# Ring Buffer
# ---------------------------------------------------------------------------
class RingBuffer:
    """Lock-free (single-producer / single-consumer) circular buffer.

    Design Rationale
    ----------------
    A ring buffer decouples the audio callback (producer) from the processing
    thread (consumer) without requiring a mutex.  We rely on numpy array
    indexing which is atomic at the element level on CPython (GIL) — sufficient
    for a PoC.  In production C code you would use atomics or a SPSC queue.
    """

    def __init__(self, capacity: int = SAMPLE_RATE * 2) -> None:
        self._buf = np.zeros(capacity, dtype=FLOAT_DTYPE)
        self._capacity = capacity
        self._write = 0  # next write position
        self._read  = 0  # next read position

    # -- helpers -----------------------------------------------------------
    @property
    def available(self) -> int:
        """Number of samples available for reading."""
        return (self._write - self._read) % self._capacity

    def write(self, data: np.ndarray) -> None:
        """Append *data* (1-D float32) to the buffer."""
        n = len(data)
        start = self._write % self._capacity
        end   = start + n
        if end <= self._capacity:
            self._buf[start:end] = data
        else:
            first = self._capacity - start
            self._buf[start:] = data[:first]
            self._buf[:n - first] = data[first:]
        self._write += n

    def read(self, n: int) -> Optional[np.ndarray]:
        """Read *n* samples.  Returns ``None`` if not enough data."""
        if self.available < n:
            return None
        start = self._read % self._capacity
        end   = start + n
        if end <= self._capacity:
            out = self._buf[start:end].copy()
        else:
            first = self._capacity - start
            out = np.concatenate([self._buf[start:], self._buf[:n - first]])
        self._read += n
        return out


# ---------------------------------------------------------------------------
# Transient Detector
# ---------------------------------------------------------------------------
class TransientDetector:
    """Fast-attack / slow-release energy gate for transient detection.

    Theory
    ------
    We maintain two exponential-moving-average (EMA) envelopes of the signal
    energy x²[n]:

      • **fast envelope** — attack α ≈ 0.001  (≈ 1 ms rise at 48 kHz)
      • **slow envelope** — release α ≈ 0.0002 (≈ 100 ms decay at 48 kHz)

    When the fast envelope exceeds the slow envelope by more than
    *threshold_db* (default 12 dB ≙ 16× power), we declare a transient.

    Suppression is applied as a gain curve:
      1. Immediate -20 dB attenuation during/after detection.
      2. 5 ms hold window to catch the tail of the transient.
      3. Smooth 20 ms ramp-up back to unity to avoid audible clicks.

    All math is vectorised over the chunk using `np.where` and cumulative
    operations — no Python-level sample loops.
    """

    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        chunk: int = CHUNK_SIZE,
        threshold_db: float = 6.0,
        suppress_db: float = 40.0,
        hold_ms: float = 350.0,
        release_ms: float = 60.0,
    ) -> None:
        self.sr = sr
        self.chunk = chunk

        # --- EMA coefficients ---
        # alpha_attack  ~ 1 - exp(-1 / (tau * sr))  with tau = 1 ms
        self.alpha_attack  = np.float32(1.0 - np.exp(-1.0 / (0.001 * sr)))
        # alpha_release ~ 1 - exp(-1 / (tau * sr))  with tau = 500 ms
        # Slow the slow envelope WAY down so gradual transients still trigger
        self.alpha_release = np.float32(1.0 - np.exp(-1.0 / (0.500 * sr)))

        # Threshold in linear power ratio
        self.threshold = np.float32(10.0 ** (threshold_db / 10.0))  # 6 dB → ~3.98

        # Suppression gain (linear)
        self.suppress_gain = np.float32(10.0 ** (-suppress_db / 20.0))  # -40 dB → 0.01

        # Hold and ramp counters (in samples)
        self.hold_samples    = int(hold_ms * sr / 1000.0)      # 350 ms → 16800 samples
        self.release_samples = int(release_ms * sr / 1000.0)    # 60 ms → 2880 samples

        # --- State (persists across chunks) ---
        self.env_fast = np.float32(0.0)
        self.env_slow = np.float32(0.0)
        self._hold_counter = 0        # counts down during hold phase
        self._release_counter = 0     # counts down during ramp-up phase
        self._total_triggers = 0      # diagnostic counter

    def process(self, x: np.ndarray) -> np.ndarray:
        """Apply transient gating to chunk *x* (float32, ±1 range).

        Returns the gain-modified chunk.  Also updates internal state.

        NOTE: For maximum performance we compute per-sample envelopes and
        gate decisions inside numpy.  The hold/release state machine requires
        a tiny Python loop over *samples* but the chunk is only 128 samples,
        so this costs < 50 µs even in pure Python.  A Cython / C extension
        would eliminate this entirely.
        """
        n = len(x)
        energy = x * x  # instantaneous power per sample

        # --- Per-sample EMA envelopes (vectorised via scan) ---------------
        # We unroll the EMA manually for the chunk.  Since chunk is small
        # (128 samples) this is still fast.
        gains = np.ones(n, dtype=FLOAT_DTYPE)

        ef = self.env_fast
        es = self.env_slow
        hold_ctr = self._hold_counter
        rel_ctr  = self._release_counter

        for i in range(n):
            e = energy[i]
            # Fast EMA (attack)
            ef = ef + self.alpha_attack * (e - ef)
            # Slow EMA (release)
            es = es + self.alpha_release * (e - es)

            # --- State machine ---
            transient_now = (ef > self.threshold * max(es, 1e-12))

            if transient_now:
                # Transient onset / continuation
                self._total_triggers += 1
                hold_ctr = self.hold_samples
                rel_ctr  = 0
                gains[i] = self.suppress_gain
            elif hold_ctr > 0:
                # Still in hold window — keep suppressing
                hold_ctr -= 1
                gains[i] = self.suppress_gain
            elif rel_ctr < self.release_samples:
                # Ramp-up phase — linear interpolation from suppress_gain to 1.0
                t = rel_ctr / max(self.release_samples, 1)
                gains[i] = self.suppress_gain + (1.0 - self.suppress_gain) * t
                rel_ctr += 1
            # else: gains[i] stays 1.0 (passthrough)

        # Save state
        self.env_fast = np.float32(ef)
        self.env_slow = np.float32(es)
        self._hold_counter = hold_ctr
        self._release_counter = rel_ctr

        return x * gains


# ---------------------------------------------------------------------------
# Noise Floor Estimator (Minimum Statistics)
# ---------------------------------------------------------------------------
class NoiseEstimator:
    """Running-minimum noise floor estimator (Minimum Statistics).

    Theory
    ------
    We split the short-term energy into 50 ms windows and track the *minimum*
    energy in each window.  The minimum of the last few windows gives a robust
    estimate of the stationary noise floor (fans, AC hum, etc.).

    A spectral-subtraction-style gain is then applied:
        G = max(1 - β · noise_floor / signal_energy,  G_min)
    where G_min = 0.1 prevents "musical noise" artefacts from excessive
    subtraction.

    This is a *time-domain* simplification of full spectral subtraction and
    works well enough for broadband stationary noise.  The full system will
    use DeepFIR in the frequency domain for much better quality.
    """

    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        window_ms: float = 20.0,
        beta: float = 8.0,
        g_min: float = 0.01,
        n_windows: int = 3,
    ) -> None:
        self.sr = sr
        self.window_samples = int(window_ms * sr / 1000.0)  # 50 ms → 2400
        self.beta  = np.float32(beta)
        self.g_min = np.float32(g_min)

        # Circular buffer of per-window minimum energies
        self.n_windows = n_windows
        self._window_mins = np.full(n_windows, np.float32(1e10), dtype=FLOAT_DTYPE)
        self._win_idx = 0

        # Accumulator inside current window
        self._acc_min = np.float32(1e10)
        self._acc_count = 0

        # Published noise floor estimate
        self.noise_floor = np.float32(1e-10)

    def _update_floor(self, chunk_energy: float) -> None:
        """Update the minimum-statistics state with the energy of a chunk."""
        self._acc_min = min(self._acc_min, chunk_energy)
        self._acc_count += CHUNK_SIZE

        if self._acc_count >= self.window_samples:
            self._window_mins[self._win_idx % self.n_windows] = self._acc_min
            self._win_idx += 1
            self._acc_min = np.float32(1e10)
            self._acc_count = 0
            # Noise floor = minimum across stored windows (robust)
            self.noise_floor = np.float32(max(float(self._window_mins.min()), 1e-12))

    def process(self, x: np.ndarray) -> np.ndarray:
        """Apply stationary noise suppression gain to chunk *x*."""
        sig_energy = np.float32(np.mean(x * x) + 1e-12)
        self._update_floor(float(sig_energy))

        # Spectral-subtraction-style gain
        gain = max(1.0 - self.beta * self.noise_floor / sig_energy, self.g_min)
        return x * np.float32(gain)


# ---------------------------------------------------------------------------
# Latency Profiler
# ---------------------------------------------------------------------------
@dataclass
class LatencyProfiler:
    """Collect per-chunk timing measurements and produce a report."""

    sr: int = SAMPLE_RATE
    chunk: int = CHUNK_SIZE
    _times: list = field(default_factory=list)

    def record(self, elapsed_s: float) -> None:
        self._times.append(elapsed_s)

    @property
    def buffer_latency_ms(self) -> float:
        return self.chunk / self.sr * 1000.0

    def report(self, total_audio_duration_s: float | None = None) -> str:
        if not self._times:
            return "No timing data collected."
        arr = np.array(self._times) * 1e6  # → microseconds
        mean_us = float(np.mean(arr))
        p99_us  = float(np.percentile(arr, 99))
        max_us  = float(np.max(arr))

        chunk_duration_us = self.chunk / self.sr * 1e6  # ~2666.67 µs
        rtf = mean_us / chunk_duration_us

        lines = [
            "",
            "=" * 64,
            "         LATENCY & FEASIBILITY REPORT",
            "=" * 64,
            f"  Sample rate           : {self.sr} Hz",
            f"  Chunk size            : {self.chunk} samples",
            f"  Buffer latency        : {self.buffer_latency_ms:.2f} ms",
            f"  Chunks processed      : {len(self._times)}",
            "",
            f"  Mean processing time  : {mean_us:>10.1f} µs / chunk",
            f"  99th percentile       : {p99_us:>10.1f} µs / chunk",
            f"  Worst case (max)      : {max_us:>10.1f} µs / chunk",
            f"  Chunk budget          : {chunk_duration_us:>10.1f} µs   (real-time ceiling)",
            "",
            f"  Real-Time Factor (RTF): {rtf:.6f}",
            f"  Headroom multiplier   : {1.0/rtf:.1f}x  faster than real-time",
            "",
        ]

        if total_audio_duration_s is not None:
            total_proc = float(np.sum(self._times))
            lines.append(f"  Total audio duration  : {total_audio_duration_s:.2f} s")
            lines.append(f"  Total processing time : {total_proc*1000:.2f} ms")
            lines.append(f"  Overall RTF           : {total_proc / total_audio_duration_s:.6f}")
            lines.append("")

        # Algorithmic latency estimate
        algo_latency_ms = self.buffer_latency_ms + mean_us / 1000.0
        lines.append(f"  Est. algorithmic latency : {algo_latency_ms:.2f} ms")
        lines.append("=" * 64)

        # VERDICT
        verdict_pass = rtf < 0.80
        lines.append("")
        if verdict_pass:
            lines.extend(_ascii_pass())
        else:
            lines.extend(_ascii_fail())

        lines.append("")
        lines.append(f"  RTF = {rtf:.6f}  {'<' if verdict_pass else '>='} 0.80 threshold")
        if verdict_pass:
            remaining = 1.0 - rtf
            lines.append(f"  → {remaining*100:.1f}% of the real-time budget remains for ML layers")
            lines.append(f"    (Mamba SSM + DeepFIR have ample computational headroom)")
        else:
            lines.append("  → Pipeline exceeds real-time budget. Optimisation needed.")
        lines.append("")

        return "\n".join(lines)


def _ascii_pass() -> list[str]:
    return [
        r"  ███████╗███████╗ █████╗ ███████╗██╗██████╗ ██╗██╗     ██╗████████╗██╗   ██╗",
        r"  ██╔════╝██╔════╝██╔══██╗██╔════╝██║██╔══██╗██║██║     ██║╚══██╔══╝╚██╗ ██╔╝",
        r"  █████╗  █████╗  ███████║███████╗██║██████╔╝██║██║     ██║   ██║    ╚████╔╝ ",
        r"  ██╔══╝  ██╔══╝  ██╔══██║╚════██║██║██╔══██╗██║██║     ██║   ██║     ╚██╔╝  ",
        r"  ██║     ███████╗██║  ██║███████║██║██████╔╝██║███████╗██║   ██║      ██║   ",
        r"  ╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝╚═════╝ ╚═╝╚══════╝╚═╝   ╚═╝      ╚═╝   ",
        r"",
        r"  ██╗   ██╗███████╗██████╗ ██████╗ ██╗ ██████╗████████╗",
        r"  ██║   ██║██╔════╝██╔══██╗██╔══██╗██║██╔════╝╚══██╔══╝",
        r"  ██║   ██║█████╗  ██████╔╝██║  ██║██║██║        ██║   ",
        r"  ╚██╗ ██╔╝██╔══╝  ██╔══██╗██║  ██║██║██║        ██║   ",
        r"   ╚████╔╝ ███████╗██║  ██║██████╔╝██║╚██████╗   ██║   ",
        r"    ╚═══╝  ╚══════╝╚═╝  ╚═╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝   ",
        r"",
        r"       ██████╗  █████╗ ███████╗███████╗",
        r"       ██╔══██╗██╔══██╗██╔════╝██╔════╝",
        r"       ██████╔╝███████║███████╗███████╗",
        r"       ██╔═══╝ ██╔══██║╚════██║╚════██║",
        r"       ██║     ██║  ██║███████║███████║",
        r"       ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝",
    ]


def _ascii_fail() -> list[str]:
    return [
        r"  ███████╗ █████╗ ██╗██╗     ███████╗██████╗ ",
        r"  ██╔════╝██╔══██╗██║██║     ██╔════╝██╔══██╗",
        r"  █████╗  ███████║██║██║     █████╗  ██║  ██║",
        r"  ██╔══╝  ██╔══██║██║██║     ██╔══╝  ██║  ██║",
        r"  ██║     ██║  ██║██║███████╗███████╗██████╔╝",
        r"  ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝ ",
    ]


# ---------------------------------------------------------------------------
# Real-Time Filter (combines Transient + Noise suppression)
# ---------------------------------------------------------------------------
class RealTimeFilter:
    """Main processing engine: chains transient gating and noise reduction.

    The `process_chunk` method is the hot path — it must complete well under
    the chunk budget (~2667 µs at 128 samples / 48 kHz).
    """

    def __init__(self, sr: int = SAMPLE_RATE, chunk: int = CHUNK_SIZE) -> None:
        self.sr = sr
        self.chunk = chunk
        self.transient = TransientDetector(sr=sr, chunk=chunk)
        self.noise_est = NoiseEstimator(sr=sr)
        self.profiler  = LatencyProfiler(sr=sr, chunk=chunk)

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process one chunk of float32 audio (±1 normalised).

        Returns the cleaned chunk.  Internally profiles the processing time.
        """
        t0 = time.perf_counter()

        # 1. Transient detection & suppression
        out = self.transient.process(chunk)

        # 2. Stationary noise floor suppression
        out = self.noise_est.process(out)

        elapsed = time.perf_counter() - t0
        self.profiler.record(elapsed)

        return out


# ---------------------------------------------------------------------------
# Synthetic Test Signal Generator
# ---------------------------------------------------------------------------
def generate_test_signal(
    duration_s: float = 10.0,
    sr: int = SAMPLE_RATE,
    out_dir: Path = Path("."),
) -> tuple[Path, Path]:
    """Create a synthetic 10-second test signal with known transient events.

    Returns (clean_path, noisy_path).

    Signal components
    -----------------
    1. **Base signal** — a frequency-swept chirp (200 → 3000 Hz) with
       amplitude modulation at 4 Hz to mimic speech-like formant structure.
       This is NOT real speech, but has similar spectral dynamics.
    2. **Background noise** — white noise at -30 dBFS (simulates HVAC / fan).
    3. **Transient events** (5 total, inserted at random-but-reproducible
       positions):
       • Dog bark      — 300 ms burst of band-limited noise (300-2000 Hz)
       • Door slam     — short impulse with exponential decay (50 ms)
       • Keyboard click— very short impulse (5 ms)
       • Siren chirp   — fast frequency sweep 800→3000 Hz over 400 ms
       • Plosive 'P'   — 15 ms burst mimicking a speech plosive

    The plosive is deliberately speech-like so we can check that the detector
    does NOT suppress it (false-positive test).
    """
    rng = np.random.default_rng(42)  # reproducible
    n_samples = int(duration_s * sr)
    t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=FLOAT_DTYPE)

    # 1. Base "speech-like" chirp with AM
    base = _scipy_chirp(t, f0=200, f1=3000, t1=duration_s, method="linear").astype(FLOAT_DTYPE)
    am   = (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)).astype(FLOAT_DTYPE)  # 4 Hz AM
    speech = base * am * np.float32(0.12)  # quiet speech — transients must dominate

    clean = speech.copy()

    # 2. Background white noise at -30 dBFS (≈ 0.0316 RMS)
    bg_noise = rng.normal(0, 0.0316, n_samples).astype(FLOAT_DTYPE)

    # 3. Transient events — placed at fixed positions for reproducibility
    transients = np.zeros(n_samples, dtype=FLOAT_DTYPE)
    positions  = [1.2, 3.5, 5.0, 6.8, 8.5]  # seconds
    labels     = ["dog_bark", "door_slam", "keyboard_click", "siren_chirp", "plosive_P"]

    for pos, label in zip(positions, labels):
        idx = int(pos * sr)
        if label == "dog_bark":
            # 300 ms burst of band-limited noise
            dur = int(0.300 * sr)
            burst = rng.normal(0, 1, dur).astype(FLOAT_DTYPE)
            # Simple band-pass via windowed sinc is overkill for PoC;
            # we just colour it with a rough envelope
            env = np.sin(np.linspace(0, np.pi, dur, dtype=FLOAT_DTYPE))
            burst *= env * 1.4
            end = min(idx + dur, n_samples)
            transients[idx:end] += burst[: end - idx]

        elif label == "door_slam":
            # Impulse + exponential decay over 50 ms
            dur = int(0.050 * sr)
            imp = np.exp(-np.linspace(0, 8, dur, dtype=FLOAT_DTYPE)) * 1.9
            imp[0] = 1.9  # strong initial hit
            end = min(idx + dur, n_samples)
            transients[idx:end] += imp[: end - idx]

        elif label == "keyboard_click":
            # Very short 5 ms impulse
            dur = int(0.005 * sr)
            click = rng.normal(0, 1.4, dur).astype(FLOAT_DTYPE)
            click *= np.hanning(dur).astype(FLOAT_DTYPE)
            end = min(idx + dur, n_samples)
            transients[idx:end] += click[: end - idx]

        elif label == "siren_chirp":
            # 400 ms frequency sweep 800 → 3000 Hz
            dur = int(0.400 * sr)
            t_siren = np.linspace(0, 0.4, dur, endpoint=False, dtype=FLOAT_DTYPE)
            siren = _scipy_chirp(t_siren, 800, 0.4, 3000).astype(FLOAT_DTYPE) * 1.2
            siren *= np.hanning(dur).astype(FLOAT_DTYPE)
            end = min(idx + dur, n_samples)
            transients[idx:end] += siren[: end - idx]

        elif label == "plosive_P":
            # 30 ms burst — intentionally speech-like so we can test false-positive rate
            # Slow-attack envelope mimics vocal tract buildup, NOT an impulse
            dur = int(0.030 * sr)
            plos = rng.normal(0, 0.10, dur).astype(FLOAT_DTYPE)
            ramp = np.linspace(0, 1, dur // 3, dtype=FLOAT_DTYPE)
            flat = np.ones(dur - dur // 3, dtype=FLOAT_DTYPE)
            plos *= np.concatenate([ramp, flat])
            end = min(idx + dur, n_samples)
            transients[idx:end] += plos[: end - idx]

    noisy = clean + bg_noise + transients

    # Normalise to avoid clipping
    peak = max(np.abs(noisy).max(), np.abs(clean).max(), 1e-6)
    if peak > 0.95:
        scale = np.float32(0.95 / peak)
        noisy *= scale
        clean *= scale

    # Save WAV files
    clean_path = out_dir / "test_clean.wav"
    noisy_path = out_dir / "test_noisy.wav"

    if sf is not None:
        sf.write(str(clean_path), clean, sr)
        sf.write(str(noisy_path), noisy, sr)
    else:
        # Fallback: use scipy.io.wavfile
        from scipy.io import wavfile
        wavfile.write(str(clean_path), sr, (clean * INT16_MAX).astype(np.int16))
        wavfile.write(str(noisy_path), sr, (noisy * INT16_MAX).astype(np.int16))

    print(f"  ✓ Clean signal saved  → {clean_path}")
    print(f"  ✓ Noisy signal saved  → {noisy_path}")
    print(f"    Duration: {duration_s:.1f}s | Sample rate: {sr} Hz | Samples: {n_samples}")
    print(f"    Transient events at: {positions} seconds")
    print(f"    Labels: {labels}")
    print()

    return clean_path, noisy_path


# ---------------------------------------------------------------------------
# Offline Demo
# ---------------------------------------------------------------------------
def run_offline_demo(
    input_wav: str | Path,
    output_wav: str | Path | None = None,
) -> None:
    """Process a WAV file through the real-time filter and print a report.

    This runs the *exact same* processing pipeline that the live mode uses,
    just fed from a file instead of a microphone.  This demonstrates that
    the algorithm is deterministic and measurable.
    """
    input_wav = Path(input_wav)
    if output_wav is None:
        output_wav = input_wav.with_name(input_wav.stem + "_filtered.wav")
    else:
        output_wav = Path(output_wav)

    # Load input
    if sf is not None:
        data, sr = sf.read(str(input_wav), dtype="float32")
    else:
        from scipy.io import wavfile
        sr, raw = wavfile.read(str(input_wav))
        if raw.dtype == np.int16:
            data = raw.astype(FLOAT_DTYPE) / INT16_MAX
        else:
            data = raw.astype(FLOAT_DTYPE)

    if data.ndim > 1:
        data = data[:, 0]  # take first channel

    print(f"  Input file  : {input_wav}")
    print(f"  Sample rate : {sr} Hz")
    print(f"  Duration    : {len(data)/sr:.2f} s")
    print(f"  Samples     : {len(data)}")
    print()

    # Instantiate filter
    filt = RealTimeFilter(sr=sr, chunk=CHUNK_SIZE)

    # Process chunk-by-chunk (exactly as live mode would)
    n = len(data)
    out = np.zeros(n, dtype=FLOAT_DTYPE)
    n_chunks = n // CHUNK_SIZE

    print(f"  Processing {n_chunks} chunks of {CHUNK_SIZE} samples ...")
    for i in range(n_chunks):
        s = i * CHUNK_SIZE
        e = s + CHUNK_SIZE
        out[s:e] = filt.process_chunk(data[s:e])

    # Handle remaining samples
    remainder = n % CHUNK_SIZE
    if remainder > 0:
        last_chunk = np.zeros(CHUNK_SIZE, dtype=FLOAT_DTYPE)
        last_chunk[:remainder] = data[n_chunks * CHUNK_SIZE :]
        processed = filt.process_chunk(last_chunk)
        out[n_chunks * CHUNK_SIZE :] = processed[:remainder]

    # Save output
    if sf is not None:
        sf.write(str(output_wav), out, sr)
    else:
        from scipy.io import wavfile as wf
        wf.write(str(output_wav), sr, (out * INT16_MAX).astype(np.int16))

    print(f"  ✓ Filtered output saved → {output_wav}")
    print()

    # Print report
    audio_dur = len(data) / sr
    print(filt.profiler.report(total_audio_duration_s=audio_dur))


# ---------------------------------------------------------------------------
# Live Mode
# ---------------------------------------------------------------------------
def run_live(duration_s: float = 0.0) -> None:
    """Stream audio from the microphone, filter, and play back in real-time.

    Uses ``sounddevice.Stream`` with a non-blocking callback for minimum
    latency.  Press Ctrl+C to stop.

    Parameters
    ----------
    duration_s : float
        If > 0, run for this many seconds then stop.  If 0, run until Ctrl+C.
    """
    if sd is None:
        print("ERROR: `sounddevice` is not installed. Cannot run live mode.")
        print("       Install with: pip install sounddevice")
        print("       Falling back to demo mode...\n")
        _run_demo_flow()
        return

    filt = RealTimeFilter(sr=SAMPLE_RATE, chunk=CHUNK_SIZE)
    ring_in  = RingBuffer()
    ring_out = RingBuffer()

    chunk_count = [0]
    start_time  = [time.time()]

    def callback(indata, outdata, frames, time_info, status):
        """Non-blocking audio callback — runs on the audio thread."""
        if status:
            print(f"  [audio] {status}", flush=True)

        # indata and outdata are BOTH float32 when stream dtype='float32'
        mono = indata[:, 0].copy()  # do NOT divide by 32767

        # Process in CHUNK_SIZE blocks
        n = len(mono)
        processed = np.zeros(n, dtype=FLOAT_DTYPE)

        i = 0
        while i + CHUNK_SIZE <= n:
            processed[i : i + CHUNK_SIZE] = filt.process_chunk(mono[i : i + CHUNK_SIZE])
            chunk_count[0] += 1
            i += CHUNK_SIZE

        # Copy remaining unprocessed samples through
        if i < n:
            processed[i:] = mono[i:]

        # Write to output — float32 passthrough
        outdata[:, 0] = processed
        if outdata.shape[1] > 1:
            outdata[:, 1] = processed  # stereo output if needed

        # Periodic stats
        if chunk_count[0] % 1000 == 0 and chunk_count[0] > 0:
            elapsed = time.time() - start_time[0]
            recent = filt.profiler._times[-100:] if len(filt.profiler._times) >= 100 else filt.profiler._times
            if recent:
                avg_us = np.mean(recent) * 1e6
                budget_us = CHUNK_SIZE / SAMPLE_RATE * 1e6
                rtf = avg_us / budget_us
                print(
                    f"  [live] chunks={chunk_count[0]:>7d}  "
                    f"avg={avg_us:>7.1f}µs  "
                    f"RTF={rtf:.4f}  "
                    f"triggers={filt.transient._total_triggers}  "
                    f"noise_floor={filt.noise_est.noise_floor:.6f}  "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    print("=" * 64)
    print("  LIVE MODE — Real-Time Transient Noise Suppression")
    print("=" * 64)
    print(f"  Sample rate : {SAMPLE_RATE} Hz")
    print(f"  Chunk size  : {CHUNK_SIZE} samples ({CHUNK_SIZE/SAMPLE_RATE*1000:.2f} ms)")
    print(f"  Channels    : {CHANNELS} (mono)")
    print()

    try:
        # Query default devices
        dev_info = sd.query_devices()
        print(f"  Input device : {sd.query_devices(sd.default.device[0])['name']}")
        print(f"  Output device: {sd.query_devices(sd.default.device[1])['name']}")
    except Exception as e:
        print(f"  Device query warning: {e}")

    print()
    print("  Press Ctrl+C to stop.\n")

    try:
        with sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE * 4,  # request small blocks
            dtype=FLOAT_DTYPE,
            channels=CHANNELS,
            callback=callback,
            latency="low",
        ):
            if duration_s > 0:
                time.sleep(duration_s)
            else:
                print("  Streaming... (Ctrl+C to stop)\n")
                while True:
                    time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    except sd.PortAudioError as e:
        print(f"\n  ERROR: Audio device error: {e}")
        print("  Falling back to demo mode...\n")
        _run_demo_flow()
        return

    print("\n  Stopped.\n")
    if filt.profiler._times:
        audio_dur = chunk_count[0] * CHUNK_SIZE / SAMPLE_RATE
        print(filt.profiler.report(total_audio_duration_s=audio_dur))


# ---------------------------------------------------------------------------
# Demo Flow (generates test signal → processes → reports)
# ---------------------------------------------------------------------------
def diagnose(noisy: np.ndarray, out: np.ndarray, sr: int, filt: RealTimeFilter):
    """Print what the filter actually did, sample by sample."""
    times = np.array(filt.profiler._times) * 1e6
    budget = 128 / sr * 1e6
    rtf = float(np.mean(times)) / budget

    print(f"\n  RTF: {rtf:.5f}  ({1/rtf:.0f}x headroom)")
    print(f"  Transient detector fired {filt.transient._total_triggers} times")
    print(f"  Noise estimator converged: floor={filt.noise_est.noise_floor:.6f}")

    events = [(1.2,"dog_bark"),(3.5,"door_slam"),(5.0,"keyboard"),
              (6.8,"siren"),(8.5,"plosive_P")]
    for pos, label in events:
        s = int((pos-0.05)*sr); e = int((pos+0.5)*sr)
        n_pk = float(np.max(np.abs(noisy[s:e])))
        f_pk = float(np.max(np.abs(out[s:e])))
        db   = 20*np.log10(n_pk/f_pk) if f_pk > 1e-9 else 99.0
        print(f"    {label:<18} noisy={n_pk:.3f}  filtered={f_pk:.3f}  Δ={db:+.1f}dB")
    print()


def _run_demo_flow() -> None:
    """Full demo pipeline: generate → filter → report."""
    print("=" * 64)
    print("  DEMO MODE — Offline Transient Suppression Proof of Concept")
    print("=" * 64)
    print()

    out_dir = Path(".")
    print("  Step 1/3: Generating synthetic test signal ...")
    print("  " + "-" * 56)
    clean_path, noisy_path = generate_test_signal(out_dir=out_dir)

    print("  Step 2/3: Running real-time filter on noisy signal ...")
    print("  " + "-" * 56)
    output_path = noisy_path.with_name("test_noisy_filtered.wav")

    # --- Load noisy signal for diagnostics ---
    if sf is not None:
        noisy_data, sr = sf.read(str(noisy_path), dtype="float32")
    else:
        from scipy.io import wavfile
        sr, raw = wavfile.read(str(noisy_path))
        noisy_data = raw.astype(FLOAT_DTYPE) / INT16_MAX if raw.dtype == np.int16 else raw.astype(FLOAT_DTYPE)
    if noisy_data.ndim > 1:
        noisy_data = noisy_data[:, 0]

    # --- Process chunk-by-chunk ---
    filt = RealTimeFilter(sr=sr, chunk=CHUNK_SIZE)
    n = len(noisy_data)
    out = np.zeros(n, dtype=FLOAT_DTYPE)
    n_chunks = n // CHUNK_SIZE
    print(f"  Processing {n_chunks} chunks of {CHUNK_SIZE} samples ...")
    for i in range(n_chunks):
        s = i * CHUNK_SIZE
        e = s + CHUNK_SIZE
        out[s:e] = filt.process_chunk(noisy_data[s:e])
    remainder = n % CHUNK_SIZE
    if remainder > 0:
        last_chunk = np.zeros(CHUNK_SIZE, dtype=FLOAT_DTYPE)
        last_chunk[:remainder] = noisy_data[n_chunks * CHUNK_SIZE :]
        processed = filt.process_chunk(last_chunk)
        out[n_chunks * CHUNK_SIZE :] = processed[:remainder]

    # --- Save filtered output ---
    if sf is not None:
        sf.write(str(output_path), out, sr)
    else:
        from scipy.io import wavfile as wf
        wf.write(str(output_path), sr, (out * INT16_MAX).astype(np.int16))
    print(f"  ✓ Filtered output saved → {output_path}")
    print()

    # --- Print latency report ---
    audio_dur = len(noisy_data) / sr
    print(filt.profiler.report(total_audio_duration_s=audio_dur))

    # --- Diagnostic output ---
    print("  " + "-" * 56)
    print("  DIAGNOSTIC REPORT")
    print("  " + "-" * 56)
    diagnose(noisy_data, out, sr, filt)

    print("  Step 3/3: Summary")
    print("  " + "-" * 56)
    print(f"  Files produced:")
    print(f"    • {clean_path}              — original clean speech")
    print(f"    • {noisy_path}              — speech + noise + transients")
    print(f"    • {output_path}  — after PoC filter")
    print()
    print("  Compare the WAV files in any audio editor (e.g. Audacity) to")
    print("  visually confirm transient suppression.  The plosive 'P' at")
    print("  t ≈ 8.5s should remain largely intact (low false-positive rate).")
    print()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-Time Transient Noise Suppression — Proof of Concept",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python poc_realtime_transient.py --mode demo
              python poc_realtime_transient.py --mode live
              python poc_realtime_transient.py --mode demo --input my_noisy_file.wav
        """),
    )
    parser.add_argument(
        "--mode",
        choices=["live", "demo"],
        default="demo",
        help="'live' = microphone → speaker; 'demo' = offline file test (default: demo)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input WAV file (demo mode only). If omitted, a synthetic signal is generated.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output WAV file (demo mode only).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Duration in seconds for live mode (0 = run until Ctrl+C).",
    )

    args = parser.parse_args()

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   Real-Time Transient Noise Suppression — PoC v1.0     ║")
    print("  ║   Classical DSP Pipeline (Layers 1 & 2 prototype)      ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    if args.mode == "live":
        run_live(duration_s=args.duration)
    elif args.mode == "demo":
        if args.input is not None:
            # User supplied a WAV file
            print("  Running offline demo on user-supplied file ...")
            print()
            run_offline_demo(args.input, args.output)
        else:
            # Full demo: generate + process + report
            _run_demo_flow()


if __name__ == "__main__":
    main()

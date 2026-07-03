"""
Acoustic Echo Cancellation (AEC) — V3.

The chain (mic + reference → cleaned near-end):

  1. **GCC-PHAT one-shot delay calibration.**  Real loopback paths on
     macOS/Windows have 10–80 ms of latency between what the reference
     signal "says" is playing and what the mic actually captures.  A
     generalised cross-correlation with phase transform finds that delay
     once at session start; we apply it as a delay line on the reference
     before feeding it into the linear AEC.

       * ``argmax(|cc|)`` (magnitude) — handles polarity-inverted loopback.
       * Sliding-window calibration buffer (cap = ``_calib_samples_target``)
         — no unbounded growth when the far-end is silent for a long time.
       * Weak-peak retry — a single low-coherence window does not
         permanently lock in ``delay = 0``; calibration stays armed until a
         confident peak is found or a 10-second watchdog times out.

  2. **Partitioned-block FDAF NLMS.**  Same as V2; Geigel DTD with
     ~1 s freeze on detect; mu = 0.15 for stability under bursty far-end.

  3. **NLES (nonlinear echo suppression, per-bin Wiener)** on the linear
     residual.  Critically: compares the residual ``E`` against the
     **estimated echo at the mic** ``Y`` (= ``W·X`` from the linear filter
     output) — NOT the raw line-level reference ``X``.  Using the raw ``X``
     biases the ratio upward by the echo-path attenuation (~20–60 dB),
     pinning the gain to its floor on every far-end frame.

         G(k) = clip(1 − β · |Y(k)|² / |E(k)|², G_min, 1)

  4. **50%-overlap OLA reconstruction** for the NLES output.  V3-α used a
     ``y + (1−win)·e`` reconstruction that passed unsuppressed residual
     through the block edges, producing a periodic block-rate artifact at
     strong attenuation.  The Hann² overlap-add gives constant-gain
     reconstruction across block boundaries.

  5. **Proper magnitude-squared coherence** between residual ``E`` and
     reference ``X`` via Welch-style averaging across recent blocks.  V3-α
     used per-bin ``|E|·|X|/(sqrt(|E|²)·sqrt(|X|²))`` which is identically
     1.0 by construction and made ``res_band`` always saturate.

  6. **Fused ``echo_conf``** ∈ [0, 1] from DT state, RES coherence, ERLE,
     and NLES attenuation.  ERLE is initialised at the "converged" end of
     the score (so the EWMA doesn't pin echo_conf to 1.0 immediately after
     calibration completes).  Surfaced on ``ctx.echo_conf``.

Note: DFN3 is treated as a black box.  All non-linear suppression happens
in this file, around the existing enhancement model.
"""

from __future__ import annotations

import numpy as np
from collections import deque

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext, Stage


class _GeigelDTD:
    """Geigel double-talk detector — declares DT when |d| > T · max(|x[-D:]|)."""

    def __init__(self, sr: int, history_ms: float = 100.0, threshold: float = 0.5) -> None:
        self.D = int(sr * history_ms / 1000.0)
        self.T = threshold
        self._x_hist = np.zeros(self.D, dtype=np.float64)

    def reset(self) -> None:
        self._x_hist[:] = 0.0

    def __call__(self, d: np.ndarray, x: np.ndarray) -> bool:
        n = len(x)
        if n >= self.D:
            self._x_hist = x[-self.D:].astype(np.float64).copy()
        else:
            self._x_hist = np.roll(self._x_hist, -n)
            self._x_hist[-n:] = x.astype(np.float64)
        max_d = float(np.max(np.abs(d))) if len(d) else 0.0
        max_x = float(np.max(np.abs(self._x_hist))) + 1e-9
        return max_d > self.T * max_x


class _CoherenceEstimator:
    """Welch-style magnitude-squared coherence per FFT bin, averaged across
    recent blocks.  Splits the bins into 3 bands at the end.

    MSC(k) = |E[X·conj(Y)]|² / (E[|X|²] · E[|Y|²])

    The per-bin expectations are EWMAs over recent blocks (α=0.2 ≈ 5-block
    half-life).  Coherence is bounded ∈ [0, 1] and is genuinely 1.0 only
    when X and Y are linearly related — which is exactly the property we
    want from a residual-echo cue.
    """

    def __init__(self, n_bins: int, alpha: float = 0.2) -> None:
        self._alpha = alpha
        # Per-bin running estimates of cross- and auto-spectra.
        self._sxy = np.zeros(n_bins, dtype=np.complex128)
        self._sxx = np.zeros(n_bins, dtype=np.float64)
        self._syy = np.zeros(n_bins, dtype=np.float64)
        self._initialized = False

    def reset(self) -> None:
        self._sxy[:] = 0.0
        self._sxx[:] = 0.0
        self._syy[:] = 0.0
        self._initialized = False

    def update(self, E: np.ndarray, X: np.ndarray) -> tuple[float, float, float]:
        """Update running estimates with this block's E and X, return per-band MSC."""
        a = self._alpha
        if not self._initialized:
            self._sxy = E * np.conj(X)
            self._sxx = np.abs(E) ** 2
            self._syy = np.abs(X) ** 2
            self._initialized = True
        else:
            self._sxy = (1.0 - a) * self._sxy + a * (E * np.conj(X))
            self._sxx = (1.0 - a) * self._sxx + a * (np.abs(E) ** 2)
            self._syy = (1.0 - a) * self._syy + a * (np.abs(X) ** 2)
        denom = (self._sxx * self._syy) + 1e-12
        msc = (np.abs(self._sxy) ** 2) / denom
        msc = np.clip(msc, 0.0, 1.0)
        nb = len(msc) // 3
        if nb == 0:
            return 0.0, 0.0, 0.0
        return (
            float(np.mean(msc[:nb])),
            float(np.mean(msc[nb:2 * nb])),
            float(np.mean(msc[2 * nb:])),
        )


def _gcc_phat_delay(near: np.ndarray, ref: np.ndarray,
                    max_lag_samples: int, peak_threshold: float) -> tuple[int, float]:
    """One-shot GCC-PHAT delay estimator.

    Returns ``(delay_samples, normalised_peak)``.  ``delay_samples`` is
    positive when ``ref`` precedes ``near`` (typical loopback case).

    Uses ``argmax(|cc|)`` so polarity-inverted loopbacks are handled
    correctly: some sound-card / Bluetooth-codec combinations invert one
    channel's polarity, producing a negative correlation peak that
    ``argmax(cc)`` would miss.

    PHAT (phase transform) whitens the cross-spectrum so the result is
    independent of either signal's spectral colour — robust to coloured
    loopbacks and to broadband noise on the mic.
    """
    n = min(len(near), len(ref))
    if n < 256:
        return 0, 0.0
    near = near[:n].astype(np.float64) - float(np.mean(near[:n]))
    ref = ref[:n].astype(np.float64) - float(np.mean(ref[:n]))
    fft_n = 1 << int(np.ceil(np.log2(2 * n)))
    N = np.fft.rfft(near, fft_n)
    R = np.fft.rfft(ref, fft_n)
    cross = N * np.conj(R)
    cross /= (np.abs(cross) + 1e-9)
    cc = np.fft.irfft(cross, fft_n)
    if max_lag_samples >= len(cc):
        max_lag_samples = len(cc) - 1
    window = cc[: max_lag_samples + 1]
    # Take the largest-MAGNITUDE peak (handles polarity-inverted loopback).
    peak_idx = int(np.argmax(np.abs(window)))
    peak_val = abs(float(window[peak_idx]))
    norm = float(np.max(np.abs(cc))) + 1e-9
    peak_norm = peak_val / norm
    if peak_norm < peak_threshold:
        return 0, peak_norm
    return peak_idx, peak_norm


class _NlesOlaProcessor:
    """50%-overlap-add NLES processor.

    Reasoning: a non-overlapping window with the V3-α ``y + (1-win)·e``
    reconstruction passed unsuppressed residual through the block edges at
    strong attenuation, producing a periodic block-rate artifact.  Proper
    Hann² OLA gives constant-gain reconstruction.

    Hann² overlap-add: with analysis Hann window ``w`` and synthesis Hann
    window ``w``, frames hopped by ``N/2`` sum to ``w² + w²(shifted) = 1``
    by the Hann² constant-overlap-add identity.

    Block convention: caller hands us N-sample residual blocks.  We hold a
    half-block lookback so we can compute one frame per half-block and OLA
    them.  Each call produces N samples of output corresponding to the
    PRIOR N samples of input (one half-block of additional algorithmic
    delay, ~5–10 ms at typical AEC block sizes).
    """

    def __init__(self, N: int) -> None:
        self.N = N
        self._half = N // 2
        # Hann analysis only.  At 50% overlap, Hann[k] + Hann[k + N/2] = 1.0
        # for all k ∈ [0, N/2) — perfect constant-overlap-add.  Synthesis is
        # rectangular (no window applied to irfft output).
        self._win = np.hanning(N).astype(np.float64)
        # State carried across calls.  Per call we compute two frames:
        #   * straddle frame:  prev_block[h:N] ++ this_block[:h]   = positions [(i)·N − h, i·N + h − 1]
        #   * this-block frame:                  this_block         = positions [i·N, (i+1)·N − 1]
        # Output emitted: the PREVIOUS block, fully OLA-reconstructed.  This
        # introduces a one-block (~10 ms at default block size) algorithmic
        # delay, which is the unavoidable cost of correct 50%-OLA at 0
        # lookahead.
        self._prev_e = np.zeros(self._half, dtype=np.float64)
        self._prev_y = np.zeros(self._half, dtype=np.float64)
        # OLA contributions kept until the next call can complete them.
        # ``_pending_first_half``: the first h samples of the next output
        # block (assembled from frame_straddle_PREV[h:] + frame_this_PREV[:h]).
        # ``_pending_this_second_half``: frame_this_PREV[h:], used to complete
        # the second half of the next output block.
        self._pending_first_half = np.zeros(self._half, dtype=np.float64)
        self._pending_this_second_half = np.zeros(self._half, dtype=np.float64)
        # First-call guard — until the first call has run, there is no
        # previous block to emit, so we emit zeros (one block of warmup).
        self._initialised = False
        self._last_atten_db = 0.0

    def reset(self) -> None:
        self._prev_e[:] = 0.0
        self._prev_y[:] = 0.0
        self._pending_first_half[:] = 0.0
        self._pending_this_second_half[:] = 0.0
        self._initialised = False
        self._last_atten_db = 0.0

    def _process_frame(self, e_frame: np.ndarray, y_frame: np.ndarray,
                        beta: float, g_min_lin: float) -> tuple[np.ndarray, float]:
        """One N-sample Hann-windowed frame → spectrally-masked time output."""
        win = self._win
        E = np.fft.rfft(e_frame * win)
        Y = np.fft.rfft(y_frame * win)
        e_power = np.abs(E) ** 2 + 1e-12
        y_power = np.abs(Y) ** 2
        # Per-bin Wiener — y_power = ESTIMATED ECHO AT THE MIC (|W·X|²),
        # not the raw line-level reference power.
        gain = np.clip(1.0 - beta * (y_power / e_power), g_min_lin, 1.0)
        y_out = np.fft.irfft(E * gain, n=self.N)
        avg_atten_db = float(np.mean(20.0 * np.log10(gain + 1e-9)))
        return y_out, -avg_atten_db  # positive = attenuation amount

    def process(self, e_block: np.ndarray, y_block: np.ndarray,
                 beta: float, g_min_lin: float) -> tuple[np.ndarray, float]:
        """Emit the PREVIOUS block's NLES output as a clean 50%-OLA Hann
        reconstruction.  Per-call delay = one N-sample block."""
        N = self.N
        h = self._half
        e_block = e_block.astype(np.float64)
        y_block = y_block.astype(np.float64)

        # Compute the two frames anchored in THIS call's input:
        #   straddle frame anchored at absolute position i·N − h
        #     = [prev_block_second_half, this_block_first_half]
        #   this-block frame anchored at absolute position i·N
        #     = this_block
        e_straddle = np.concatenate([self._prev_e, e_block[:h]])
        y_straddle = np.concatenate([self._prev_y, y_block[:h]])
        frame_straddle, atten_s = self._process_frame(
            e_straddle, y_straddle, beta, g_min_lin
        )
        frame_this, atten_t = self._process_frame(
            e_block, y_block, beta, g_min_lin
        )
        self._last_atten_db = 0.5 * (atten_s + atten_t)

        # Emit output for the PREVIOUS block:
        #   first half  = _pending_first_half (set in the previous call)
        #   second half = _pending_this_second_half + frame_straddle[:h]
        # Both halves sum exactly to e_prev_block[k] at unity gain by Hann COLA.
        if not self._initialised:
            out_block = np.zeros(N, dtype=np.float64)
            self._initialised = True
        else:
            out_block = np.empty(N, dtype=np.float64)
            out_block[:h] = self._pending_first_half
            out_block[h:] = self._pending_this_second_half + frame_straddle[:h]

        # Stage the contributions to the NEXT emitted block:
        #   _pending_first_half = frame_straddle[h:] + frame_this[:h]
        #   _pending_this_second_half = frame_this[h:]
        self._pending_first_half = frame_straddle[h:] + frame_this[:h]
        self._pending_this_second_half = frame_this[h:].copy()

        # Slide the half-block buffers (input history for next call's straddle).
        self._prev_e = e_block[-h:].copy()
        self._prev_y = y_block[-h:].copy()
        return out_block, self._last_atten_db


def _echo_confidence(dt_active: bool, res_band: tuple[float, float, float],
                     erle_db: float, nles_atten_db: float, converged: bool) -> float:
    """Fuse {DT, RES per-band coherence, ERLE, NLES attenuation} → echo_conf.

    ``converged`` is False during the first few seconds after calibration
    (or at session start) while ERLE has not yet had time to converge.
    During that window we suppress the ERLE cue so a stale 0-dB ERLE
    doesn't spuriously pin echo_conf to 1.0.
    """
    dt_score = 1.0 if dt_active else 0.0
    res_score = max(res_band) if res_band else 0.0
    if converged:
        erle_score = float(np.clip((12.0 - erle_db) / 9.0, 0.0, 1.0))
    else:
        erle_score = 0.0
    nles_score = float(np.clip(nles_atten_db / 20.0, 0.0, 1.0))
    return float(np.clip(max(dt_score, res_score, erle_score, nles_score), 0.0, 1.0))


class AEC(Stage):
    name = "aec"

    # GCC-PHAT calibration parameters.
    _CALIB_WINDOW_S = 0.5         # sliding-window target size
    _MIN_REF_RMS = 5e-4
    _CALIB_TIMEOUT_S = 10.0       # if no confident peak by then, accept delay=0
    # ERLE convergence — number of NLMS update blocks before erle_score is trusted.
    _ERLE_CONVERGE_BLOCKS = 40

    def __init__(self, cfg: PipelineConfig, block: int = 512,
                 mu: float | None = None) -> None:
        self.cfg = cfg
        self.enabled = cfg.aec_enabled
        self.N = block
        self.fft = 2 * block
        self.mu_base = float(cfg.aec_step_size) if mu is None else float(mu)
        tail = int(cfg.sample_rate * cfg.aec_filter_ms / 1000.0)
        self.P = max(1, int(np.ceil(tail / block)))

        self._W = np.zeros((self.P, self.fft // 2 + 1), dtype=np.complex128)
        self._Xp = np.zeros((self.P, self.fft // 2 + 1), dtype=np.complex128)
        self._x_prev = np.zeros(block, dtype=np.float64)
        self._in_near = np.zeros(0, dtype=np.float64)
        self._in_ref = np.zeros(0, dtype=np.float64)
        self._out = np.zeros(0, dtype=np.float64)
        # ERLE initialised to a "converged" mid-value so the early-EWMA
        # readings don't pin echo_conf to 1.0 — but echo_conf also gates on
        # ``_converged`` below for the first ~40 NLMS updates so even this
        # mid value isn't trusted at session start.
        self._erle = 12.0
        self._update_count = 0
        self._dtd = _GeigelDTD(cfg.sample_rate, threshold=cfg.aec_dtd_threshold)
        block_ms = 1000.0 * block / cfg.sample_rate
        self._dt_hold_blocks = max(1, int(cfg.aec_dt_freeze_ms / max(block_ms, 1e-3)))
        self._dt_remaining = 0
        self._last_res = (0.0, 0.0, 0.0)
        self._last_dt = False
        self._last_nles_atten_db = 0.0
        self._last_echo_conf = 0.0
        # Real Welch-style coherence estimator (replaces the broken V3-α formula).
        self._coh = _CoherenceEstimator(n_bins=self.fft // 2 + 1, alpha=0.2)
        # 50%-overlap NLES processor (replaces the V3-α edge-passthrough hack).
        self._nles = _NlesOlaProcessor(N=block)

        # ── GCC-PHAT calibration state (sliding-window buffer) ───────────
        self._calib_done = False
        self._calib_samples_target = int(cfg.sample_rate * self._CALIB_WINDOW_S)
        self._calib_timeout_samples = int(cfg.sample_rate * self._CALIB_TIMEOUT_S)
        # Sliding-window deques cap memory regardless of how long calibration
        # waits for a confident peak.
        self._calib_near = deque(maxlen=self._calib_samples_target)
        self._calib_ref = deque(maxlen=self._calib_samples_target)
        # Total samples seen during calibration (for the timeout watchdog).
        self._calib_total_samples = 0
        self._ref_delay = 0
        self._ref_delay_buf = np.zeros(0, dtype=np.float64)

    def reset(self) -> None:
        self._W[:] = 0
        self._Xp[:] = 0
        self._x_prev[:] = 0
        self._in_near = np.zeros(0)
        self._in_ref = np.zeros(0)
        self._out = np.zeros(0)
        self._erle = 12.0
        self._update_count = 0
        self._dtd.reset()
        self._dt_remaining = 0
        self._last_res = (0.0, 0.0, 0.0)
        self._last_dt = False
        self._last_nles_atten_db = 0.0
        self._last_echo_conf = 0.0
        self._coh.reset()
        self._nles.reset()
        self._calib_done = False
        self._calib_near.clear()
        self._calib_ref.clear()
        self._calib_total_samples = 0
        self._ref_delay = 0
        self._ref_delay_buf = np.zeros(0, dtype=np.float64)

    # ── GCC-PHAT calibration ─────────────────────────────────────────────
    def _accumulate_calibration(self, near: np.ndarray, ref: np.ndarray) -> None:
        """Append into the sliding-window calibration buffers."""
        self._calib_near.extend(near.astype(np.float64).tolist())
        self._calib_ref.extend(ref.astype(np.float64).tolist())
        self._calib_total_samples += len(near)

    def _try_calibrate(self) -> None:
        """Run GCC-PHAT periodically once buffers fill.  Does not lock in a
        bad estimate on a weak peak — calibration stays armed until either
        the peak threshold is cleared or the watchdog timeout fires (in
        which case we accept delay=0 to start adapting)."""
        if self._calib_done or not self.cfg.aec_gcc_phat:
            return
        # Watchdog: if calibration has been pending too long (silent reference,
        # or all weak peaks), accept delay=0 so the AEC can start adapting.
        if self._calib_total_samples >= self._calib_timeout_samples:
            self._ref_delay = 0
            self._ref_delay_buf = np.zeros(0, dtype=np.float64)
            self._calib_done = True
            self._calib_near.clear()
            self._calib_ref.clear()
            return
        # Need at least the target window's worth of samples and a non-trivial ref.
        if len(self._calib_near) < self._calib_samples_target:
            return
        ref_arr = np.fromiter(self._calib_ref, dtype=np.float64, count=len(self._calib_ref))
        ref_rms = float(np.sqrt(np.mean(ref_arr * ref_arr))) if len(ref_arr) else 0.0
        if ref_rms < self._MIN_REF_RMS:
            return     # buffer keeps sliding; try again next call
        near_arr = np.fromiter(self._calib_near, dtype=np.float64, count=len(self._calib_near))
        max_lag = int(self.cfg.sample_rate * self.cfg.aec_gcc_max_delay_ms / 1000.0)
        delay, peak = _gcc_phat_delay(
            near_arr, ref_arr,
            max_lag_samples=max_lag,
            peak_threshold=self.cfg.aec_gcc_peak_threshold,
        )
        # Only commit when the peak cleared the threshold.  Otherwise stay
        # armed and wait for a better window.
        if peak < self.cfg.aec_gcc_peak_threshold:
            return
        delay = int(np.clip(delay, 0, max_lag))
        self._ref_delay = delay
        self._ref_delay_buf = np.zeros(delay, dtype=np.float64)
        self._calib_done = True
        self._calib_near.clear()
        self._calib_ref.clear()

    def _delay_reference(self, ref: np.ndarray) -> np.ndarray:
        """Apply the calibrated delay to a reference chunk."""
        d = self._ref_delay
        if d <= 0:
            return ref
        n = len(ref)
        if n >= d:
            out = np.concatenate([self._ref_delay_buf, ref[: n - d]])
            self._ref_delay_buf = ref[n - d:].astype(np.float64).copy()
        else:
            out = self._ref_delay_buf[:n].copy()
            self._ref_delay_buf = np.concatenate(
                [self._ref_delay_buf[n:], ref.astype(np.float64)]
            )
        return out

    # ── PBFDAF inner loop ────────────────────────────────────────────────
    def _process_block(self, d: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cancel echo from one N-sample near-end block.  Returns
        ``(e_linear, x, y_echo_est)`` so the caller can use the *estimated
        echo at the mic* (y_echo_est = W·X) — not the raw reference — as
        the NLES Wiener numerator."""
        dt = self._dtd(d, x)
        if dt:
            self._dt_remaining = self._dt_hold_blocks
        elif self._dt_remaining > 0:
            self._dt_remaining -= 1
        self._last_dt = dt or self._dt_remaining > 0

        frame = np.concatenate([self._x_prev, x])
        X = np.fft.rfft(frame)
        self._Xp = np.roll(self._Xp, 1, axis=0)
        self._Xp[0] = X

        Y = np.sum(self._W * self._Xp, axis=0)
        y_full = np.fft.irfft(Y)
        y_echo_est = y_full[self.N:]      # last N samples = echo estimate at mic
        e = d - y_echo_est

        mu_eff = self.mu_base if not self._last_dt else 0.0
        if mu_eff > 0.0:
            E = np.fft.rfft(np.concatenate([np.zeros(self.N), e]))
            power = np.sum(np.abs(self._Xp) ** 2, axis=0) + 1e-6
            for p in range(self.P):
                G = np.conj(self._Xp[p]) * E / power
                g = np.fft.irfft(G)
                g[self.N:] = 0.0
                self._W[p] += mu_eff * np.fft.rfft(g)
            self._update_count += 1

        self._x_prev = x
        de = float(np.mean(d * d)) + 1e-12
        ee = float(np.mean(e * e)) + 1e-12
        self._erle = 0.95 * self._erle + 0.05 * (10.0 * np.log10(de / ee))
        # Real coherence (Welch-style EWMA across blocks) between residual
        # and far-end reference.  Genuine 0 when the linear filter has fully
        # cancelled; near 1.0 when residual still tracks the reference.
        E_block = np.fft.rfft(e)
        X_block = np.fft.rfft(x)
        self._last_res = self._coh.update(E_block, X_block)
        return e, x, y_echo_est

    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.reference is None:
            ctx.dt_active = False
            ctx.res_band = (0.0, 0.0, 0.0)
            ctx.echo_conf = 0.0
            return ctx

        # Calibration phase: accumulate a sliding window of (near, ref)
        # audio.  Until calibration completes, pass through the mic and emit
        # echo_conf=0 / dt=False / res=(0,0,0).  A meta flag tells downstream
        # stages to suspend their adaptive updates.
        if not self._calib_done and self.cfg.aec_gcc_phat:
            self._accumulate_calibration(ctx.audio, ctx.reference)
            self._try_calibrate()
            ctx.dt_active = False
            ctx.res_band = (0.0, 0.0, 0.0)
            ctx.echo_conf = 0.0
            ctx.meta["aec_calibrating"] = 1.0
            return ctx

        # Apply the calibrated delay to the incoming reference.
        delayed_ref = self._delay_reference(ctx.reference.astype(np.float64))
        self._in_near = np.concatenate([self._in_near, ctx.audio.astype(np.float64)])
        self._in_ref = np.concatenate([self._in_ref, delayed_ref])

        while len(self._in_near) >= self.N and len(self._in_ref) >= self.N:
            d = self._in_near[: self.N]; self._in_near = self._in_near[self.N:]
            x = self._in_ref[: self.N];  self._in_ref = self._in_ref[self.N:]
            e_lin, x_used, y_echo_est = self._process_block(d, x)
            # NLES: 50%-OLA per-bin Wiener using the ESTIMATED echo y_echo_est
            # (W·X) — NOT the raw reference x.  This is the key correctness fix.
            if self.cfg.nles_enabled:
                g_min_lin = 10.0 ** (self.cfg.nles_min_gain_db / 20.0)
                e_post, atten_db = self._nles.process(
                    e_lin, y_echo_est,
                    beta=self.cfg.nles_beta,
                    g_min_lin=g_min_lin,
                )
                self._last_nles_atten_db = atten_db
                self._out = np.concatenate([self._out, e_post])
            else:
                self._last_nles_atten_db = 0.0
                self._out = np.concatenate([self._out, e_lin])

        n = len(ctx.audio)
        if len(self._out) >= n:
            out = self._out[:n]; self._out = self._out[n:]
            ctx.audio = out.astype("float32")
        else:
            ready = len(self._out)
            out = np.concatenate([self._out, np.zeros(n - ready, dtype=np.float64)])
            self._out = np.zeros(0, dtype=np.float64)
            ctx.audio = out.astype("float32")
            ctx.meta["aec_warmup"] = 1.0

        # echo_conf gates ERLE behind a convergence flag — first N NLMS
        # updates are not trusted (filter is still cold).
        converged = self._update_count >= self._ERLE_CONVERGE_BLOCKS
        echo_conf = _echo_confidence(
            dt_active=self._last_dt,
            res_band=self._last_res,
            erle_db=self._erle,
            nles_atten_db=self._last_nles_atten_db,
            converged=converged,
        )
        self._last_echo_conf = echo_conf

        ctx.meta["erle_db"] = self._erle
        ctx.meta["aec_dt"] = 1.0 if self._last_dt else 0.0
        ctx.meta["res_lo"] = self._last_res[0]
        ctx.meta["res_md"] = self._last_res[1]
        ctx.meta["res_hi"] = self._last_res[2]
        ctx.meta["nles_atten_db"] = self._last_nles_atten_db
        ctx.meta["echo_conf"] = echo_conf
        ctx.meta["aec_ref_delay_samples"] = float(self._ref_delay)
        ctx.meta["aec_converged"] = 1.0 if converged else 0.0
        ctx.meta["aec_calibrating"] = 0.0
        ctx.dt_active = self._last_dt
        ctx.res_band = self._last_res
        ctx.echo_conf = echo_conf
        return ctx

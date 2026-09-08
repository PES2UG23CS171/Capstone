"""
Deferred-tail timeline + intrusive-metric alignment regression tests.

Same conventions as the other suites: dependency-free runner
(``python -m tests.test_deferred_tail``), pytest-compatible, skips cleanly
when a heavy backend is unavailable.

The unit tests drive the REAL ``Enhancement`` stage (its slicing / history /
bypass logic runs unchanged) with a deterministic FAKE DeepFilterNet3 backend:
``enhance`` is replaced by a scaled identity (×0.5), so a "wet" sample is
distinguishable from a "dry" one and every emitted sample can be traced back
to exactly one input sample.  No model weights are needed.  One integration
test uses the installed DFN3 backend when it is available.

What is pinned (2026-09-08 deferred-tail work):
  1. dfn_tail_ms=20 → exactly 960 samples @ 48 kHz (algorithmic latency 40 ms)
  2. tail 0 ms reproduces the previous (zero-added-latency) behaviour
  3. bypass and DFN3-active paths sit on the same delayed timeline
  4. active↔bypass transitions never insert or delete samples
  5. reset() clears the history AND the deferred-tail state
  6. partial / short final blocks (shorter than the tail) are handled
  7. intrusive scoring never compares zero padding with valid reference audio
  8. artificial delays of +960, −960 and +15 samples are estimated exactly,
     and impossible / mismatched alignments are rejected
  9. valid-overlap alignment preserves identity and SI-SDR scale invariance
 10. repeated runs are bit-identical (output and metrics)
"""
from __future__ import annotations

import hashlib

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages import enhancement as E
from voiceiso.stages.base import FrameContext

SR = 48_000
B = 4_800          # 100 ms block
TAIL = 960         # 20 ms @ 48 kHz
WET = 0.5          # fake DFN3 = identity × WET


class _Skip(Exception):
    pass


# ── deterministic fake DFN3 backend ───────────────────────────────────────────
class _FakeState:
    def __init__(self, sr=SR, fft_size=960, hop_size=480, nb_bands=32):
        pass

    def sr(self):
        return SR

    def fft_size(self):
        return 960

    def hop_size(self):
        return 480


def _fake_init_df(**kw):
    return object(), _FakeState(), None


def _fake_enhance(model, state, x, atten_lim_db=None):
    return x * WET


class _FakeDF:
    """Swap the DF backend for the deterministic fake while constructing a stage."""

    def __enter__(self):
        try:
            import torch  # noqa: F401 — the stage's tensor plumbing needs torch itself
        except Exception as exc:  # pragma: no cover
            raise _Skip(f"torch unavailable ({exc})")
        self._saved = (E._HAS_DF, getattr(E, "init_df", None), getattr(E, "enhance", None))
        E._HAS_DF = True
        E.init_df = _fake_init_df
        E.enhance = _fake_enhance
        return self

    def __exit__(self, *a):
        E._HAS_DF, E.init_df, E.enhance = self._saved


def _mk(tail_ms: float, hist_ms: float = 160.0) -> E.Enhancement:
    with _FakeDF():
        enh = E.Enhancement(PipelineConfig(dfn_tail_ms=tail_ms, dfn_history_ms=hist_ms), threads=1)
    assert enh.backend == "deepfilternet3"
    return enh


def _ramp(n: int) -> np.ndarray:
    """Strictly positive, unique-valued signal: every sample identifies its index."""
    return ((np.arange(n) + 1) / float(n)).astype(np.float32)


def _drive(enh: E.Enhancement, x: np.ndarray, supp_seq, blocks=None):
    """Push ``x`` through the stage; returns (output, per-sample wet flag).

    ``supp_seq[k]`` is the suppression for block k (≤ 0.08 → bypass).
    ``blocks`` optionally gives explicit block lengths (for partial blocks).
    """
    out = np.zeros_like(x)
    wet = np.zeros(len(x), dtype=np.float32)
    starts = []
    if blocks is None:
        starts = [(s, min(B, len(x) - s)) for s in range(0, len(x), B)]
    else:
        s = 0
        for L in blocks:
            starts.append((s, L)); s += L
        assert s == len(x)
    for k, (s, L) in enumerate(starts):
        with _FakeDF():
            ctx = FrameContext(audio=x[s:s + L].copy(), sample_rate=SR)
            ctx.suppression = float(supp_seq[k % len(supp_seq)])
            y = enh.process(ctx).audio
        assert len(y) == L, f"block {k}: emitted {len(y)} samples for {L} input samples"
        out[s:s + L] = y
        wet[s:s + L] = ctx.meta["enh_wet"]
    return out, wet


def _check_timeline(x, out, wet, delay):
    """Every emitted sample j ≥ delay must equal x[j-delay] × (WET if the block
    that emitted it ran DFN3 else 1); samples before ``delay`` are zero history."""
    assert np.all(out[:delay] == 0.0), "the first `delay` samples must be zero history"
    j = np.arange(delay, len(x))
    expect = x[j - delay] * np.where(wet[j] > 0.5, WET, 1.0)
    bad = np.where(out[j] != expect.astype(np.float32))[0]
    assert bad.size == 0, (f"timeline broken at {bad.size} samples, first at j={j[bad[0]]}: "
                           f"got {out[j[bad[0]]]}, expected {expect[bad[0]]}")


# ── 1. tail = 20 ms is exactly 960 samples ────────────────────────────────────
def test_tail_20ms_is_960_samples():
    enh = _mk(20.0)
    assert enh._tail == TAIL
    assert enh.output_delay_samples == TAIL
    assert enh._hist_len == 16 * 480 + TAIL, "history must cover priming + tail"
    assert PipelineConfig(dfn_tail_ms=20).algorithmic_latency_ms == 40.0
    x = _ramp(3 * B)
    out, wet = _drive(enh, x, [1.0])
    _check_timeline(x, out, wet, TAIL)


# ── 2. tail 0 reproduces the previous zero-added-latency behaviour ────────────
def test_tail_0_preserves_previous_behaviour():
    enh = _mk(0.0)
    assert enh._tail == 0 and enh.output_delay_samples == 0
    x = _ramp(3 * B)
    out, wet = _drive(enh, x, [1.0])
    assert np.array_equal(out, (x * WET).astype(np.float32)), "active path must be sample-aligned"
    enh.reset()
    out, wet = _drive(enh, x, [0.0])
    assert np.array_equal(out, x), "bypass path must be the dry block, untouched"
    assert enh._delayed_dry(x[:B]) is x[:B] or np.array_equal(enh._delayed_dry(x[:B]), x[:B])


# ── 3. bypass and active paths share one delayed timeline ────────────────────
def test_bypass_and_active_same_timeline():
    x = _ramp(4 * B)
    enh = _mk(20.0)
    out_a, wet_a = _drive(enh, x, [1.0])
    _check_timeline(x, out_a, wet_a, TAIL)
    enh.reset()
    out_b, wet_b = _drive(enh, x, [0.0])
    _check_timeline(x, out_b, wet_b, TAIL)
    assert np.all(wet_a == 1.0) and np.all(wet_b == 0.0)
    # identical timelines: bypass output == active output / WET
    assert np.allclose(out_b[TAIL:], out_a[TAIL:] / WET, rtol=0, atol=1e-7)


# ── 4. transitions insert / delete nothing ────────────────────────────────────
def test_transitions_do_not_splice():
    x = _ramp(12 * B)
    rng = np.random.default_rng(11)
    for seq in ([1.0, 0.0], [0.0, 1.0], [1.0, 1.0, 0.0], list(rng.integers(0, 2, 12).astype(float))):
        enh = _mk(20.0)
        out, wet = _drive(enh, x, seq)
        _check_timeline(x, out, wet, TAIL)      # exact per-sample provenance ⇒ no splice
        assert 0 < wet.mean() < 1 or len(set(seq)) == 1


# ── 5. reset clears history and the deferred tail ─────────────────────────────
def test_reset_clears_history_and_tail_state():
    enh = _mk(20.0)
    x = _ramp(3 * B)
    _drive(enh, x, [1.0])
    assert np.any(enh._hist != 0.0), "history must hold real audio after processing"
    enh.reset()
    assert np.all(enh._hist == 0.0) and len(enh._hist) == enh._hist_len
    assert enh.output_delay_samples == TAIL, "reset must not change the configured delay"
    out, wet = _drive(enh, x, [1.0])
    assert np.all(out[:TAIL] == 0.0), "after reset the first tail-length of output is zero history"
    _check_timeline(x, out, wet, TAIL)


# ── 6. partial / short final blocks ───────────────────────────────────────────
def test_partial_and_short_blocks():
    blocks = [B, B, 1234, 500, 100, 7]          # includes blocks shorter than the tail
    x = _ramp(sum(blocks))
    enh = _mk(20.0)
    out, wet = _drive(enh, x, [1.0, 0.0, 1.0], blocks=blocks)
    _check_timeline(x, out, wet, TAIL)


def _fake_pipeline(enh):
    """A StreamingPipeline whose only stage is the fake-backed enhancement."""
    from voiceiso.pipeline import StreamingPipeline

    class _Pre:
        def set_prev_noise_class(self, c):
            pass

    p = StreamingPipeline.__new__(StreamingPipeline)
    p.cfg = enh.cfg
    p.stages = [enh]
    p.preprocessing = _Pre()
    p._nan_warned = False
    p._nan_streak = 0
    return p


def test_process_signal_flush_recovers_tail_and_alignment():
    """process_signal(flush=True) must return len(x) samples ALIGNED with x
    (initial delay removed, final deferred samples recovered); flush=False
    must return the raw delayed stream."""
    for n in (1, 100, 500, TAIL, TAIL + 1, B - 1, B, B + 1, 3 * B + 321):
        x = _ramp(n)
        enh = _mk(20.0)
        with _FakeDF():
            y = _fake_pipeline(enh).process_signal(x)
        assert len(y) == n
        assert np.array_equal(y, (x * WET).astype(np.float32)), f"n={n}: flushed output not aligned/complete"
        enh = _mk(20.0)
        with _FakeDF():
            yd = _fake_pipeline(enh).process_signal(x, flush=False)
        assert len(yd) == n
        assert np.all(yd[:min(TAIL, n)] == 0.0)
        if n > TAIL:
            assert np.array_equal(yd[TAIL:], (x[:n - TAIL] * WET).astype(np.float32))


# ── 7. scoring never compares zero padding with reference audio ───────────────
def test_valid_overlap_excludes_padding():
    from voiceiso.bench.benchmark import estimate_lag, valid_overlap
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(SR * 2).astype(np.float32)
    for lag in (TAIL, -TAIL, 15):
        if lag > 0:
            x = np.concatenate([np.zeros(lag, np.float32), ref[:-lag]])      # output lags
        else:
            x = np.concatenate([ref[-lag:], np.zeros(-lag, np.float32)])     # output leads
        al = estimate_lag(ref, x)
        assert al.ok and al.lag == lag, (al.lag, al.reason)
        r, e = valid_overlap(ref, x, al.lag)
        assert len(r) == len(e) == len(ref) - abs(lag) == al.n_valid
        assert np.array_equal(r, e), "overlap must be the identical samples — no padding scored"
        assert not np.any(e == 0.0), "the zero padding must never enter the scored segment"


# ── 8. known delays are estimated exactly; impossible ones are rejected ───────
def test_lag_estimation_known_delays_and_rejection():
    from voiceiso.bench.benchmark import estimate_lag
    rng = np.random.default_rng(5)
    ref = rng.standard_normal(SR * 2).astype(np.float32)

    def delayed(sig, lag):
        if lag > 0:
            return np.concatenate([np.zeros(lag, np.float32), sig[:-lag]])
        if lag < 0:
            return np.concatenate([sig[-lag:], np.zeros(-lag, np.float32)])
        return sig.copy()

    for lag in (960, -960, 15, 0, 975):
        al = estimate_lag(ref, delayed(ref, lag))
        assert al.ok and al.lag == lag, f"lag {lag}: got {al.lag} ({al.reason})"
    # scale + additive noise still resolve exactly
    noisy = (0.3 * delayed(ref, 960) + 0.05 * rng.standard_normal(len(ref))).astype(np.float32)
    al = estimate_lag(ref, noisy)
    assert al.ok and al.lag == 960, (al.lag, al.reason)
    # unrelated signal → REJECTED, never a silent score
    al = estimate_lag(ref, rng.standard_normal(len(ref)).astype(np.float32))
    assert not al.ok and al.reason, "an unrelated signal must be rejected with a reason"
    # true lag beyond the search window → REJECTED (no in-window peak is trustworthy)
    al = estimate_lag(ref, delayed(ref, 3000), max_lag=2400)
    assert not al.ok and al.reason, "an out-of-window lag must be rejected, not mis-scored"
    # silence → rejected
    al = estimate_lag(ref, np.zeros_like(ref))
    assert not al.ok and "silent" in al.reason


# ── 8b. a KNOWN pipeline delay overrides a jittery estimate, and rejects a wrong one
def test_known_delay_resolution():
    from voiceiso.bench.benchmark import Alignment, resolve_lag
    est = Alignment(lag=973, n_valid=10_000 - 973, corr=0.6, peak_ratio=3.0, ok=True)
    al = resolve_lag(est, expected=975)
    assert al.ok and al.lag == 975 and al.n_valid == 10_000 - 975, al
    bad = resolve_lag(Alignment(900, 9_100, 0.6, 3.0, True), expected=975)
    assert not bad.ok and "disagrees" in bad.reason
    assert resolve_lag(est, expected=None) is est
    rej = Alignment(0, 0, 0.0, 0.0, False, "silent input")
    assert resolve_lag(rej, expected=975) is rej


# ── 9. identity and scale invariance after valid-overlap alignment ────────────
def test_alignment_preserves_identity_and_scale_invariance():
    from voiceiso.bench import metrics as M
    from voiceiso.bench.benchmark import estimate_lag, valid_overlap
    rng = np.random.default_rng(9)
    ref = rng.standard_normal(SR).astype(np.float32)
    # (a) identity: a scaled, delayed copy must score as (near-)perfect
    x = np.concatenate([np.zeros(TAIL, np.float32), ref[:-TAIL]]) * 0.3
    al = estimate_lag(ref, x)
    assert al.ok and al.lag == TAIL
    r, e = valid_overlap(ref, x, al.lag)
    assert M.si_sdr(r, e) > 60.0, "a scaled, delayed identity must score as (near-)perfect"
    assert abs(M.correlation(r, e) - 1.0) < 1e-6
    # (b) scale invariance on a NON-degenerate estimate (≈ 30 dB, so the
    #     metric's 1e-9 epsilon is negligible): scaling the estimate by 4×
    #     must not change SI-SDR.
    est = (ref + 0.03 * rng.standard_normal(SR)).astype(np.float32)
    xd = np.concatenate([np.zeros(TAIL, np.float32), est[:-TAIL]])
    al = estimate_lag(ref, xd)
    r, e = valid_overlap(ref, xd, al.lag)
    s1, s4 = M.si_sdr(r, e), M.si_sdr(r, 4.0 * e)
    assert 25.0 < s1 < 40.0, s1
    assert abs(s1 - s4) < 1e-4, f"SI-SDR must be scale-invariant ({s1:.6f} vs {s4:.6f})"


# ── 10. determinism ───────────────────────────────────────────────────────────
def test_repeated_runs_identical():
    from voiceiso.bench import metrics as M
    from voiceiso.bench.benchmark import estimate_lag, valid_overlap
    rng = np.random.default_rng(21)
    x = (0.1 * rng.standard_normal(6 * B)).astype(np.float32)
    hashes, scores = [], []
    for _ in range(2):
        enh = _mk(20.0)
        out, _ = _drive(enh, x, [1.0, 0.0, 1.0, 1.0])
        hashes.append(hashlib.sha256(out.tobytes()).hexdigest())
        al = estimate_lag(x, out)
        r, e = valid_overlap(x, out, al.lag)
        scores.append((al.lag, M.si_sdr(r, e), M.correlation(r, e)))
    assert hashes[0] == hashes[1], "same input + config must give bit-identical output"
    assert scores[0] == scores[1], "metrics must be identical across repeated runs"


# ── optional integration: the installed DeepFilterNet3 backend ────────────────
def test_integration_real_dfn3_tail_flush_and_determinism():
    try:
        from voiceiso.pipeline import StreamingPipeline
        pipe = StreamingPipeline(PipelineConfig())
    except Exception as exc:  # pragma: no cover
        raise _Skip(f"pipeline unavailable ({exc})")
    if pipe.enhancement.backend != "deepfilternet3":
        raise _Skip("DeepFilterNet3 backend not available")
    from tests.test_robustness import _speech_like
    from voiceiso.bench.benchmark import estimate_lag
    assert pipe.enhancement._tail == TAIL
    assert pipe.output_delay_samples == TAIL + pipe.multiband.output_delay_samples == 975
    x = _speech_like(2.0)
    pipe.reset()
    y = pipe.process_signal(x)
    assert len(y) == len(x)
    assert np.all(np.isfinite(y))
    assert np.sqrt(np.mean(y[-975:] ** 2)) > 1e-4, "the flushed final samples must carry audio"
    al = estimate_lag(x, y)
    if al.ok:
        assert abs(al.lag) <= 2, f"flushed output must be aligned with the input (lag {al.lag})"
    pipe.reset()
    y2 = pipe.process_signal(x)
    assert np.array_equal(y, y2), "same input + config must be bit-identical across runs"
    pipe.reset()
    yd = pipe.process_signal(x, flush=False)
    al_d = estimate_lag(x, yd)
    if al_d.ok:
        assert al_d.lag == 975, f"unflushed stream must lag by the constant delay (got {al_d.lag})"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    npass = nskip = nfail = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            npass += 1
        except _Skip as s:
            print(f"SKIP  {fn.__name__}  ({s})")
            nskip += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}  -> {type(e).__name__}: {e}")
            nfail += 1
    print(f"\n{npass} passed, {nskip} skipped, {nfail} failed")
    return nfail


if __name__ == "__main__":
    import sys
    sys.exit(1 if _run_all() else 0)

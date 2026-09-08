"""
Benchmark harness.

Measures, over a set of (clean, noisy) pairs:
  * Quality (intrusive):  SI-SDR(i), segmental SNR, correlation, PESQ-wb, STOI
  * Quality (non-intrusive): DNSMOS P.835 SIG / BAK / OVRL (mean, p50, p95)
  * Speed:    RTF, per-block latency p50/p95/p99
  * Resource: peak RSS (RAM), CPU utilisation %

Two higher-level entry points build on :func:`run_benchmark`:
  * :func:`summarize_runs`   — print a comparison table across labelled runs
    (used for the SNR sweep and per-class breakdown).

Acceptance criteria — three DIFFERENT things, reported separately (see
``SUCCESS`` and :meth:`BenchResult.report`):

  * **Phase-2 project objective** (the capstone's release-gating criterion,
    from the Phase-2 review decks): SI-SDRi ≥ +4.0 dB at 5 dB input SNR,
    end-to-end call latency < 300 ms.
  * **Aspirational "approaching Krisp on CPU" targets** (set with the June
    2026 redesign; never a release gate): SI-SDRi ≥ +10 dB, PESQ-wb ≥ 2.6,
    DNSMOS OVRL ≥ 3.0, RAM ≤ 300 MB.
  * **Real-time budget**: RTF ≤ 0.5, algorithmic latency ≤ 40 ms.

The report never prints "PASS": it states, per criterion, met / below, and
labels the condition (synthetic white noise vs real-noise corpora) — the two
are not comparable and must not be quoted against each other.

Intrusive metrics (SI-SDR, seg-SNR, correlation, PESQ, STOI) are scored ONLY
over the valid overlap of reference and output after lag estimation — never
against zero padding — see :func:`estimate_lag` / :func:`valid_overlap`.
DNSMOS is non-intrusive and is scored on the full, unaligned output.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from voiceiso.bench import metrics as M
from voiceiso.config import PipelineConfig
from voiceiso.pipeline import StreamingPipeline

SUCCESS = {
    # Phase-2 project objective — the only release-gating quality criterion.
    "si_sdri_objective_db": 4.0,
    # Aspirational Krisp-parity targets (June 2026 redesign) — informational.
    "si_sdri_db": 10.0,
    "pesq_wb": 2.6,
    "dnsmos_ovrl": 3.0,
    "ram_mb": 300.0,
    # Real-time budget.
    "rtf": 0.5,
    "latency_ms": 40.0,
}


def _peak_rss_mb() -> float:
    import resource, sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024  # bytes vs KB


@dataclass
class Alignment:
    """Result of :func:`estimate_lag`.  ``lag > 0`` means ``x`` lags ``ref``."""
    lag: int
    n_valid: int          # overlapping samples available for scoring
    corr: float           # normalised cross-correlation at the chosen lag
    peak_ratio: float     # best peak / strongest competing peak (≥ 1)
    ok: bool
    reason: str = ""


def overlap_slices(n: int, lag: int) -> Tuple[slice, slice]:
    """Index ranges of the VALID overlap of two length-``n`` signals when the
    second lags the first by ``lag`` samples.  No padding anywhere:

      * lag > 0 (output lags reference): ``ref[:n-lag]`` vs ``x[lag:n]``
      * lag < 0 (output leads):          ``ref[-lag:n]`` vs ``x[:n+lag]``
    """
    if lag > 0:
        return slice(0, n - lag), slice(lag, n)
    if lag < 0:
        return slice(-lag, n), slice(0, n + lag)
    return slice(0, n), slice(0, n)


def valid_overlap(ref: np.ndarray, x: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Crop ``ref`` and ``x`` to their valid overlap for a known/estimated
    ``lag`` (see :func:`overlap_slices`).  Returns equal-length views."""
    n = min(len(ref), len(x))
    s_ref, s_x = overlap_slices(n, lag)
    return ref[s_ref], x[s_x]


def estimate_lag(ref: np.ndarray, x: np.ndarray, max_lag: int = 2400,
                 min_overlap_frac: float = 0.5, min_corr: float = 0.1,
                 min_peak_ratio: float = 1.1) -> Alignment:
    """Estimate how many samples ``x`` lags ``ref`` (cross-correlation, FFT).

    The pipeline delays its output by a constant, known amount (the enhancement
    stage's deferred tail, 960 samples by default, plus the multi-band FIR's
    15-sample group delay = 975); SI-SDR is exquisitely delay-sensitive (a
    15-sample shift alone caps it at ≈ −5 dB), so intrusive metrics must be
    scored on the aligned VALID overlap only.  This function only *estimates*
    the lag; :func:`valid_overlap` does the cropping (never zero padding).

    The estimate is REJECTED (``ok=False``, with a reason) instead of returned
    silently when it is impossible or ambiguous:
      * the best peak sits on the search-window edge (true lag may be outside);
      * the normalised correlation at the peak is below ``min_corr`` (the
        signals do not match — e.g. wrong pair, polarity inverted);
      * a competing peak outside the main lobe (±5 ms) is within
        ``min_peak_ratio`` of the best one (ambiguous, e.g. periodic input);
      * the valid overlap would be shorter than ``min_overlap_frac`` of ``n``.
    """
    from scipy.signal import correlate
    n = min(len(ref), len(x))
    if n == 0:
        return Alignment(0, 0, 0.0, 0.0, False, "empty input")
    r = ref[:n].astype(np.float64); e = x[:n].astype(np.float64)
    r = r - r.mean(); e = e - e.mean()
    nr, ne = float(np.linalg.norm(r)), float(np.linalg.norm(e))
    if nr < 1e-12 or ne < 1e-12:
        return Alignment(0, n, 0.0, 0.0, False, "silent input — nothing to align")
    max_lag = int(min(max_lag, n - 1))
    c = correlate(e, r, mode="full", method="fft")
    mid = n - 1
    w = c[mid - max_lag: mid + max_lag + 1]
    k = int(np.argmax(w))
    lag = k - max_lag
    peak = float(w[k])
    corr = peak / (nr * ne)
    guard = 240                                   # 5 ms main lobe @ 48 kHz
    mask = np.ones(len(w), dtype=bool)
    mask[max(0, k - guard): k + guard + 1] = False
    second = float(np.max(w[mask])) if mask.any() else 0.0
    ratio = peak / max(second, 1e-12) if second > 0 else float("inf")
    n_valid = n - abs(lag)
    ok, reason = True, ""
    if k == 0 or k == len(w) - 1:
        ok, reason = False, (f"best peak on the search-window edge (|lag| = {abs(lag)} "
                             f"≥ {max_lag}) — the true lag may lie outside the window")
    elif corr < min_corr:
        ok, reason = False, (f"normalised correlation {corr:.3f} < {min_corr} — "
                             "output does not match the reference")
    elif ratio < min_peak_ratio:
        ok, reason = False, (f"ambiguous lag — a competing peak is within {ratio:.2f}× "
                             f"of the best (need ≥ {min_peak_ratio})")
    elif n_valid < min_overlap_frac * n:
        ok, reason = False, (f"valid overlap {n_valid} samples < "
                             f"{min_overlap_frac:.0%} of {n}")
    return Alignment(int(lag), int(n_valid), float(corr), float(ratio), ok, reason)


def resolve_lag(al: Alignment, expected: Optional[int] = None, tol: int = 5) -> Alignment:
    """Reconcile an ESTIMATED lag with a KNOWN pipeline delay.

    When the pipeline's constant output delay is known (``expected`` =
    ``StreamingPipeline.output_delay_samples``) the exact value is used for
    scoring: cross-correlation on long or noise-dominated material jitters by
    a few samples, and SI-SDR punishes even a 2–3-sample offset.  The estimate
    then acts as a cross-check — a valid estimate that disagrees with the
    known delay by more than ``tol`` samples means the output is NOT on the
    expected timeline (a stage removed, a splice, the wrong pair) and the
    alignment is REJECTED instead of silently scored.  Without ``expected``
    the estimate is returned unchanged.
    """
    if expected is None or not al.ok:
        return al
    if abs(al.lag - expected) > tol:
        return Alignment(al.lag, al.n_valid, al.corr, al.peak_ratio, False,
                         f"estimated lag {al.lag} disagrees with the known pipeline delay "
                         f"{expected} by more than {tol} samples")
    n = al.n_valid + abs(al.lag)                 # original overlap length
    return Alignment(int(expected), int(n - abs(expected)), al.corr, al.peak_ratio, True, "")


def _align_to_ref(ref: np.ndarray, x: np.ndarray, max_lag: int = 2400) -> Tuple[np.ndarray, int]:
    """DEPRECATED — kept only for external callers.  Shifts ``x`` onto
    ``ref``'s timeline and ZERO-PADS the vacated end, so any intrusive metric
    computed on the result scores padding against valid reference audio.
    Use :func:`estimate_lag` + :func:`valid_overlap` instead."""
    al = estimate_lag(ref, x, max_lag=max_lag)
    lag = al.lag
    if lag > 0:
        x = np.concatenate([x[lag:], np.zeros(lag, dtype=x.dtype)])
    elif lag < 0:
        x = np.concatenate([np.zeros(-lag, dtype=x.dtype), x[:lag]])
    return x, lag


def _agg(vals: List[float]) -> Dict[str, float]:
    """mean / p50 / p95 of a list (empty → zeros)."""
    if not vals:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


@dataclass
class BenchResult:
    quality: Dict[str, float] = field(default_factory=dict)
    rtf: float = 0.0
    cpu_pct: float = 0.0
    latency_ms: Dict[str, float] = field(default_factory=dict)
    ram_mb: float = 0.0
    # DNSMOS aggregates: {sig: {mean,p50,p95}, bak: {...}, ovrl: {...}}
    dnsmos: Dict[str, Dict[str, float]] = field(default_factory=dict)
    backends: Dict[str, object] = field(default_factory=dict)
    label: str = ""
    n_pairs: int = 0
    block_ms: float = 0.0      # block size the pipeline was driven at
    # Estimated output/reference lag (samples); intrusive metrics are scored
    # on the valid overlap only (see estimate_lag / valid_overlap).
    align_lag: Dict[str, float] = field(default_factory=dict)
    # Valid scored samples per pair (mean / min) and their duration.
    valid_samples: Dict[str, float] = field(default_factory=dict)
    # Pairs whose alignment was rejected (excluded from intrusive metrics).
    rejected: List[str] = field(default_factory=list)
    # sha256 (first 16 hex) of every pair's raw pipeline output, in order —
    # identical across runs ⇔ bit-identical processing.
    out_hash: str = ""

    def report(self) -> str:
        q = self.quality
        cond = ("synthetic white noise" if "SYNTH" in self.label.upper()
                else "as labelled")
        lines = ["=" * 60, f"  voiceiso BENCHMARK{(' — ' + self.label) if self.label else ''}", "=" * 60]
        lines.append(f"  backends           : {self.backends}")
        lines.append(f"  pairs              : {self.n_pairs}"
                     + (f"  (intrusive metrics on {self.n_pairs - len(self.rejected)}; "
                        f"{len(self.rejected)} rejected — see below)" if self.rejected else ""))
        if self.block_ms:
            tag = ("low-latency/low-quality" if self.block_ms <= 40
                   else "design-point (matches live)" if self.block_ms <= 120
                   else "max-quality/high-latency")
            lines.append(f"  block size         : {self.block_ms:.0f} ms  [{tag}]")
        for k in ("si_sdr_in", "si_sdr_out", "si_sdri", "seg_snr_out",
                  "corr_out", "pesq_in", "pesq_out", "stoi_out"):
            if k in q:
                lines.append(f"  {k:<18} : {q[k]:+.3f}")
        if self.align_lag:
            lines.append(f"  output delay       : {self.align_lag['known']} samples known (scored at this lag, "
                         "valid overlap only); cross-correlation estimate "
                         f"mean {self.align_lag['mean']:+.1f} min {self.align_lag['min']:+d} "
                         f"max {self.align_lag['max']:+d}")
        if self.valid_samples:
            v = self.valid_samples
            lines.append(f"  valid scored       : mean {v['mean']:.0f} samples "
                         f"({v['mean_s']:.3f} s/pair)  min {v['min']:.0f}  "
                         f"total {v['total_s']:.2f} s")
        for r in self.rejected:
            lines.append(f"  REJECTED           : {r}")
        if self.dnsmos:
            for sub in ("sig", "bak", "ovrl"):
                if sub in self.dnsmos:
                    d = self.dnsmos[sub]
                    lines.append(f"  DNSMOS {sub.upper():<11} : "
                                 f"mean {d['mean']:.3f}  p50 {d['p50']:.3f}  p95 {d['p95']:.3f}"
                                 "   [non-intrusive; full unaligned output]")
        lines.append(f"  RTF                : {self.rtf:.3f}")
        lines.append(f"  CPU                : {self.cpu_pct:.0f} %")
        for k, v in self.latency_ms.items():
            lines.append(f"  latency {k:<10} : {v:.2f} ms")
        lines.append(f"  peak RAM           : {self.ram_mb:.0f} MB")
        if self.out_hash:
            lines.append(f"  output sha256      : {self.out_hash}  (identical across runs ⇔ deterministic)")
        lines.append("-" * 60)
        si = q.get("si_sdri", float("nan"))
        obj = SUCCESS["si_sdri_objective_db"]
        asp = SUCCESS["si_sdri_db"]
        lines.append(f"  condition          : {cond}")
        lines.append(f"  Phase-2 objective  : SI-SDRi ≥ {obj:.1f} dB  → "
                     f"{'met' if si >= obj else 'BELOW'} ({si:+.2f} dB)")
        lines.append(f"  aspirational       : SI-SDRi ≥ {asp:.0f} dB (Krisp-parity)  → "
                     f"{'met' if si >= asp else 'not met'}")
        lines.append(f"  real-time budget   : RTF ≤ {SUCCESS['rtf']}  → "
                     f"{'met' if self.rtf <= SUCCESS['rtf'] else 'EXCEEDED'} ({self.rtf:.3f})")
        lines.append("=" * 60)
        return "\n".join(lines)


def run_benchmark(pairs: List[Tuple[np.ndarray, np.ndarray]], sr: int = 48_000,
                  cfg: PipelineConfig | None = None,
                  dnsmos_model: Optional[str] = None,
                  label: str = "", block_ms: float = 100.0,
                  warmup: int = 1) -> BenchResult:
    """``pairs`` = list of (clean, noisy) float32 arrays at ``sr``.

    ``block_ms`` is the block size the pipeline is driven at.  Default 100 ms to
    MATCH the live paths (AppConfig.block_size, LiveStream) — DFN3's design point
    where it has enough per-call STFT context to actually enhance (at 20 ms,
    SI-SDRi is negative; see Docs/ARCHITECTURE.md §1.4). Pass ``block_ms=20`` to see
    the low-latency-but-low-quality point.  This is the honest, demo-matching
    operating point — the old code hard-coded 100 ms (= ``cfg.win*5``) too, but
    the app then shipped 20 ms, so the headline numbers didn't match the demo.

    ``warmup`` pairs are processed (and timed-out) before timing starts so model
    init / first-call graph optimisation don't skew RTF.

    ``dnsmos_model`` (optional) path to the DNSMOS sig_bak_ovr.onnx — when set
    and loadable, SIG/BAK/OVRL are computed on each enhanced output and
    aggregated (mean/p50/p95).
    """
    cfg = cfg or PipelineConfig(sample_rate=sr)
    # DNSMOS model path can also come from cfg.
    dnsmos_model = dnsmos_model or cfg.dnsmos_model_path
    pipe = StreamingPipeline(cfg)
    res = BenchResult(backends=pipe.backend_summary, label=label, n_pairs=len(pairs),
                      block_ms=block_ms)

    # ── Warm-up (excluded from timing): run a few blocks so model init /
    #    first-call ORT graph optimisation don't inflate the reported RTF. ──
    block = max(1, int(round(sr * block_ms / 1000.0)))
    if pairs and warmup > 0:
        _, warm_noisy = pairs[0]
        for _ in range(warmup):
            pipe.reset()
            for s in range(0, min(len(warm_noisy), block * 25), block):
                pipe.process_block(warm_noisy[s:s + block])

    si_in, si_out, seg_out, corr_out = [], [], [], []
    pesq_in, pesq_out, stoi_out = [], [], []
    dns_sig, dns_bak, dns_ovrl = [], [], []
    align_lags: List[int] = []
    valid_counts: List[int] = []
    per_block_ms: List[float] = []
    total_audio = 0.0
    t_proc = 0.0       # wall-clock spent in the pipeline
    cpu_proc = 0.0     # CPU-time spent in the pipeline (process_time)
    hasher = hashlib.sha256()
    expected_delay = int(pipe.output_delay_samples)   # deferred tail + FIR group delay

    for idx, (clean, noisy) in enumerate(pairs):
        pipe.reset()
        out = np.zeros(len(noisy), dtype=np.float32)
        t0 = time.perf_counter()
        c0 = time.process_time()
        for s in range(0, len(noisy), block):
            seg = noisy[s:s + block]
            tb = time.perf_counter()
            ctx = pipe.process_block(seg)
            per_block_ms.append((time.perf_counter() - tb) * 1000.0)
            out[s:s + len(seg)] = ctx.audio[: len(seg)]
        t_proc += time.perf_counter() - t0
        cpu_proc += time.process_time() - c0
        total_audio += len(noisy) / sr
        hasher.update(np.ascontiguousarray(out, dtype=np.float32).tobytes())

        # Non-intrusive DNSMOS: scored on the FULL, unaligned output (the
        # constant delay and the un-emitted tail are irrelevant to it).
        if dnsmos_model:
            dm = M.dnsmos(out, sr, dnsmos_model)
            if dm is not None:
                dns_sig.append(dm["sig"]); dns_bak.append(dm["bak"]); dns_ovrl.append(dm["ovrl"])

        # Intrusive metrics: estimate the lag, reject impossible/ambiguous
        # alignments, and score ONLY the valid overlap — the block loop above
        # does not flush the deferred tail, so the last ``lag`` output samples
        # are never scored (and no zeros are ever scored against speech).
        est = estimate_lag(clean, out)
        align_lags.append(est.lag)
        al = resolve_lag(est, expected_delay)      # exact known delay, estimate as cross-check
        if not al.ok:
            res.rejected.append(f"pair {idx}: {al.reason}")
            continue
        ref_seg, out_seg = valid_overlap(clean, out, al.lag)
        s_ref, _ = overlap_slices(min(len(clean), len(out)), al.lag)
        noisy_seg = noisy[s_ref]              # same reference samples for the "in" numbers
        valid_counts.append(len(ref_seg))

        mi = M.all_metrics(ref_seg, noisy_seg, sr)
        mo = M.all_metrics(ref_seg, out_seg, sr)
        si_in.append(mi["si_sdr"]); si_out.append(mo["si_sdr"])
        seg_out.append(mo["seg_snr"]); corr_out.append(mo["corr"])
        if "pesq_wb" in mi: pesq_in.append(mi["pesq_wb"])
        if "pesq_wb" in mo: pesq_out.append(mo["pesq_wb"])
        if "stoi" in mo: stoi_out.append(mo["stoi"])

    q = res.quality
    if si_in:
        q["si_sdr_in"] = float(np.mean(si_in)); q["si_sdr_out"] = float(np.mean(si_out))
        q["si_sdri"] = q["si_sdr_out"] - q["si_sdr_in"]
        q["seg_snr_out"] = float(np.mean(seg_out)); q["corr_out"] = float(np.mean(corr_out))
    if pesq_in: q["pesq_in"] = float(np.mean(pesq_in))
    if pesq_out: q["pesq_out"] = float(np.mean(pesq_out))
    if stoi_out: q["stoi_out"] = float(np.mean(stoi_out))

    if dns_sig:
        res.dnsmos = {"sig": _agg(dns_sig), "bak": _agg(dns_bak), "ovrl": _agg(dns_ovrl)}

    if align_lags:
        la = np.asarray(align_lags)
        res.align_lag = {"known": int(expected_delay), "mean": float(np.mean(la)),
                         "min": int(np.min(la)), "max": int(np.max(la)),
                         "max_abs": int(np.max(np.abs(la)))}
    if valid_counts:
        vc = np.asarray(valid_counts, dtype=np.float64)
        res.valid_samples = {"mean": float(vc.mean()), "min": float(vc.min()),
                             "mean_s": float(vc.mean() / sr), "total_s": float(vc.sum() / sr)}
    res.out_hash = hasher.hexdigest()[:16]

    res.rtf = t_proc / max(total_audio, 1e-9)
    # CPU utilisation: CPU-seconds in the pipeline / wall-seconds in the
    # pipeline, ×100.  Can exceed 100% (DFN3 uses multiple threads).
    res.cpu_pct = 100.0 * cpu_proc / max(t_proc, 1e-9)
    arr = np.array(per_block_ms)
    res.latency_ms = {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "algorithmic": cfg.algorithmic_latency_ms,
    }
    res.ram_mb = _peak_rss_mb()
    return res


def summarize_runs(runs: Dict[str, BenchResult]) -> str:
    """Render a one-row-per-run comparison table (for SNR sweep / per-class)."""
    if not runs:
        return "(no runs)"
    has_dns = any(r.dnsmos for r in runs.values())
    has_pesq = any("pesq_out" in r.quality for r in runs.values())
    header = ["label", "n", "SI-SDRi", "corr"]
    if has_pesq:
        header.append("PESQ")
    if has_dns:
        header += ["DNSMOS-OVRL", "DNSMOS-BAK"]
    header += ["RTF", "CPU%"]
    widths = [14, 4, 8, 6] + ([6] if has_pesq else []) + ([12, 11] if has_dns else []) + [6, 6]

    def row(cells):
        return "  " + "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    lines = ["=" * 78, "  voiceiso BENCHMARK SUMMARY", "=" * 78, row(header), "-" * 78]
    for label, r in runs.items():
        q = r.quality
        cells = [label, r.n_pairs, f"{q.get('si_sdri', 0):+.2f}", f"{q.get('corr_out', 0):.3f}"]
        if has_pesq:
            cells.append(f"{q.get('pesq_out', float('nan')):.2f}")
        if has_dns:
            ovrl = r.dnsmos.get("ovrl", {}).get("mean", float("nan"))
            bak = r.dnsmos.get("bak", {}).get("mean", float("nan"))
            cells += [f"{ovrl:.2f}", f"{bak:.2f}"]
        cells += [f"{r.rtf:.3f}", f"{r.cpu_pct:.0f}"]
        lines.append(row(cells))
    lines.append("=" * 78)
    return "\n".join(lines)

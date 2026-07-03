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

Success criteria (targets for "approaching Krisp on CPU"):
  * SI-SDRi  ≥ +10 dB at 5 dB input SNR
  * PESQ-wb  ≥ 2.6
  * DNSMOS OVRL ≥ 3.0  (noisy ~2.5–3.0; good enhancement 3.0–3.3+)
  * RTF      ≤ 0.5
  * latency  ≤ 40 ms algorithmic
  * RAM      ≤ 300 MB resident
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from voiceiso.bench import metrics as M
from voiceiso.config import PipelineConfig
from voiceiso.pipeline import StreamingPipeline

SUCCESS = {
    "si_sdri_db": 10.0,
    "pesq_wb": 2.6,
    "dnsmos_ovrl": 3.0,
    "rtf": 0.5,
    "latency_ms": 40.0,
    "ram_mb": 300.0,
}


def _peak_rss_mb() -> float:
    import resource, sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024  # bytes vs KB


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

    def report(self) -> str:
        q = self.quality
        lines = ["=" * 60, f"  voiceiso BENCHMARK{(' — ' + self.label) if self.label else ''}", "=" * 60]
        lines.append(f"  backends           : {self.backends}")
        lines.append(f"  pairs              : {self.n_pairs}")
        if self.block_ms:
            tag = ("low-latency/low-quality" if self.block_ms <= 40
                   else "design-point (matches live)" if self.block_ms <= 120
                   else "max-quality/high-latency")
            lines.append(f"  block size         : {self.block_ms:.0f} ms  [{tag}]")
        for k in ("si_sdr_in", "si_sdr_out", "si_sdri", "seg_snr_out",
                  "corr_out", "pesq_in", "pesq_out", "stoi_out"):
            if k in q:
                lines.append(f"  {k:<18} : {q[k]:+.3f}")
        if self.dnsmos:
            for sub in ("sig", "bak", "ovrl"):
                if sub in self.dnsmos:
                    d = self.dnsmos[sub]
                    lines.append(f"  DNSMOS {sub.upper():<11} : "
                                 f"mean {d['mean']:.3f}  p50 {d['p50']:.3f}  p95 {d['p95']:.3f}")
        lines.append(f"  RTF                : {self.rtf:.3f}")
        lines.append(f"  CPU                : {self.cpu_pct:.0f} %")
        for k, v in self.latency_ms.items():
            lines.append(f"  latency {k:<10} : {v:.2f} ms")
        lines.append(f"  peak RAM           : {self.ram_mb:.0f} MB")
        lines.append("-" * 60)
        passed = (
            q.get("si_sdri", -9) >= SUCCESS["si_sdri_db"]
            and self.rtf <= SUCCESS["rtf"]
        )
        lines.append(f"  VERDICT: {'PASS' if passed else 'review'} "
                     f"(SI-SDRi≥{SUCCESS['si_sdri_db']}dB, RTF≤{SUCCESS['rtf']})")
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
    SI-SDRi is negative; see ARCHITECTURE.md §1.4).  Pass ``block_ms=20`` to see
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
    per_block_ms: List[float] = []
    total_audio = 0.0
    t_proc = 0.0       # wall-clock spent in the pipeline
    cpu_proc = 0.0     # CPU-time spent in the pipeline (process_time)

    for clean, noisy in pairs:
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

        mi = M.all_metrics(clean, noisy, sr)
        mo = M.all_metrics(clean, out, sr)
        si_in.append(mi["si_sdr"]); si_out.append(mo["si_sdr"])
        seg_out.append(mo["seg_snr"]); corr_out.append(mo["corr"])
        if "pesq_wb" in mi: pesq_in.append(mi["pesq_wb"])
        if "pesq_wb" in mo: pesq_out.append(mo["pesq_wb"])
        if "stoi" in mo: stoi_out.append(mo["stoi"])

        if dnsmos_model:
            dm = M.dnsmos(out, sr, dnsmos_model)
            if dm is not None:
                dns_sig.append(dm["sig"]); dns_bak.append(dm["bak"]); dns_ovrl.append(dm["ovrl"])

    q = res.quality
    q["si_sdr_in"] = float(np.mean(si_in)); q["si_sdr_out"] = float(np.mean(si_out))
    q["si_sdri"] = q["si_sdr_out"] - q["si_sdr_in"]
    q["seg_snr_out"] = float(np.mean(seg_out)); q["corr_out"] = float(np.mean(corr_out))
    if pesq_in: q["pesq_in"] = float(np.mean(pesq_in))
    if pesq_out: q["pesq_out"] = float(np.mean(pesq_out))
    if stoi_out: q["stoi_out"] = float(np.mean(stoi_out))

    if dns_sig:
        res.dnsmos = {"sig": _agg(dns_sig), "bak": _agg(dns_bak), "ovrl": _agg(dns_ovrl)}

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

"""
Benchmark harness.

Measures, over a set of (clean, noisy) pairs:
  * Quality:  SI-SDR(i), segmental SNR, correlation, PESQ-wb, STOI (if available)
  * Speed:    RTF, per-block latency p50/p95/p99
  * Resource: peak RSS (RAM)

Success criteria (targets for "approaching Krisp on CPU"):
  * SI-SDRi  ≥ +10 dB at 5 dB input SNR
  * PESQ-wb  ≥ 2.6   (noisy ~1.5–2.0)
  * RTF      ≤ 0.5   on one mid laptop core (headroom for the rest of the app)
  * latency  ≤ 40 ms algorithmic
  * RAM      ≤ 300 MB resident
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from voiceiso.bench import metrics as M
from voiceiso.config import PipelineConfig
from voiceiso.pipeline import StreamingPipeline

SUCCESS = {
    "si_sdri_db": 10.0,
    "pesq_wb": 2.6,
    "rtf": 0.5,
    "latency_ms": 40.0,
    "ram_mb": 300.0,
}


def _peak_rss_mb() -> float:
    import resource, sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024  # bytes vs KB


@dataclass
class BenchResult:
    quality: Dict[str, float] = field(default_factory=dict)
    rtf: float = 0.0
    latency_ms: Dict[str, float] = field(default_factory=dict)
    ram_mb: float = 0.0
    backends: Dict[str, object] = field(default_factory=dict)

    def report(self) -> str:
        q = self.quality
        lines = ["=" * 56, "  voiceiso BENCHMARK", "=" * 56]
        lines.append(f"  backends           : {self.backends}")
        for k in ("si_sdr_in", "si_sdr_out", "si_sdri", "seg_snr_out",
                  "corr_out", "pesq_in", "pesq_out", "stoi_out"):
            if k in q:
                lines.append(f"  {k:<18} : {q[k]:+.3f}")
        lines.append(f"  RTF                : {self.rtf:.3f}")
        for k, v in self.latency_ms.items():
            lines.append(f"  latency {k:<10} : {v:.2f} ms")
        lines.append(f"  peak RAM           : {self.ram_mb:.0f} MB")
        lines.append("-" * 56)
        passed = (
            q.get("si_sdri", -9) >= SUCCESS["si_sdri_db"]
            and self.rtf <= SUCCESS["rtf"]
        )
        lines.append(f"  VERDICT: {'PASS' if passed else 'review'} "
                     f"(SI-SDRi≥{SUCCESS['si_sdri_db']}dB, RTF≤{SUCCESS['rtf']})")
        lines.append("=" * 56)
        return "\n".join(lines)


def run_benchmark(pairs: List[Tuple[np.ndarray, np.ndarray]], sr: int = 48_000,
                  cfg: PipelineConfig | None = None) -> BenchResult:
    """``pairs`` = list of (clean, noisy) float32 arrays at ``sr``."""
    cfg = cfg or PipelineConfig(sample_rate=sr)
    pipe = StreamingPipeline(cfg)
    res = BenchResult(backends=pipe.backend_summary)

    si_in, si_out, seg_out, corr_out = [], [], [], []
    pesq_in, pesq_out, stoi_out = [], [], []
    per_block_ms: List[float] = []
    total_audio = 0.0
    t_proc = 0.0
    block = cfg.win * 5

    for clean, noisy in pairs:
        pipe.reset()
        out = np.zeros(len(noisy), dtype=np.float32)
        t0 = time.perf_counter()
        for s in range(0, len(noisy), block):
            seg = noisy[s:s + block]
            tb = time.perf_counter()
            ctx = pipe.process_block(seg)
            per_block_ms.append((time.perf_counter() - tb) * 1000.0)
            out[s:s + len(seg)] = ctx.audio[: len(seg)]
        t_proc += time.perf_counter() - t0
        total_audio += len(noisy) / sr

        mi = M.all_metrics(clean, noisy, sr)
        mo = M.all_metrics(clean, out, sr)
        si_in.append(mi["si_sdr"]); si_out.append(mo["si_sdr"])
        seg_out.append(mo["seg_snr"]); corr_out.append(mo["corr"])
        if "pesq_wb" in mi: pesq_in.append(mi["pesq_wb"])
        if "pesq_wb" in mo: pesq_out.append(mo["pesq_wb"])
        if "stoi" in mo: stoi_out.append(mo["stoi"])

    q = res.quality
    q["si_sdr_in"] = float(np.mean(si_in)); q["si_sdr_out"] = float(np.mean(si_out))
    q["si_sdri"] = q["si_sdr_out"] - q["si_sdr_in"]
    q["seg_snr_out"] = float(np.mean(seg_out)); q["corr_out"] = float(np.mean(corr_out))
    if pesq_in: q["pesq_in"] = float(np.mean(pesq_in))
    if pesq_out: q["pesq_out"] = float(np.mean(pesq_out))
    if stoi_out: q["stoi_out"] = float(np.mean(stoi_out))

    res.rtf = t_proc / max(total_audio, 1e-9)
    arr = np.array(per_block_ms)
    res.latency_ms = {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "algorithmic": cfg.algorithmic_latency_ms,
    }
    res.ram_mb = _peak_rss_mb()
    return res

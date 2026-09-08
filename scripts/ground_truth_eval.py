"""Ground-truth evaluation on REAL recordings (clean / noisy pairs).

The synthetic benchmark (`python -m voiceiso bench`) mixes white noise; this
tool scores the deployed pipeline against a user-made *clean* recording and
the *same take* with real noise mixed in (claps, plates, car …), which is how
the 2026-09-08 streaming fix and the rollback decision were measured.

Steps
-----
1. ``prep``   decode both files (any format afconvert/soundfile reads), align
              the noisy take to the clean one, extract the exact mix gain and
              the isolated noise track, and save a pair directory.
2. ``remix``  build a new pair from an existing one at a chosen SNR
              (speech-active power vs whole-file noise power), or a
              clean-only pair (damage measurement).
3. ``run``    stream a pair through the pipeline with optional config
              overrides / stage removals and score it: SI-SDRi, PESQ-WB,
              STOI (20 s segments), DNSMOS, impulsive-event cut in silence vs
              during speech, transient-kick count, RTF.  Appends one row to
              ``results.tsv`` in the pair root.

Example
-------
  python -m scripts.ground_truth_eval prep --noisy "Personal Test/audio demo.m4a" \
      --clean "Personal Test/pes jargon.m4a" --tag jargon
  python -m scripts.ground_truth_eval remix --tag jargon --snr 5
  python -m scripts.ground_truth_eval run --tag jargon --name baseline
  python -m scripts.ground_truth_eval run --tag jargon_snr5 --name nokick --cfg kick_enabled=False
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 48_000
BLK = 4_800
ROOT_DEFAULT = Path("Personal Test") / "gt"


# ── helpers ──────────────────────────────────────────────────────────────────
def _load48(path: Path) -> np.ndarray:
    """Decode to 48 kHz mono float32 (afconvert for m4a/aac, soundfile otherwise)."""
    if path.suffix.lower() in (".m4a", ".aac", ".mp3", ".caf", ".mp4"):
        tmp = path.with_suffix(".decoded48.wav")
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEF32@48000", str(path), str(tmp)],
                       check=True, capture_output=True)
        x, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
        tmp.unlink(missing_ok=True)
    else:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x.astype(np.float32)


def _levels(v: np.ndarray, fr: int = SR // 100) -> np.ndarray:
    nb = len(v) // fr
    return 20 * np.log10(np.sqrt((v[: nb * fr].reshape(nb, fr) ** 2).mean(1)) + 1e-12)


def _events(nd: np.ndarray, thresh_db: float = 15.0):
    """Impulsive events in a level track: ≥ thresh above the local (±0.5 s) median."""
    med = np.array([np.median(nd[max(0, i - 50): i + 50]) for i in range(len(nd))])
    ev = np.where(nd - med >= thresh_db)[0]
    out = []
    for i in ev:
        if out and i - out[-1][1] <= 5:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def _save_pair(d: Path, noisy, clean, noise) -> None:
    d.mkdir(parents=True, exist_ok=True)
    sf.write(str(d / "noisy48.wav"), noisy.astype(np.float32), SR)
    sf.write(str(d / "clean_scaled48.wav"), clean.astype(np.float32), SR)
    sf.write(str(d / "noise48.wav"), noise.astype(np.float32), SR)


# ── prep ─────────────────────────────────────────────────────────────────────
def cmd_prep(a) -> int:
    from voiceiso.bench.benchmark import estimate_lag, valid_overlap
    x, c = _load48(Path(a.noisy)), _load48(Path(a.clean))
    al = estimate_lag(c, x)
    if not al.ok:
        print(f"[{a.tag}] ALIGNMENT REJECTED — {al.reason}")
        return 2
    # Crop BOTH takes to their valid overlap (no zero padding anywhere).
    c, xa = valid_overlap(c, x, al.lag)
    xa = xa.astype(np.float64); c = c.astype(np.float64); L = len(c); lag = al.lag
    alpha = float(np.dot(xa, c) / np.dot(c, c))
    s = alpha * c; n = xa - s
    snr = 10 * np.log10(np.sum(s ** 2) / np.sum(n ** 2))
    ev = _events(_levels(n)); sd = _levels(s)
    n_sp = sum(1 for p, q in ev if sd[p:q + 1].max() > -35)
    _save_pair(Path(a.root) / a.tag, xa, s, n)
    print(f"[{a.tag}] {L / SR:.1f}s valid overlap  lag {lag:+d} samples (corr {al.corr:.3f}, "
          f"peak ratio {al.peak_ratio:.2f})  mix gain {alpha:+.3f}  mix SNR {snr:+.1f} dB  "
          f"impulsive events {len(ev)} ({n_sp} during speech)")
    if abs(np.dot(xa, s) / np.linalg.norm(xa) / np.linalg.norm(s)) < 0.3:
        print("  WARNING: weak correlation — is the clean file really the same take?")
    return 0


# ── remix ────────────────────────────────────────────────────────────────────
def cmd_remix(a) -> int:
    d = Path(a.root) / a.tag
    s, _ = sf.read(str(d / "clean_scaled48.wav"), dtype="float32")
    n, _ = sf.read(str(d / "noise48.wav"), dtype="float32")
    L = min(len(s), len(n)); s, n = s[:L], n[:L]
    if a.clean_only:
        _save_pair(Path(a.root) / f"{a.tag}_clean", s, s, np.zeros_like(s))
        print(f"built {a.tag}_clean (damage-only pair)"); return 0
    sd = _levels(s); act = sd > -40
    ps = np.mean(s[: len(sd) * (SR // 100)].reshape(-1, SR // 100)[act] ** 2); pn = np.mean(n ** 2)
    g = np.sqrt(ps / (pn * 10 ** (a.snr / 10)))
    _save_pair(Path(a.root) / f"{a.tag}_snr{int(a.snr)}", s + g * n, s, g * n)
    print(f"built {a.tag}_snr{int(a.snr)} (noise gain {20 * np.log10(g):+.1f} dB vs the original mix)")
    return 0


# ── run ──────────────────────────────────────────────────────────────────────
def cmd_run(a) -> int:
    from pesq import pesq
    from pystoi import stoi
    from voiceiso.bench import metrics as M
    from voiceiso.bench.benchmark import estimate_lag, overlap_slices, resolve_lag, valid_overlap
    from voiceiso.config import PipelineConfig
    from voiceiso.pipeline import StreamingPipeline

    d = Path(a.root) / a.tag
    x, _ = sf.read(str(d / "noisy48.wav"), dtype="float32")
    s, _ = sf.read(str(d / "clean_scaled48.wav"), dtype="float32")
    n, _ = sf.read(str(d / "noise48.wav"), dtype="float32")
    L = min(len(x), len(s), len(n)); x, s, n = x[:L], s[:L], n[:L]

    cfg = PipelineConfig()
    for kv in a.cfg:
        k, v = kv.split("=", 1); cur = getattr(cfg, k)
        if isinstance(cur, bool): v = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int): v = int(v)
        elif isinstance(cur, float): v = float(v)
        setattr(cfg, k, v)
    pipe = StreamingPipeline(cfg)
    if a.remove:
        keep = [st for st in pipe.stages if not any(r.lower() in type(st).__name__.lower() for r in a.remove)]
        print("dropped stages:", [type(st).__name__ for st in pipe.stages if st not in keep])
        pipe.stages = keep
    pipe.reset(); pipe.warmup()

    out = np.zeros_like(x); kicks = 0; t0 = time.time()
    for i in range(0, (L // BLK) * BLK, BLK):
        ctx = pipe.process_block(x[i:i + BLK].copy())
        out[i:i + BLK] = ctx.audio[:BLK]
        kicks += int((ctx.meta or {}).get("transient_kick", 0))
    rtf = (time.time() - t0) / (L / SR)
    (d / "out").mkdir(exist_ok=True)
    sf.write(str(d / "out" / f"{a.name}.wav"), out, SR)

    # Intrusive metrics on the VALID overlap only (the block loop above does not
    # flush the deferred tail; the un-emitted final samples are never scored,
    # and no zero padding is ever compared with reference speech).  Reject
    # impossible / ambiguous alignments instead of scoring them.
    est_al = estimate_lag(s, out)
    al = resolve_lag(est_al, pipe.output_delay_samples)   # exact known delay; estimate cross-checks it
    if not al.ok:
        print(f"[{a.tag}/{a.name}] ALIGNMENT REJECTED — {al.reason}")
        return 2
    ref, est = valid_overlap(s, out, al.lag)
    s_ref, _ = overlap_slices(min(len(s), len(out)), al.lag)
    x_seg, n_seg = x[s_ref], n[s_ref]          # same reference indices for the "in" numbers
    si_in, si_out = M.si_sdr(ref, x_seg), M.si_sdr(ref, est)
    r16, y16, x16 = (resample_poly(v, 1, 3) for v in (ref, est, x_seg))
    seg = 16_000 * 20; P, P0, T, T0 = [], [], [], []
    for i in range(0, len(r16) - seg + 1, seg):
        try:
            P.append(pesq(16_000, r16[i:i + seg], y16[i:i + seg], "wb"))
            P0.append(pesq(16_000, r16[i:i + seg], x16[i:i + seg], "wb"))
        except Exception:
            pass
        T.append(stoi(r16[i:i + seg], y16[i:i + seg], 16_000)); T0.append(stoi(r16[i:i + seg], x16[i:i + seg], 16_000))
    dn = M.dnsmos(out, SR, cfg.dnsmos_model_path) or {}   # non-intrusive: full, unaligned output

    nd, sd = _levels(n_seg), _levels(ref); rd = _levels(est - ref)
    cut_sp, cut_sil = [], []
    for p, q in _events(nd):
        cut = nd[p:q + 1].max() - rd[p:q + 3].max()
        (cut_sp if sd[p:q + 1].max() > -35 else cut_sil).append(cut)
    row = dict(pair=a.tag, name=a.name, lag=al.lag, lag_est=est_al.lag, n_valid=len(ref), sisdri=round(si_out - si_in, 2),
               pesq=round(float(np.mean(P)), 3) if P else None, dpesq=round(float(np.mean(P)) - float(np.mean(P0)), 3) if P else None,
               stoi=round(float(np.mean(T)), 3), dstoi=round(float(np.mean(T)) - float(np.mean(T0)), 3),
               sig=round(dn.get("sig", float("nan")), 2), bak=round(dn.get("bak", float("nan")), 2), ovrl=round(dn.get("ovrl", float("nan")), 2),
               cut_speech=round(float(np.median(cut_sp)), 1) if cut_sp else None, n_sp=len(cut_sp),
               cut_silence=round(float(np.median(cut_sil)), 1) if cut_sil else None, n_sil=len(cut_sil),
               kicks=kicks, rtf=round(rtf, 2))
    print("\t".join(f"{k}={v}" for k, v in row.items()))
    res = Path(a.root) / "results.tsv"; new = not res.exists()
    with open(res, "a") as f:
        if new: f.write("\t".join(row.keys()) + "\n")
        f.write("\t".join(str(v) for v in row.values()) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT_DEFAULT), help="pair root (default: 'Personal Test/gt')")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--noisy", required=True); p.add_argument("--clean", required=True); p.add_argument("--tag", required=True); p.set_defaults(fn=cmd_prep)
    r = sub.add_parser("remix"); r.add_argument("--tag", required=True); r.add_argument("--snr", type=float, default=5.0); r.add_argument("--clean-only", action="store_true"); r.set_defaults(fn=cmd_remix)
    u = sub.add_parser("run"); u.add_argument("--tag", required=True); u.add_argument("--name", required=True)
    u.add_argument("--cfg", action="append", default=[], help="key=value PipelineConfig override (repeatable)")
    u.add_argument("--remove", action="append", default=[], help="stage class-name substring to drop (repeatable)")
    u.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

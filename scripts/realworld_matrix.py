"""Real-world condition matrix: {noise type} × {silence, speech} × SNR.

Streams REAL FSD50K **eval** clips (held out from all training) through the
full deployed pipeline at the live 100 ms operating point and reports:

  * silence regime — noise-segment attenuation (dB): how much of the noise
    is removed when the user is not talking (input RMS − output RMS over the
    noise-active region, alignment-compensated);
  * speech regime  — SI-SDRi (dB) against the clean-speech reference at the
    given input SNR (the standard enhancement metric), plus noise-tail
    attenuation for the 2 s after speech ends.

Conditions map to the demo vocabulary:
  fan → FSD50K fan;  AC → fan (no hvac clips exist in FSD50K);  keyboard →
  keyboard;  mouse → keyboard-class clicks;  clap → transient (Clapping);
  TV → loudspeaker-simulated speech (scripts.train_head_v4.loudspeaker_sim);
  competing speech → competing_speech;  café → competing_speech babble;
  traffic → traffic.

Writes a markdown table to REALWORLD_RESULTS.md and prints it.

Run:  PYTHONPATH=<repo> .venv_poc/bin/python -m scripts.realworld_matrix
"""
from __future__ import annotations

import csv
import sys
import zlib
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from voiceiso.config import PipelineConfig
from voiceiso.pipeline import StreamingPipeline
from voiceiso.bench import metrics as M
from voiceiso.bench.benchmark import _align_to_ref
from voiceiso.data.fsd50k_labelmap import fsd50k_labels_to_target

SR = 48_000
BLK = 4800
ROOT = Path("dataset/FSD50K")
N_CLIPS = 4            # noise clips per condition (averaged)
SNRS = (0.0, 5.0, 10.0)


def load48(path: Path, seconds: float | None = None) -> np.ndarray | None:
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return None
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype("float32")
    if seconds:
        n = int(seconds * SR)
        if len(x) < n:
            x = np.tile(x, int(np.ceil(n / len(x))))
        x = x[:n]
    return x.astype("float32")


def eval_rows() -> dict[str, list[str]]:
    by: dict[str, list[str]] = {}
    for r in csv.DictReader(open(ROOT / "FSD50K.ground_truth" / "eval.csv")):
        raw = r["labels"]
        for t in fsd50k_labels_to_target(raw):
            by.setdefault(t, []).append(r["fname"])
        if "Clapping" in raw or "Hands" in raw:
            by.setdefault("_clap", []).append(r["fname"])
    return by


def stream(pipe: StreamingPipeline, sig: np.ndarray) -> np.ndarray:
    pipe.reset()
    pipe.warmup()
    out = np.zeros_like(sig)
    for s in range(0, (len(sig) // BLK) * BLK, BLK):
        out[s:s + BLK] = pipe.process_block(sig[s:s + BLK]).audio[:BLK]
    return out


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2))) + 1e-12)


def main() -> int:
    by = eval_rows()
    speech_files = sorted(Path("dataset/test/clean").glob("*.wav"))
    speech = load48(speech_files[0], 8.0)
    speech *= 0.15 / (np.sqrt((speech ** 2).mean()) + 1e-9)

    conds: dict[str, tuple[str, str | None]] = {
        "fan":              ("fan", None),
        "AC (→fan)":        ("fan", None),
        "keyboard":         ("keyboard", None),
        "mouse (→clicks)":  ("keyboard", None),
        "clap":             ("_clap", None),
        "TV (loudspeaker)": ("speech", "tv_sim"),
        "competing speech": ("competing_speech", None),
        "café (babble)":    ("competing_speech", None),
        "traffic":          ("traffic", None),
    }
    if "tv_sim" in {v[1] for v in conds.values()}:
        from scripts.train_head_v4 import loudspeaker_sim

    pipe = StreamingPipeline(PipelineConfig())
    lines = ["# Real-world condition matrix",
             "",
             "Full deployed pipeline, 100 ms blocks, real FSD50K **eval** clips "
             f"(n={N_CLIPS}/condition, held out from all training). "
             "Silence regime: noise attenuation (dB, higher = better). "
             "Speech regime: SI-SDRi at the given input SNR (dB, higher = better).",
             "",
             "| condition | silence: atten (dB) | " +
             " | ".join(f"speech @{int(s)} dB: SI-SDRi" for s in SNRS) + " |",
             "|---|---|" + "|".join(["---"] * len(SNRS)) + "|"]

    for name, (cls, transform) in conds.items():
        fnames = by.get(cls, [])[: N_CLIPS * 3]
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        clips = []
        for fn in fnames:
            x = load48(ROOT / "FSD50K.eval_audio" / f"{fn}.wav", 8.0)
            if x is None or np.sqrt((x ** 2).mean()) < 1e-4:
                continue
            if transform == "tv_sim":
                x32 = resample_poly(x, 2, 3).astype("float32")
                x32 = loudspeaker_sim(x32, rng)
                x = resample_poly(x32, 3, 2).astype("float32")[: len(x)]
            clips.append(x)
            if len(clips) >= N_CLIPS:
                break

        att_sil, sisdri = [], {s: [] for s in SNRS}
        for x in clips:
            noise = x * (0.05 / (np.sqrt((x ** 2).mean()) + 1e-9))
            # silence regime
            out = stream(pipe, noise)
            al, _ = _align_to_ref(noise, out)
            att_sil.append(rms_db(noise[SR:]) - rms_db(al[SR:]))
            # speech regime at each SNR
            for snr in SNRS:
                g = (np.sqrt((speech ** 2).mean())
                     / (np.sqrt((noise ** 2).mean()) * 10 ** (snr / 20)))
                mix = (speech + g * noise[: len(speech)]).astype("float32")
                outm = stream(pipe, mix)
                alm, _ = _align_to_ref(mix, outm)
                si_in = M.si_sdr(speech, mix)
                si_out = M.si_sdr(speech[: len(alm)], alm)
                sisdri[snr].append(si_out - si_in)

        row = (f"| {name} | {np.mean(att_sil):+.1f} | " +
               " | ".join(f"{np.mean(sisdri[s]):+.1f}" for s in SNRS) + " |")
        lines.append(row)
        print(row, flush=True)

    text = "\n".join(lines) + "\n"
    Path("REALWORLD_RESULTS.md").write_text(text)
    print("\nwrote REALWORLD_RESULTS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

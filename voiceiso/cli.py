"""
voiceiso command-line interface.

    python -m voiceiso enhance  in.wav [out.wav]   # offline file enhancement
    python -m voiceiso live     [--duration S]     # mic → pipeline → speakers
    python -m voiceiso bench    [--snr 5] [--n 10] [--data data]
    python -m voiceiso info                         # show backends / latency
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.pipeline import StreamingPipeline


def _read(path: str):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    return x, sr


def cmd_enhance(args) -> None:
    import soundfile as sf
    x, sr = _read(args.input)
    pipe = StreamingPipeline(PipelineConfig(sample_rate=sr))
    print("backends:", pipe.backend_summary)
    y = pipe.process_signal(x)
    out = args.output or str(Path(args.input).with_name(Path(args.input).stem + "_enhanced.wav"))
    sf.write(out, y, sr)
    print(f"wrote {out}")


def cmd_live(args) -> None:
    from voiceiso.io.audio_stream import LiveStream
    LiveStream(PipelineConfig()).run(duration_s=args.duration)


def cmd_bench(args) -> None:
    from voiceiso.bench.benchmark import run_benchmark
    from voiceiso.data.dynamic_mixer import DynamicMixer
    import glob

    pairs = []
    mixer = DynamicMixer(data_root=args.data, snr_range=(args.snr, args.snr))
    if mixer.available():
        pairs = mixer.build_benchmark_set(args.n)
        print(f"built {len(pairs)} dynamic pairs @ {args.snr} dB SNR")
    else:
        # Fallback: synthesise from any local clean speech + white/transient noise.
        clean_files = sorted(glob.glob("dataset/test/clean/*.wav"))[: args.n]
        if not clean_files:
            print("No corpora found under", args.data, "and no dataset/test/clean/*.wav")
            sys.exit(1)
        rng = np.random.default_rng(0)
        for f in clean_files:
            c, sr = _read(f)
            sp = np.mean(c ** 2)
            noise = rng.normal(0, np.sqrt(sp / 10 ** (args.snr / 10)), len(c)).astype("float32")
            pairs.append((c, (c + noise).astype("float32")))
        print(f"built {len(pairs)} synthetic pairs @ {args.snr} dB SNR (fallback)")

    res = run_benchmark(pairs, sr=48_000)
    print(res.report())


def cmd_info(args) -> None:
    pipe = StreamingPipeline(PipelineConfig())
    print("backends:", pipe.backend_summary)
    print("stages  :", [s.name for s in pipe.stages])


def main() -> None:
    p = argparse.ArgumentParser(prog="voiceiso", description="CPU voice-isolation pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enhance"); e.add_argument("input"); e.add_argument("output", nargs="?")
    e.set_defaults(func=cmd_enhance)

    l = sub.add_parser("live"); l.add_argument("--duration", type=float, default=0.0)
    l.set_defaults(func=cmd_live)

    b = sub.add_parser("bench")
    b.add_argument("--snr", type=float, default=5.0); b.add_argument("--n", type=int, default=10)
    b.add_argument("--data", default="data"); b.set_defaults(func=cmd_bench)

    i = sub.add_parser("info"); i.set_defaults(func=cmd_info)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

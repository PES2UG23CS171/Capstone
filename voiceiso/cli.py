"""
voiceiso command-line interface.

    python -m voiceiso enhance  in.wav [out.wav]   # offline file enhancement
    python -m voiceiso live     [--duration S]     # mic → pipeline → speakers
    python -m voiceiso bench    [--snr 5] [--n 10] [--data data]
                                [--sweep] [--per-class] [--dnsmos-model PATH]
    python -m voiceiso enroll   in.wav [--out vec.npy]  # target-speaker enrollment
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
    try:
        x, sr = _read(args.input)
    except Exception as exc:  # noqa: BLE001 — unreadable/corrupt/non-audio input
        sys.exit(f"error: cannot read {args.input!r} as audio "
                 f"({type(exc).__name__}: {exc})")
    if len(x) == 0:
        sys.exit(f"error: {args.input!r} contains no audio samples")
    # DFN3 is a 48 kHz model — resample arbitrary user WAVs in, process at
    # 48 kHz, resample back out (previously any non-48 kHz file crashed).
    target_sr = 48_000
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        print(f"resampling {sr} Hz → {target_sr} Hz for DFN3 (output written at {sr} Hz)")
        x48 = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
    else:
        x48 = x
    pipe = StreamingPipeline(PipelineConfig())
    print("backends:", pipe.backend_summary)
    y = pipe.process_signal(x48)
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        y = resample_poly(y, sr // g, target_sr // g).astype(np.float32)
        y = y[: len(x)]
    out = args.output or str(Path(args.input).with_name(Path(args.input).stem + "_enhanced.wav"))
    sf.write(out, np.clip(y, -1.0, 1.0), sr)
    print(f"wrote {out}")


def cmd_live(args) -> None:
    from voiceiso.io.audio_stream import LiveStream
    LiveStream(PipelineConfig(),
               input_device=args.input_device,
               output_device=args.output_device).run(duration_s=args.duration)


def cmd_devices(args) -> None:
    """List audio devices; flag virtual devices usable as a call-mode sink."""
    import sounddevice as sd
    virtual_markers = ("blackhole", "loopback", "vb-cable", "soundflower")
    found_virtual = False
    for i, d in enumerate(sd.query_devices()):
        tags = []
        if d["max_input_channels"]:
            tags.append(f"in:{d['max_input_channels']}")
        if d["max_output_channels"]:
            tags.append(f"out:{d['max_output_channels']}")
        v = any(m in d["name"].lower() for m in virtual_markers)
        found_virtual |= v
        print(f"  [{i:2d}] {d['name']:<40s} {' '.join(tags):<12s}"
              f" {int(d['default_samplerate'])} Hz{'   <- VIRTUAL (call-mode sink)' if v else ''}")
    if not found_virtual:
        print("\nNo virtual audio device found. For call mode (Google Meet etc.) "
              "install BlackHole:\n    brew install blackhole-2ch\n"
              "then: python -m voiceiso live --output BlackHole  (and select "
              "'BlackHole 2ch' as the microphone in the conferencing app).")


# SNR sweep points (dB) and the noise classes for per-class evaluation.
_SWEEP_SNRS = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
_PER_CLASS = ("fan", "hvac", "traffic", "keyboard", "dog_bark",
              "music", "television", "competing_speech")


def _synth_banner(data_root: str) -> None:
    """Loudly flag that we are NOT using real noise corpora."""
    print("=" * 70)
    print("!! SYNTHETIC BENCHMARK — white-noise mixes from dataset/test/clean !!")
    print(f"!! No labelled noise corpora found under {data_root!r}.")
    print("!! These numbers are an EASIER, additive-white-noise condition and are")
    print("!! NOT comparable to the real-corpora SI-SDRi/PESQ figures in the docs.")
    print("!! Provide DNS5/MUSAN/DEMAND/WHAM/FSD50K under --data for real mixes.")
    print("=" * 70)


def _synth_pairs(n: int, snr: float):
    """Fallback synthetic (clean, white-noise) pairs from dataset/test/clean."""
    import glob
    clean_files = sorted(glob.glob("dataset/test/clean/*.wav"))[:n]
    if not clean_files:
        return None
    rng = np.random.default_rng(0)
    pairs = []
    for f in clean_files:
        c, _sr = _read(f)
        sp = np.mean(c ** 2)
        noise = rng.normal(0, np.sqrt(sp / 10 ** (snr / 10)), len(c)).astype("float32")
        pairs.append((c, (c + noise).astype("float32")))
    return pairs


def cmd_bench(args) -> None:
    from voiceiso.bench.benchmark import run_benchmark, summarize_runs
    from voiceiso.data.dynamic_mixer import DynamicMixer

    cfg = PipelineConfig()
    dnsmos_model = args.dnsmos_model or cfg.dnsmos_model_path
    if dnsmos_model:
        print(f"DNSMOS model: {dnsmos_model}")
    else:
        print("DNSMOS: disabled — model not shipped. Run "
              "`python -m scripts.fetch_dnsmos` to download sig_bak_ovr.onnx "
              "into checkpoints/ (auto-wired thereafter), or pass --dnsmos-model PATH.")

    try:
        mixer = DynamicMixer(data_root=args.data, snr_range=(args.snr, args.snr))
        have_corpora = mixer.available()
    except Exception:
        mixer = None
        have_corpora = False

    # ── SNR sweep ────────────────────────────────────────────────────────
    if args.sweep:
        runs = {}
        for snr in _SWEEP_SNRS:
            if have_corpora:
                m = DynamicMixer(data_root=args.data, snr_range=(snr, snr))
                pairs = m.build_benchmark_set(args.n)
                src = "dynamic"
            else:
                pairs = _synth_pairs(args.n, snr)
                src = "synthetic"
            if not pairs:
                print("No corpora and no dataset/test/clean/*.wav — cannot bench.")
                sys.exit(1)
            label = f"{snr:+.0f}dB" + ("" if have_corpora else "·synth")
            print(f"  [{src}] {label}: {len(pairs)} pairs")
            runs[label] = run_benchmark(pairs, sr=48_000, dnsmos_model=dnsmos_model,
                                        label=label, block_ms=args.block_ms)
        print(summarize_runs(runs))
        return

    # ── Per-class evaluation ─────────────────────────────────────────────
    if args.per_class:
        if not have_corpora:
            print("Per-class evaluation requires labelled noise corpora under",
                  args.data, "— none found.")
            sys.exit(1)
        avail = mixer.available_classes()
        runs = {}
        for cls in _PER_CLASS:
            n_files = avail.get(cls, 0)
            if n_files == 0:
                print(f"  skip {cls}: no matching noise files")
                continue
            m = DynamicMixer(data_root=args.data, snr_range=(args.snr, args.snr))
            pairs = m.build_benchmark_set(args.n, noise_class=cls)
            print(f"  {cls}: {len(pairs)} pairs from {n_files} noise files @ {args.snr} dB")
            runs[cls] = run_benchmark(pairs, sr=48_000, dnsmos_model=dnsmos_model,
                                      label=cls, block_ms=args.block_ms)
        if not runs:
            print("No per-class noise files matched any of:", ", ".join(_PER_CLASS))
            sys.exit(1)
        print(summarize_runs(runs))
        return

    # ── Single-SNR run (default) ─────────────────────────────────────────
    if have_corpora:
        pairs = mixer.build_benchmark_set(args.n)
        print(f"built {len(pairs)} dynamic pairs @ {args.snr} dB SNR")
        label = f"{args.snr:+.0f}dB"
    else:
        _synth_banner(args.data)
        pairs = _synth_pairs(args.n, args.snr)
        if not pairs:
            print("No corpora found under", args.data, "and no dataset/test/clean/*.wav")
            sys.exit(1)
        print(f"built {len(pairs)} SYNTHETIC white-noise pairs @ {args.snr} dB SNR")
        label = f"{args.snr:+.0f}dB·SYNTH"

    res = run_benchmark(pairs, sr=48_000, dnsmos_model=dnsmos_model,
                        label=label, block_ms=args.block_ms)
    print(res.report())


def cmd_enroll(args) -> None:
    """Compute the target speaker's x-vector from a clip and persist it.

    The pipeline can use this enrolled embedding for two related features:
      * CompetingSpeech / TargetSpeakerMask gating (cosine similarity)
      * VoiceFilter-Lite mask conditioning

    Requires an ECAPA-TDNN ONNX model at ``cfg.speaker_model_path``.  If the
    model isn't available the command refuses cleanly rather than persisting a
    bogus vector.
    """
    from voiceiso.stages.speaker_embedder import SpeakerEmbedder

    audio, sr = _read(args.input)
    duration_s = len(audio) / float(sr) if sr > 0 else 0.0
    if duration_s < 3.0:
        print(f"warning: enrollment clip is only {duration_s:.1f}s — "
              "10s+ of clean speech is recommended.")

    # Pick the output path: --out overrides cfg.speaker_enroll_path.  If neither
    # is set, default to checkpoints/speaker.npy under cwd.
    cfg = PipelineConfig()
    out_path = args.out or cfg.speaker_enroll_path or "checkpoints/speaker.npy"
    cfg.speaker_enroll_path = out_path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    embedder = SpeakerEmbedder(cfg)
    if embedder.backend == "passthrough":
        model_path = cfg.speaker_model_path or "(unset)"
        print(f"ERROR: ECAPA-TDNN model not available (cfg.speaker_model_path = "
              f"{model_path!r}).", file=sys.stderr)
        print("  Set cfg.speaker_model_path to an ECAPA-TDNN ONNX file before "
              "enrolling.", file=sys.stderr)
        sys.exit(2)

    ok = embedder.enroll_from_audio(audio, sr)
    if not ok:
        print("ERROR: enrollment failed (embedding norm too small — silence input?)",
              file=sys.stderr)
        sys.exit(2)
    print(f"enrolled: wrote 192-dim x-vector → {out_path}")


def cmd_info(args) -> None:
    pipe = StreamingPipeline(PipelineConfig())
    print("backends:", pipe.backend_summary)
    print("stages  :", [s.name for s in pipe.stages])


def main() -> None:
    p = argparse.ArgumentParser(prog="voiceiso", description="CPU voice-isolation pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enhance"); e.add_argument("input"); e.add_argument("output", nargs="?")
    e.set_defaults(func=cmd_enhance)

    l = sub.add_parser("live", help="mic → pipeline → output device")
    l.add_argument("--duration", type=float, default=0.0)
    l.add_argument("--input", dest="input_device", default=None,
                   help="input device (index or name substring, e.g. 'MacBook')")
    l.add_argument("--output", dest="output_device", default=None,
                   help="output device (index or name substring). Call mode: "
                        "'BlackHole' → then pick BlackHole as the mic in Meet/Zoom")
    l.set_defaults(func=cmd_live)

    d = sub.add_parser("devices", help="list audio devices (flags call-mode sinks)")
    d.set_defaults(func=cmd_devices)

    b = sub.add_parser("bench")
    b.add_argument("--snr", type=float, default=5.0, help="input SNR (dB) for single / per-class runs")
    b.add_argument("--n", type=int, default=10, help="pairs per condition")
    b.add_argument("--data", default="data", help="corpora root for DynamicMixer")
    b.add_argument("--sweep", action="store_true",
                   help="run the SNR sweep (-10..15 dB) and print a summary table")
    b.add_argument("--per-class", dest="per_class", action="store_true",
                   help="evaluate per noise class (fan, hvac, traffic, keyboard, …)")
    b.add_argument("--dnsmos-model", dest="dnsmos_model", default=None,
                   help="path to DNSMOS sig_bak_ovr.onnx (enables SIG/BAK/OVRL)")
    b.add_argument("--block-ms", dest="block_ms", type=float, default=100.0,
                   help="pipeline block size in ms (default 100 = live design "
                        "point; use 20 for the low-latency/low-quality point)")
    b.set_defaults(func=cmd_bench)

    en = sub.add_parser("enroll", help="enrol the target speaker from a clip")
    en.add_argument("input", help="WAV file with ~10 s of clean target speech")
    en.add_argument("--out", help="output .npy path (defaults to cfg.speaker_enroll_path)")
    en.set_defaults(func=cmd_enroll)

    i = sub.add_parser("info"); i.set_defaults(func=cmd_info)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

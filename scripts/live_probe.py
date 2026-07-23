"""Live diagnostic probe — records what the pipeline actually sees and does.

Opens the SAME audio configuration as the desktop app (48 kHz, 100 ms blocks,
default devices), runs the SAME StreamingPipeline with the same warm-up, and
captures three artefacts into diag_capture/:

    probe_in.wav    exactly what the microphone delivered to the pipeline
    probe_out.wav   exactly what the pipeline emitted (what you should hear)
    probe_log.txt   per-block decision state (label/SNR/suppression/VAD/…)

Run it, make the sounds that are leaking (fan / claps / finger snaps /
talking), then hand the three files over for analysis.

    .venv_poc/bin/python -m scripts.live_probe --seconds 25

The processed audio is also played to the output device so you can listen
while it records (use headphones).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--out", default="diag_capture")
    ap.add_argument("--monitor", action="store_true", default=True,
                    help="play processed audio while recording (default on)")
    args = ap.parse_args()

    import sounddevice as sd
    import soundfile as sf
    from voiceiso.config import PipelineConfig
    from voiceiso.pipeline import StreamingPipeline

    sr, blk = 48_000, 4800
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    print("devices:", flush=True)
    print(f"  input : {sd.query_devices(sd.default.device[0])['name']}")
    print(f"  output: {sd.query_devices(sd.default.device[1])['name']}")

    pipe = StreamingPipeline(PipelineConfig())
    print("backends:", pipe.backend_summary, flush=True)
    if pipe.backend_summary.get("enhancement") != "deepfilternet3":
        print("!! DFN3 NOT LOADED — this probe would record passthrough only. Aborting.")
        return 1
    print("warming up…", flush=True)
    pipe.warmup()

    n_blocks = int(np.ceil(args.seconds * sr / blk))
    rec_in = np.zeros(n_blocks * blk, dtype=np.float32)
    rec_out = np.zeros(n_blocks * blk, dtype=np.float32)
    log_lines: list[str] = []
    state = {"i": 0, "xruns": 0, "clip": 0}

    def cb(indata, outdata, frames, time_info, status):
        i = state["i"]
        if i >= n_blocks:
            outdata[:] = 0
            raise sd.CallbackStop
        if status:
            state["xruns"] += 1
        x = indata[:, 0].copy()
        if np.max(np.abs(x)) >= 0.999:
            state["clip"] += 1
        t0 = time.perf_counter()
        ctx = pipe.process_block(x)
        rtf = (time.perf_counter() - t0) / (frames / sr)
        y = ctx.audio[:frames].astype(np.float32)
        if len(y) < frames:
            y = np.concatenate([y, np.zeros(frames - len(y), dtype=np.float32)])
        rec_in[i * blk:(i + 1) * blk] = x[:blk]
        rec_out[i * blk:(i + 1) * blk] = y[:blk]
        m = ctx.meta
        log_lines.append(
            f"t={i/10:5.1f}s in_pk={20*np.log10(np.max(np.abs(x))+1e-12):6.1f} "
            f"out_pk={20*np.log10(np.max(np.abs(y))+1e-12):6.1f} "
            f"label={m.get('noise_label')!s:8s} conf={m.get('noise_conf',0):.2f} "
            f"snr={ctx.snr_db:5.1f} supp={ctx.suppression:.2f} wet={m.get('enh_wet',0):.0f} "
            f"atten={m.get('enh_atten_db',0):5.1f} vad={ctx.vad_prob:.2f} sp={int(ctx.is_speech)} "
            f"band=({ctx.band_gain[0]:.2f},{ctx.band_gain[1]:.2f},{ctx.band_gain[2]:.2f}) "
            f"pf={ctx.postfilter_strength:.2f} rb={m.get('enh_rollback',0):.2f} rtf={rtf:.2f}")
        outdata[:, 0] = y if args.monitor else 0.0
        if outdata.shape[1] > 1:
            outdata[:, 1] = outdata[:, 0]
        state["i"] = i + 1

    print(f"\nRECORDING {args.seconds:.0f}s — make the leaking sounds now "
          f"(fan… clap… snap… talk…). Ctrl-C stops early.", flush=True)
    try:
        with sd.Stream(samplerate=sr, blocksize=blk, channels=(1, 2),
                       dtype="float32", callback=cb, latency="high"):
            while state["i"] < n_blocks:
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass

    n = state["i"] * blk
    sf.write(outdir / "probe_in.wav", rec_in[:n], sr)
    sf.write(outdir / "probe_out.wav", rec_out[:n], sr)
    (outdir / "probe_log.txt").write_text("\n".join(log_lines) + "\n")
    print(f"\nwrote {outdir}/probe_in.wav, probe_out.wav, probe_log.txt "
          f"({state['i']} blocks, xruns={state['xruns']}, clipped_blocks={state['clip']})")

    # Quick on-the-spot summary: find the biggest input transients, report their
    # suppression, plus overall level change.
    x, y = rec_in[:n].astype(np.float64), rec_out[:n].astype(np.float64)
    if n > sr:
        print(f"overall: in RMS {20*np.log10(np.sqrt(np.mean(x**2))+1e-12):.1f} dBFS -> "
              f"out RMS {20*np.log10(np.sqrt(np.mean(y**2))+1e-12):.1f} dBFS")
        env = np.abs(x)
        peaks = []
        i = sr  # skip first second
        while i < n - blk:
            j = int(np.argmax(env[i:i + n])) + i if False else 0
            break
        # simple transient scan: 50 ms windows, peak-to-median ratio
        win = 2400
        med = np.median(env)
        k = sr
        found = 0
        while k < n - win and found < 8:
            pk = np.max(env[k:k + win])
            if pk > max(10 * med, 0.05):
                po = np.max(np.abs(y[k:k + win]))
                print(f"  transient @{k/sr:5.1f}s: in {20*np.log10(pk+1e-12):6.1f} dBFS -> "
                      f"out {20*np.log10(po+1e-12):6.1f} dBFS (Δ {20*np.log10((pk+1e-12)/(po+1e-12)):+.1f} dB)")
                found += 1
                k += sr // 2
            else:
                k += win
        if not found:
            print("  (no strong transients detected in the recording)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

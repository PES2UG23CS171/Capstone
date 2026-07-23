# voiceiso — CPU-only real-time voice isolation

A commercial-style voice-isolation pipeline (in the spirit of Krisp / Apple Voice
Isolation) built around **DeepFilterNet3** on CPU, wrapped with preprocessing,
AEC, Silero VAD, a **learned EfficientAT 12-class noise classifier**, a dynamic
suppression controller, speech preservation, and a residual post-filter.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design and stage reference.

## Quick start (clean clone)

```bash
# 1. Rust toolchain — required to build DeepFilterNet's native extension
curl https://sh.rustup.rs -sSf | sh -s -- -y

# 2. Python deps (Python 3.10–3.12; numpy is pinned <2 for deepfilternet)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # includes onnxruntime (needed by the classifier)

# 3. Sanity-check the backends actually loaded (NOT heuristic/passthrough)
python -m voiceiso info
```

`info` prints `backend_summary`. For a correct setup you should see
`'enhancement': 'deepfilternet3'` and `'noise_classifier': 'efficientat'`. If you
see `'heuristic'` / `'passthrough'`, the model checkpoints or `onnxruntime` are
missing — the pipeline logs a **WARNING** and reports the reason in
`noise_classifier_fallback` (it never fails silently).

### Checkpoints

The learned classifier needs `checkpoints/efficientat_head12.onnx`
(+ `.onnx.data`). `PipelineConfig` auto-resolves it relative to the repo root, so
no path needs to be set. DeepFilterNet3's own weights are downloaded by `init_df`
on first run. DNSMOS is optional: `python -m scripts.fetch_dnsmos` pulls
`sig_bak_ovr.onnx` into `checkpoints/` (auto-wired thereafter).

## Usage

```bash
python -m voiceiso info                         # backends + stage list
python -m voiceiso enhance noisy.wav out.wav    # offline file enhancement
python -m voiceiso bench --n 10                 # quality + speed report
python -m voiceiso live                         # mic → pipeline → speakers
python -m app.main                              # desktop app (GUI + tray)
```

The live paths run DFN3 at **100 ms blocks on a worker thread** (DFN3's efficient
design point; it has no cross-call GRU memory, so 20 ms blocks measurably degrade
quality — see ARCHITECTURE.md §1.4). The audio callback never runs the network,
so it is xrun-safe. All models are warmed up before the stream starts, and the
callback drains to the freshest processed block, so mouth-to-ear latency stays at
the ~200–300 ms design point instead of ratcheting after load spikes.

### Live-demo checklist

- Launch with `./run.sh` (pins the venv — running the system Python silently
  drops DFN3 to passthrough; the startup log must say
  `'enhancement': 'deepfilternet3'`).
- **Use headphones** (or a headset). AEC is off by default and mic→speaker
  loopback echoes the presenter's voice back — DFN3 *preserves* speech by design.
- **Use a wired 48 kHz-capable microphone.** Bluetooth HFP mics run at 8/16 kHz
  and can fail to open (the app streams fixed 48 kHz).
- Connect all audio devices **before** launching the app (the device list is a
  startup snapshot). macOS mic mode: **Standard** (don't stack Apple's own
  Voice Isolation in front).
- **Strength slider at 100 %** — the wet/dry mix caps total suppression at
  −20 dB when the slider sits at 90 %.
- Give it ~2 s after start before judging (first classifier label lands ~1 s).

### Demo choreography (plays to measured strengths)

1. **Fan / steady noise while talking** — headline: −40 dB fan kill, speech
   preserved.
2. **Claps & snaps in speech pauses** — the transient kick crushes every
   impulse (−48…−53 dB, no decay across repeats). Clap *between* phrases —
   a clap overlapping your own words shares the speech bands DFN3 must
   protect and only attenuates 2–4 dB (known architectural limit; say so).
3. **Keyboard while talking** — strongest classifier class (F1 0.84).
4. **Phone playing a voice next to you** *(optional segment)* — relaunch with
   `VOICEISO_HEAD=checkpoints/efficientat_head12_v4.onnx ./run.sh` to enable
   the TV-rejection head (0.86 loudspeaker rejection vs 0.54 default; the
   trade-off is documented in ARCHITECTURE.md §10).
5. To let the audience hear the processed stream: record with
   `python -m scripts.live_probe` and play `diag_capture/probe_out.wav`, or
   route the app's output into a call. The room itself always carries the
   raw sound — suppression exists only in the stream.

## Benchmarks (reproducible)

```bash
# Enhancement quality/speed at the live 100 ms operating point:
python -m voiceiso bench --n 20 --snr 5           # add --block-ms 20 for the low-latency point

# Classifier: retrain on dev (uploader-grouped) + evaluate on held-out eval:
python -m scripts.retrain_head_dev               # train v2 + full eval report
python -m scripts.retrain_head_dev --eval-only   # just the eval table
```

> Without real noise corpora under `--data`, `bench` falls back to a **synthetic
> white-noise** condition and says so loudly — those numbers are not comparable to
> real-corpora results.

### Headline results (final sprint)

Synthetic +5 dB condition, n=20, 100 ms live operating point, output
time-aligned before scoring: **SI-SDRi +5.0 dB**, PESQ 1.24 → 1.36, STOI 0.79,
DNSMOS SIG 2.94 / BAK 3.00 / OVRL 2.24, RTF 0.38 (0.47 with the DNSMOS/PESQ
harness in-process), per-block p99 66 ms vs the 100 ms budget, peak RSS
~560 MB (harness-inclusive).

Real-condition behaviour on held-out FSD50K eval clips
([REALWORLD_RESULTS.md](REALWORLD_RESULTS.md)): silence-regime attenuation
**fan +38 dB, clap +45 dB, traffic +38 dB, keyboard +27 dB, TV +19 dB,
competing speech +23 dB**; during speech at 0 dB SNR, SI-SDRi from **+4.1
(TV) to +10.2 (keyboard)**.

### Classifier (`efficientat_head12_v3.onnx`, corrected protocol + window-robust)

Trained on **FSD50K.dev_audio** (uploader-grouped train/val, same split as v2),
tested on the held-out **FSD50K.eval** set, streamed through the runtime
classifier (which now tile-pads partial buffers — previously 18.4 % of test
clips were shorter than the 4 s window and physically could not be classified,
depressing every backend's numbers):

| Backend | macro-F1 | top-1 |
|---|---|---|
| heuristic *(pre-fix harness)* | 0.089 | 0.128 |
| pretrained-direct 527→12 *(pre-fix harness)* | 0.211 | 0.258 |
| head v1 (train-on-test, inflated) | *0.698* | *0.782* |
| head v2 (dev-trained, first-4s crops) | 0.635 | 0.740 |
| **head v3 (dev-trained, window-robust — deployed)** | **0.636** | **0.742** |

v3 trains on rolling-window-consistent crops (random 4 s slices; short clips
also placed sparsely in low-level context). On event-dense full clips it ties
v2, but in the live-mic transient regime it is markedly more robust: with a
0.5 s event at the window edge, true-class posterior 0.73 → **0.89** and top-1
16/24 → **21/24** (uploader-disjoint val clips). v1's row is invalid (30 % of
the test clips were in its training pool). The remaining gap to the 0.70
macro-F1 target is the **frozen mn10_as embedding ceiling** plus data-starved
classes (`fan`: 64 dev clips → F1 0.40); see ARCHITECTURE.md §10.

**Opt-in v4 "TV-rejection" head** (`VOICEISO_HEAD=checkpoints/efficientat_head12_v4.onnx`):
trained with synthesized loudspeaker/TV speech (the `tv` class had ZERO
FSD50K positives and could never fire), 6× fan augmentation, a stratified
validation split, and per-class calibrated thresholds shipped in ONNX
metadata. On held-out eval speech it rejects loudspeaker-played speech
**0.86–0.91 vs v3's 0.54** (labels it `tv` 0.41–0.68 vs **0.00**) with no live-speech
regression — but costs ~0.02 FSD50K macro-F1 (the new tv decision claims
channel-degraded field-recorded "speech" clips), so it **failed the
pre-registered no-regression deploy gate** and ships opt-in rather than
default. Use it for demos featuring a phone/TV playing voice next to the
speaker.

## Tests

```bash
python -m tests.test_phase_a        # plain python (prints PASS/SKIP), or `pytest`
```

## Known limitations

- ECAPA speaker embedder, VoiceFilter-Lite mask, and tiny-GRU post-filter are
  scaffolded (passthrough) until models are provided.
- AEC is implemented but off by default (needs a far-end reference signal).
- No INT8 model ships; the ONNX export is FP32.

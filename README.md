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

- **Use headphones** (or a headset). AEC is off by default and mic→speaker
  loopback echoes the presenter's voice back — DFN3 *preserves* speech by design.
- **Use a wired 48 kHz-capable microphone.** Bluetooth HFP mics run at 8/16 kHz
  and can fail to open (the app streams fixed 48 kHz).
- Connect all audio devices **before** launching the app (the device list is a
  startup snapshot).
- Output gain is capped at +6 dB; the strength slider is the demo control.

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

### Classifier (`efficientat_head12_v2.onnx`, corrected protocol)

Trained on **FSD50K.dev_audio** (uploader-grouped train/val), tested on the
held-out **FSD50K.eval** set:

| Backend | macro-F1 | top-1 |
|---|---|---|
| heuristic | 0.089 | 0.128 |
| pretrained-direct (527→12) | 0.211 | 0.258 |
| deployed v1 (train-on-test, inflated) | *0.636* | *0.658* |
| **NEW v2 (dev-trained, honest)** | **0.564** | **0.621** |

v2 is the deployed default. v1's 0.636 is invalid (trained on eval clips); on a
set held out from both, v1/v2 are on par (0.528/0.503 macro-F1, v2 better top-1).
This is below the 0.70 target because the **frozen mn10_as embedding is the
ceiling** (head capacity & threshold sweeps are flat); see ARCHITECTURE.md §10.

## Tests

```bash
python -m tests.test_phase_a        # plain python (prints PASS/SKIP), or `pytest`
```

## Known limitations

- ECAPA speaker embedder, VoiceFilter-Lite mask, and tiny-GRU post-filter are
  scaffolded (passthrough) until models are provided.
- AEC is implemented but off by default (needs a far-end reference signal).
- No INT8 model ships; the ONNX export is FP32.

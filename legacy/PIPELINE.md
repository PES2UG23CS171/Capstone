# Real-Time Transient Noise Suppressor — Pipeline & Steps

> Capstone project (PES2UG23CS171). A real-time audio filter that removes
> **transient noises** (door slams, keyboard clicks, dog barks) and
> **stationary background noise** (fans, hum) from a live microphone while
> preserving speech — running on CPU with no GPU required.

This document describes the **entire pipeline end-to-end**: how training data
is built, how the neural model is designed/trained/compressed/exported, and how
the live desktop application captures, filters, and plays back audio in real
time.

---

## 1. Two tracks, one system

The project has two parallel processing tracks that share the same goal:

| Track | Purpose | Entry point |
|-------|---------|-------------|
| **A — ML model track** | Train a neural denoiser (DeepFIR + Mamba SSM), compress it, and export to ONNX for CPU inference. | `training/train.py` → `model/` → `inference/` |
| **B — Live application track** | A desktop app (PyQt6 + system tray) that filters the microphone in real time. Currently ships the **classical-DSP `RealTimeFilter`** as the active engine, with the ONNX model and a stub denoiser as alternates. | `app/main.py` → `app/audio/engine.py` → `poc_realtime_transient.py` |

```
                 ┌─────────────────────── TRACK A (offline) ───────────────────────┐
 LibriSpeech ┐   │  generate_dataset → DataLoader → CombinedModel → train (SI-SDR) │
 FreeSound   ├──►│  → prune 50% → INT8 quantize → benchmark RTF → export ONNX       │
 RIRs        ┘   └───────────────────────────────┬─────────────────────────────────┘
                                                  │  model/filter_model.onnx
                                                  ▼
                 ┌─────────────────────── TRACK B (real-time) ─────────────────────┐
   microphone ──►│  sounddevice Stream → AudioEngine (child process)               │──► speakers
                 │     ├─ RealTimeFilter  (classical DSP — active default)          │
                 │     ├─ ONNXInferenceRunner / RealTimePipeline (model track)      │
                 │     └─ StubDenoiser    (passthrough fallback)                    │
                 │  Qt control window + tray  ◄─IPC Queues─►  engine                │
                 └──────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository layout

```
config.py                     Central config — every magic number (sample rate, model dims, paths)
poc_realtime_transient.py     Classical-DSP real-time filter + offline demo (ACTIVE live engine)
generate_dataset.py           Root dataset generator
process_manager.py            Process orchestration helper

dataset/
  generate_dataset.py         Synthetic (noisy, clean) pair generator → .npz
  rir_convolver.py            Room-impulse-response near/far-field convolution
  dataset_loader.py           PyTorch TransientNoiseDataset
  metadata.json               Generation config snapshot
  train/ val/ test/           Generated clean/ + noisy/ splits

model/
  deep_fir.py                 Layer 3 — DeepFIR tap predictor (causal CNN)
  mamba_ssm.py                Layer 4 — Mamba state-space model
  combined_model.py           DeepFIR + Mamba end-to-end model
  quantize.py                 Magnitude pruning + INT8 quantization + RTF benchmark
  export_onnx.py              Torch → ONNX exporter + verification
  pretrained_stub.py          Randomly-initialised stub model (for plumbing tests)
  filter_model.onnx(.data)    Exported model

training/
  train.py                    Training loop (data → loss → backward → checkpoint → export)
  losses.py                   SI-SDR, TSS, plosive-preservation, combined losses
  evaluate.py                 Metric evaluation on the test split

inference/
  onnx_runner.py              ONNXRuntime session wrapper (low-latency CPU)
  pipeline.py                 RealTimePipeline — ring buffer → ONNX → overlap-add

audio/
  ring_buffer.py              SPSC circular buffer
  audio_io.py / virtual_device.py

app/                          The desktop application
  main.py                     Entry point — Qt app, tray, spawns audio engine
  config.py                   AppConfig dataclass (passed across process boundary)
  audio/engine.py             Audio-engine child process (sounddevice + filter)
  audio/devices.py            Device enumeration
  inference/stub.py           StubDenoiser / ONNXDenoiser placeholder
  gui/control_window.py       PyQt6 settings panel (sliders, meters)
  gui/waveform_viewer.py      Live waveform plot
  gui/tray.py                 pystray system-tray icon
  ipc/messages.py             Command/Event dataclasses for GUI↔engine IPC

checkpoints/                  best.pt, latest.pt, training_log.csv
```

---

## 3. Track A — the ML model pipeline

### Step 1 · Dataset generation  (`dataset/generate_dataset.py`, `rir_convolver.py`)

Synthetic `(noisy, clean)` pairs are produced by mixing real corpora:

1. **Clean speech** from LibriSpeech `train-clean-100`.
2. **Transient noises** from FreeSound (barks, slams, clicks).
3. **Room acoustics** — each clip is convolved with a **Room Impulse Response**
   (`get_near_field_rir` / `get_far_field_rir`, RT60 0.15–0.9 s) so the model
   learns reverberant conditions.
4. **Dynamic mixing** at a random **SNR in [−5, +20] dB**.

Config (from `dataset/metadata.json`): 48 kHz, 4 s segments (192 000 samples),
10 000 total pairs, **80/10/10** train/val/test split, seed 42. Output is saved
as `.npz` arrays consumed by the loader.

### Step 2 · Data loading  (`dataset/dataset_loader.py`)

`TransientNoiseDataset(split=…, context_window=…)` yields `(noisy, clean)`
tensors. Training uses a **short context window** (`TRAIN_CONTEXT_WINDOW = 64`)
for speed — the model is length-agnostic, so it still runs on the full 512-sample
window at inference.

### Step 3 · Model architecture  (`model/`)

End-to-end signal flow: `noisy [B,T] → DeepFIR → Mamba SSM → clean [B,T]`.
**Total parameters: 279,041 (≈1.06 MB FP32, ≈0.27 MB INT8).**

**Layer 3 — DeepFIR** (`deep_fir.py`): a tiny causal CNN that *predicts FIR
filter taps* from the audio context, used to attenuate stationary noise.

```
[B, T] → CausalConv1d(1→32, k=8) → PReLU
       → CausalConv1d(32→64, k=8) → PReLU
       → AdaptiveAvgPool1d(1) → Linear(64 → FIR_FILTER_LENGTH=64) → Tanh  → taps [B,64]
```

The taps are applied with `apply_fir_torch` (differentiable `conv1d`) during
training, and converted to **minimum-phase** (`_to_minimum_phase`, cepstral
method) for stable causal application at inference. All convolutions are
**causal** (left-only padding) so the model never looks into the future.

**Layer 4 — Mamba SSM** (`mamba_ssm.py`): a selective state-space model
(`d_model=64`, `d_state=16`, `n_layers=4`) handling temporal/transient
structure. It exposes two modes:
* `forward(use_parallel=True)` — parallel scan for training.
* `forward_recurrent(x, hidden_states)` — single-step recurrence for real-time
  streaming inference.

**CombinedModel** (`combined_model.py`) wires them:
`DeepFIR(taps)→apply → Linear(1→64) → MambaSSM → Linear(64→1)`, with
`forward_train` (full sequence) and `forward_realtime` (per-sample) paths.

### Step 4 · Losses  (`training/losses.py`)

* **SI-SDR loss** — scale-invariant signal-to-distortion ratio (primary metric).
* **TSS loss** — Transient Suppression Score: penalises residual transient
  energy *and* distortion of non-transient speech regions.
* **Plosive-preservation loss** — 10× penalty for distorting plosive segments,
  so consonants like “P/T/K” survive.
* **`combined_loss`** — weighted sum `(1.0·SI-SDR + 0.5·TSS + 1.0·plosive)`.

The current training loop optimises **SI-SDR loss** directly.

### Step 5 · Training loop  (`training/train.py`)

```
device = MPS (Apple Silicon) or CPU
optimiser = AdamW(lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(T_max=epochs)

for epoch in range(EPOCHS=50):
    for (noisy, clean) in train_loader:        # BATCH_SIZE=32
        est  = model(noisy)
        loss = si_sdr_loss(clean, est)
        loss.backward(); clip_grad_norm_(1.0); optimiser.step()
    validate → log to checkpoints/training_log.csv
    save latest.pt; save best.pt if val improves
```

**Result (50 epochs, `training_log.csv`):** validation SI-SDR loss converged
from ≈ −10 dB (epoch 1) to **≈ −26.5 dB** (epoch 50) — i.e. ≈ **+26.5 dB
SI-SDR** on validation. (Convergence curve was previously rendered to
`training_convergence.png`.)

### Step 6 · Compression  (`model/quantize.py`)

After training, the best checkpoint is:
1. **Pruned** — `apply_magnitude_pruning` zeroes the smallest 50 % of weights
   (`PRUNE_RATIO = 0.50`).
2. **Quantized** — `quantize_model_int8` applies INT8 dynamic quantization
   (`QUANTIZE_INT8 = True`), shrinking the model ≈4× (→ ≈0.27 MB).
3. **Benchmarked** — `benchmark_rtf` measures the **Real-Time Factor** to verify
   it clears the `TARGET_RTF = 0.8` budget.

### Step 7 · ONNX export  (`model/export_onnx.py`)

`export_to_onnx` traces the model to **ONNX opset 18** with a dynamic batch
axis (`noisy_audio`→`clean_audio`), writes `model/filter_model.onnx`, then
**verifies** it by running a random input through ONNX Runtime and asserting the
output shape `(1, context_window)`.

### Step 8 · Inference engine  (`inference/`)

* **`ONNXInferenceRunner`** (`onnx_runner.py`) — wraps
  `onnxruntime.InferenceSession` with low-latency CPU settings (all graph
  optimisations, sequential exec, pre-allocated input buffer, `warmup()` to
  JIT-compile).
* **`RealTimePipeline`** (`pipeline.py`) — glues the layers: write samples to
  the **ring buffer**, read the context window, run ONNX inference, return the
  centre/overlap-add output, with a `LatencyTracker` reporting avg/max latency
  and RTF. Supports `bypass_mode` (A/B) and `suppression_level` (wet/dry mix).

---

## 4. Track B — the live real-time application

### Process architecture  (`app/main.py`)

```
Main process (GUI):  QApplication (Qt event loop, main thread — required by macOS)
                     ├─ ControlWindow  (PyQt6 sliders/toggles/meters)
                     └─ TrayManager    (pystray icon, daemon thread)
        ── multiprocessing "spawn" boundary ──
Child process:       AudioEngine  (sounddevice.Stream + filter)
```

The GUI and engine communicate **only** through two `multiprocessing.Queue`
objects, with pickle-safe dataclasses defined in `app/ipc/messages.py`:
* **cmd_q (GUI→Engine):** `Command(kind, value)` — `SET_ENABLED`,
  `SET_STRENGTH`, `SET_GAIN`, `SET_INPUT_DEVICE`, `SET_OUTPUT_DEVICE`,
  `SET_PASSTHROUGH`, `GET_DEVICES`, `SHUTDOWN`.
* **evt_q (Engine→GUI):** `Event(kind, payload)` — `STATUS` (levels, xruns,
  RTF), `DEVICE_LIST`, `ERROR`, `ENGINE_STOPPED`.

> **macOS note:** `QApplication` is created **before** any child process is
> spawned (Cocoa must init first), and `app/main.py` includes a venv-relaunch
> shim so PyQt6 gets a proper bundle context.

### The audio callback  (`app/audio/engine.py`)

For every block (`block_size=1024` frames @ 48 kHz), the non-blocking
sounddevice callback:

1. Measures input peak level.
2. If **passthrough** → copy mic→output, zero processing (minimum latency).
3. Else if **enabled** and `RealTimeFilter` available → process the block in
   **128-sample sub-chunks** through the classical DSP filter, then apply the
   **wet/dry mix** controlled by the GUI *strength* slider.
4. Else → `StubDenoiser` (passthrough) or raw copy.
5. Apply output gain, clip to [−1, 1], measure output peak, write to output
   (mono→stereo duplicated if needed).
6. Periodically emit a `STATUS` event with levels, xrun count, and RTF.

### The active filter — classical DSP  (`poc_realtime_transient.py`)

`RealTimeFilter.process_chunk` chains two stages on each 128-sample chunk
(~2.67 ms, well under the real-time budget — measured **RTF ≈ 0.015–0.02**,
~50–65× headroom):

1. **`TransientDetector`** — a fast-attack / slow-release energy gate. It tracks
   a **fast EMA** (τ≈1 ms) and **slow EMA** (τ≈500 ms) of signal power. When the
   fast envelope exceeds the slow envelope by more than the threshold (and clears
   an absolute noise floor), it declares a transient and applies an attenuation
   gain, with a short **hold** then a smooth **release ramp** back to unity to
   avoid clicks.
2. **`NoiseEstimator`** — minimum-statistics stationary-noise suppression. It
   tracks the running minimum energy across short windows as the noise floor and
   applies a spectral-subtraction-style gain
   `G = max(1 − β·noise_floor/signal_energy, G_min)`.

The same code runs both **live mode** (`--mode live`, mic→speaker) and an
**offline demo** (`--mode demo`) that generates a synthetic test signal,
filters it, and prints a latency/feasibility report plus per-event suppression
diagnostics.

---

## 5. Key configuration  (`config.py` / `app/config.py`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `SAMPLE_RATE` | 48 000 Hz | VoIP-grade audio |
| `BLOCK_SIZE` (app) | 1024 frames (~21 ms) | sounddevice callback size |
| `CHUNK_SIZE` (DSP) | 128 samples (~2.67 ms) | filter processing block |
| `CONTEXT_WINDOW_SAMPLES` | 512 | window fed to the model |
| `FIR_FILTER_LENGTH` | 64 | DeepFIR taps |
| `MAMBA_D_MODEL / D_STATE / N_LAYERS` | 64 / 16 / 4 | Mamba dims |
| `PRUNE_RATIO` | 0.50 | weights removed by magnitude |
| `QUANTIZE_INT8` | True | INT8 dynamic quantization |
| `TARGET_RTF` | 0.8 | must run faster than real time |
| `SNR_RANGE_DB` | (−5, 20) | dynamic mixing range |
| `BATCH_SIZE / EPOCHS / LR` | 32 / 50 / 1e-3 | training |

### Live DSP tuning (current `TransientDetector` / `NoiseEstimator` defaults)

| Knob | Value | Notes |
|------|-------|-------|
| transient threshold | 12 dB (~16× power) | clears speech onsets, catches impulses |
| suppression depth | −18 dB | attenuate, don't mute |
| hold / release | 40 ms / 80 ms | short hold + smooth ramp |
| absolute energy gate | 1e-4 | stops false triggers in quiet passages |
| noise `beta` / `g_min` | 2.0 / 0.25 | gentle subtraction, ≤ −12 dB cut |

The live engine further retunes these for microphone use (threshold 20 dB,
suppression −15 dB, hold 25 ms, `beta`=1.5, `g_min`=0.4).

---

## 6. How to run

```bash
# Offline DSP demo: synth signal → filter → report (writes test_*.wav)
python poc_realtime_transient.py --mode demo
python poc_realtime_transient.py --mode demo --input my_noisy.wav --output out.wav

# Live DSP filter: microphone → filter → speakers (Ctrl+C to stop)
python poc_realtime_transient.py --mode live

# Launch the full desktop app (GUI + tray + audio engine)
python -m app.main

# Train the model (Track A)
python -m training.train --epochs 50            # add --resume to continue
python -m training.evaluate                     # metrics on the test split
python -m model.export_onnx                      # export to ONNX
```

Dependencies: see `requirements.txt` (PyQt6, sounddevice, numpy, scipy,
soundfile, torch/torchaudio, onnx/onnxruntime, librosa, pesq, tqdm, pyqtgraph,
Pillow, pystray).

---

## 7. Performance summary

* **Latency budget:** 128-sample chunk = 2.67 ms; measured DSP processing
  ≈40–60 µs/chunk → **RTF ≈ 0.015–0.02** (≈50–65× faster than real time),
  leaving ample headroom for the ML layers under the 0.8 target.
* **Model:** 279 k parameters, ≈1.06 MB (FP32) → ≈0.27 MB (INT8 after 50 %
  prune + INT8 quantize).
* **Training:** validation SI-SDR ≈ **+26.5 dB** after 50 epochs.
* **DSP cleaning quality** (offline demo, synthetic clip): correlation of output
  with the clean reference **0.82**, steady-noise reduced ≈ −6 dB, transients
  attenuated while speech is preserved.

---

## 8. Recent fixes

The classical DSP filter was over-suppressing — behaving as a near-total mute
gate (output↔clean correlation ≈0.18, ~96 % of samples near-silent). Fixed in
`poc_realtime_transient.py` and `app/audio/engine.py`:

1. `TransientDetector` defaults retuned (threshold 6→12 dB, hold 350→40 ms,
   suppression 40→18 dB).
2. Added an **absolute energy gate** so quiet passages no longer trigger false
   transients (was firing on every fluctuation when the slow envelope → 0).
3. `NoiseEstimator` defaults softened (`beta` 8→2, `g_min` 0.01→0.25) to stop it
   crushing speech to −40 dB.
4. Fixed minimum-statistics window accounting to use the actual chunk length.
5. Wired the GUI **strength** slider into the live filter via a wet/dry mix
   (it previously had no effect when the real filter was active).

These raised output↔clean correlation from **0.18 → 0.82** and dropped
near-silent output from **96 % → 26 %** while keeping noise/transient
suppression.

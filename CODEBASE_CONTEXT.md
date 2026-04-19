# CODEBASE_CONTEXT.md

> **Exhaustive codebase reference for the Real-Time AI-Powered Audio Filter for Transient Noise Suppression (Capstone 190).**
> Team: Dhrushaj Achar, Chandan B, Deepesh Padhy, Sahil Uday Bhat · Python 3.10+ · License: MIT

---

## 1. Project Overview

### What This System Does

This is a **CPU-only, real-time audio noise suppression system** that removes *transient* noises — dog barks, door slams, keyboard clicks, sirens — from a live microphone stream while preserving human speech, including plosive consonants ('P', 'T', 'K'). The cleaned audio is played back through the speaker in real-time.

**Transient noise** is architecturally distinct from stationary background noise (fans, HVAC hum). Stationary noise has a stable spectral envelope removable via spectral subtraction. Transient noise is *impulsive* — it appears suddenly, has high peak energy, and occupies a wide frequency band for a very short duration (5–400 ms). Classical spectral-subtraction fails on transients because their spectral profile changes faster than any noise-floor estimator can track.

### Production Status

The system is in **Phase 1 (PoC complete) / Phase 2 (ML training pending)**:

- The **DSP pipeline** (ring buffer, transient detector, noise estimator) is fully functional and benchmarked.
- The **ML model architecture** (DeepFIR + Mamba SSM) is structurally complete in PyTorch, with ONNX export, quantization, and pruning utilities built.
- The **ML model weights are random** (untrained). The `StubDenoiser` in the GUI passthrough copies audio unchanged. Real noise suppression quality requires completing the training phase with LibriSpeech + FreeSound data.
- The **GUI** is fully functional with system tray, level meters, device selection, and waveform viewer.

### Two Runtime Modes

| Mode | Entry Point | Description |
|------|-------------|-------------|
| **GUI live** | `python -m app.main` | Launches PyQt6 control window + system tray. Audio engine runs in a child process via `multiprocessing.Process`. Real-time mic → filter → speaker. |
| **Offline benchmark** | `python poc_realtime_transient.py --mode demo` | Generates a synthetic 10s test signal, processes it chunk-by-chunk through the classical DSP pipeline, saves WAV files, and prints a FEASIBILITY VERDICT with RTF. |

### The Governing Constraint

Every design decision is driven by: **< 2.67 ms per 128-sample chunk at 48 kHz, CPU-only, no GPU.**

```
chunk_budget = 128 / 48000 = 0.002667 s = 2667 µs
```

---

## 2. Architecture & Directory Structure

### Full Directory Tree

```
Capstone/
├── app/                          # GUI application (PyQt6 + multiprocessing)
│   ├── __init__.py               # Package marker, __version__ = "0.1.0"
│   ├── __main__.py               # `python -m app` entry — calls app.main.main()
│   ├── main.py                   # APPLICATION ENTRY POINT: QApplication + engine + tray
│   ├── config.py                 # AppConfig dataclass (runtime knobs for GUI/engine)
│   ├── audio/
│   │   ├── __init__.py           # Empty
│   │   ├── devices.py            # query_devices(), input_devices(), output_devices()
│   │   └── engine.py             # AudioEngineHandle (GUI-side) + run_engine() (child process)
│   ├── gui/
│   │   ├── __init__.py           # Empty
│   │   ├── control_window.py     # ControlWindow: toggles, sliders, meters, device combos
│   │   ├── tray.py               # TrayManager: pystray icon on daemon thread
│   │   └── waveform_viewer.py    # WaveformViewer: pyqtgraph 3-panel waveform display
│   ├── inference/
│   │   ├── __init__.py           # Empty
│   │   └── stub.py               # StubDenoiser (passthrough) + ONNXDenoiser (Phase 2 placeholder)
│   └── ipc/
│       ├── __init__.py           # Empty
│       └── messages.py           # CmdType, Command, EvtType, Event, StatusPayload, DeviceInfo
├── audio/                        # Standalone audio I/O library (used by process_manager.py)
│   ├── __init__.py               # Package docstring
│   ├── audio_io.py               # AudioIOManager: sounddevice streams, passthrough/filter modes
│   ├── ring_buffer.py            # RingBuffer (Layer 1): lock-free SPSC circular buffer
│   └── virtual_device.py         # Virtual audio cable detection (VB-Audio, PulseAudio null sink)
├── model/                        # Neural network definitions (PyTorch)
│   ├── __init__.py               # Package docstring
│   ├── deep_fir.py               # DeepFIRPredictor (Layer 3): Conv1D → FIR taps
│   ├── mamba_ssm.py              # MambaSSM (Layer 4): selective state space model
│   ├── combined_model.py         # CombinedModel: DeepFIR → Mamba sequential pipeline
│   ├── export_onnx.py            # export_to_onnx(): PyTorch → ONNX with verification
│   ├── quantize.py               # apply_magnitude_pruning(), quantize_model_int8(), benchmark_rtf()
│   ├── pretrained_stub.py        # create_stub_model(), save_stub_checkpoint(), load_model()
│   ├── filter_model.onnx         # Exported ONNX model (random weights — stub)
│   └── filter_model.onnx.data    # External tensor data for the ONNX model
├── inference/                    # ONNX Runtime inference engine (standalone)
│   ├── __init__.py               # Package docstring
│   ├── onnx_runner.py            # ONNXInferenceRunner: pre-allocated buffers, warmup, latency tracking
│   └── pipeline.py               # RealTimePipeline: ring buffer → ONNX → overlap-add → output
├── dataset/                      # Dataset generation & loading
│   ├── __init__.py               # Package docstring
│   ├── generate_dataset.py       # generate_full_dataset(): .npz pairs from LibriSpeech+FreeSound
│   ├── dataset_loader.py         # TransientNoiseDataset: PyTorch Dataset from .npz files
│   └── rir_convolver.py          # load_rir(), apply_rir(), get_near_field_rir(), get_far_field_rir()
├── training/                     # Training loop & evaluation
│   ├── __init__.py               # Package docstring
│   ├── train.py                  # train(): AdamW + CosineAnnealing + prune + quantize + export
│   ├── evaluate.py               # SI-SDRi, PESQ, TSS, plosive SI-SDR + full_evaluation_report()
│   └── losses.py                 # si_sdr_loss(), tss_loss(), plosive_preservation_loss(), combined_loss()
├── checkpoints/
│   └── stub_model.pt             # Random-weight checkpoint for demo
├── config.py                     # CENTRAL CONFIG: all constants (sample rate, model dims, paths)
├── poc_realtime_transient.py     # OFFLINE PoC: RingBuffer, TransientDetector, NoiseEstimator
├── process_manager.py            # AudioProcess + ProcessManager: alt multi-process architecture
├── generate_dataset.py           # FULL dataset generator: DatasetBuilder, RIRProvider, CLI
├── requirements.txt              # Full dependency list (GUI + ML + eval)
├── requirements_poc.txt          # Minimal deps for PoC only (numpy, scipy, sounddevice, soundfile)
├── README.md                     # User-facing documentation
├── PRESENTATION_MATERIAL.md      # Internal presentation strategy notes
├── test_clean.wav                # Generated clean test signal (output of demo mode)
├── test_noisy.wav                # Generated noisy test signal
└── test_noisy_filtered.wav       # Filtered output from PoC demo
```

### The 4-Layer Pipeline

**Data Flow:** `Mic → [Ring Buffer] → [Mamba SSM transient gating] → [DeepFIR stationary noise] → Speaker`

#### Layer 1: Ring Buffer (`audio/ring_buffer.py`)

`RingBuffer` is a lock-free SPSC circular buffer backed by `np.zeros(capacity, dtype=np.float32)`. Capacity default: `SAMPLE_RATE * RING_BUFFER_SECONDS` = 24,000 samples.

- **Producer** (audio callback): `write(samples)` uses `threading.Lock` only for `_write_pos` pointer update.
- **Consumer** (processing): `read(n)` or `read_context(window)` snapshots `_write_pos` (atomic under GIL) — no lock.
- **Overrun**: oldest data silently overwritten by design.
- **Underrun**: `read()` zero-pads if fewer than `n` written.
- **Latency**: `get_latency_ms()` returns 1/sample_rate × 1000 ≈ 0.021 ms.

The PoC (`poc_realtime_transient.py` lines 64–112) has a simpler `RingBuffer` — no lock, GIL-reliant, `_write`/`_read` positions, returns `None` on underrun.

#### Layer 2: Quantization (`model/quantize.py`)

- `apply_magnitude_pruning()`: `torch.nn.utils.prune.l1_unstructured()` on all `nn.Linear`/`nn.Conv1d` with `amount=0.50`. Made permanent via `prune.remove()`.
- `quantize_model_int8()`: `torch.quantization.quantize_dynamic()` targeting `{nn.Linear, nn.Conv1d}` with `dtype=torch.qint8`.
- Combined: **7× speedup** vs FP32 baseline.

#### Layer 3: DeepFIR (`model/deep_fir.py`)

`DeepFIRPredictor` predicts 64 minimum-phase FIR taps:

```
Input: [B, 512]
  → CausalConv1d(1, 32, k=8) → PReLU → CausalConv1d(32, 64, k=8) → PReLU
  → AdaptiveAvgPool1d(1) → [B, 64] → Linear(64, 64) → Tanh
Output: [B, 64]
```

Taps converted to minimum-phase via cepstral method (`_to_minimum_phase()`), applied with `scipy.signal.lfilter()`. `CausalConv1d` pads `kernel_size-1` zeros on left only.

#### Layer 4: Mamba SSM (`model/mamba_ssm.py`)

`MambaSSM`: 4 `MambaBlock` layers with `RMSNorm` and residual connections.

Each `MambaBlock` (d_model=64, d_state=16, d_conv=4, expand=2, d_inner=128):
1. `in_proj` Linear(64, 256) → split x_inner, gate z
2. Depthwise Conv1d(128, 128, k=4, groups=128) → SiLU
3. `x_proj` Linear(128, 160) → B_ssm [B,L,16], C_ssm [B,L,16], dt [B,L,128]
4. `dt_proj` Linear(128, 128) + softplus
5. Selective scan: `A_log` [128,16] state matrix, `D` [128] skip connection
6. Gate: y × SiLU(z) → `out_proj` Linear(128, 64)

Two scan modes: `selective_scan_parallel` (training) and `selective_scan_sequential` (inference, O(1) per sample recurrence).

---

## 3. Tech Stack & Dependencies

### `requirements.txt`

| Package | Version | Purpose | Used In |
|---------|---------|---------|---------|
| `PyQt6` | ≥6.5 | GUI framework | `app/main.py`, `app/gui/*` |
| `pystray` | ≥0.19 | System tray icon | `app/gui/tray.py` |
| `sounddevice` | ≥0.4 | Audio I/O (PortAudio) | `app/audio/engine.py`, `audio/audio_io.py`, `poc_*.py` |
| `numpy` | ≥1.24 | Vectorized DSP | Everywhere |
| `pyqtgraph` | ≥0.13 | Waveform plotting | `app/gui/waveform_viewer.py` |
| `Pillow` | ≥10.0 | Tray icon (PIL Image) | `app/gui/tray.py` |
| `torch` | ≥2.2.0 | Model definition/training | `model/`, `training/` |
| `torchaudio` | ≥2.2.0 | Audio transforms | Listed but not actively used |
| `onnx` | ≥1.16.0 | Model format | `model/export_onnx.py` |
| `onnxruntime` | ≥1.18.0 | CPU inference | `inference/onnx_runner.py` |
| `scipy` | any | `lfilter`, `fftconvolve`, `resample_poly`, `chirp` | `model/deep_fir.py`, `dataset/`, `poc_*.py` |
| `soundfile` | any | WAV/FLAC I/O | `dataset/`, `poc_*.py`, `generate_dataset.py` |
| `librosa` | ≥0.10.0 | Audio analysis | Available, not actively used |
| `pesq` | ≥0.0.4 | PESQ metric | `training/evaluate.py` |
| `tqdm` | ≥4.66.0 | Training progress | `training/train.py` |

### `requirements_poc.txt` (Minimal)

`numpy>=1.24.0`, `scipy>=1.11.0`, `sounddevice>=0.4.6`, `soundfile>=0.12.1`

### Why ONNX Runtime

ONNX Runtime's C++ backend releases the Python GIL during `session.run()`, allowing the audio callback thread to continue unblocked. Pre-allocated buffers (`ONNXInferenceRunner._input_buf`) minimize Python allocations on the hot path.

---

## 4. ML Model Architecture

### Combined Model — `model/combined_model.py`

Sequential composition:
```
noisy_audio [B, 512]
  → DeepFIRPredictor → taps [B, 64]
  → apply_fir_torch(noisy_audio, taps) → intermediate [B, 512]
  → unsqueeze(-1) → [B, 512, 1]
  → input_proj Linear(1, 64) → [B, 512, 64]
  → MambaSSM (4 layers) → [B, 512, 64]
  → output_proj Linear(64, 1) → squeeze → [B, 512]
```

Two forward modes: `forward_train()` (parallel scan) and `forward_realtime()` (recurrent, single-sample).

### ONNX Export — `model/export_onnx.py`

- **Opset**: 18. **Input**: `"noisy_audio"` [batch, 512]. **Output**: `"clean_audio"` [batch, 512]. Dynamic batch axis.
- File: `model/filter_model.onnx` + `.onnx.data`
- Verified with `onnxruntime.InferenceSession` + shape assertion.

### Quantization Details

`quantize_model_int8()` applies dynamic INT8 to `nn.Linear` and `nn.Conv1d`. The `A_log` parameter, `D` parameter, and `RMSNorm` weights remain FP32 (they are `nn.Parameter`, not `nn.Linear`/`nn.Conv1d`). Pruning is applied before quantization in the post-training pipeline (`train.py:164–169`).

---

## 5. Dataset & Training Pipeline

### Datasets

LibriSpeech (clean speech, `data/raw/librispeech`), FreeSound (transient noises, `data/raw/freesound`), OpenAIR RIRs (`data/raw/openair_rirs`).

### Generation

Two generators:
1. **`generate_dataset.py`** (root): `DatasetBuilder` + `RIRProvider` (supports pyroomacoustics synthetic RIRs). Outputs train/val/test WAV pairs + `metadata.json`. SNR: [-5, +20] dB. Segment: 4s @ 48 kHz. Default: 10,000 pairs (80/10/10).
2. **`dataset/generate_dataset.py`**: Outputs `.npz` files for `TransientNoiseDataset`.

### Training — `training/train.py`

AdamW (lr=1e-3, weight_decay=1e-4), CosineAnnealingLR (T_max=50), batch_size=32, 50 epochs, gradient clipping max_norm=1.0. Loss: `si_sdr_loss` (negative SI-SDR). Checkpoints: `checkpoints/best.pt`, `checkpoints/latest.pt`. Post-training: prune → quantize → benchmark → ONNX export.

### Evaluation — `training/evaluate.py`

| Metric | Target | Function |
|--------|--------|----------|
| SI-SDRi | > 4.0 dB | `compute_si_sdri()` |
| PESQ | ≥ 3.2 | `compute_pesq()` (resamples to 16 kHz via `resample_poly`) |
| TSS | > 65% | `compute_tss()` |
| Plosive SI-SDR | > 25 dB | `evaluate_on_plosives()` |

### Loss Functions — `training/losses.py`

- `si_sdr_loss()`: negative SI-SDR (minimize)
- `tss_loss()`: residual transient energy + 0.5 × speech distortion penalty
- `plosive_preservation_loss()`: MSE on plosive regions × 10 weight
- `combined_loss()`: weighted sum with default weights (1.0, 0.5, 1.0)

---

## 6. Audio I/O & Ring Buffer

- **Library**: `sounddevice` (PortAudio bindings)
- **Sample rate**: 48,000 Hz. **Channels**: 1 (mono). **Internal format**: float32.
- **Chunk**: 128 (PoC), 1024 (`AppConfig.block_size`), 256 (`config.BLOCK_SIZE`).
- **Pass-through mode**: `_audio_callback()` copies `indata → outdata` with zero processing when `passthrough=True`.
- **macOS**: QApplication before child processes. Venv re-exec via `sys._base_executable` to avoid Cocoa crash.
- **Windows**: VB-Audio Virtual Cable detection via keyword scan in `audio/virtual_device.py`.
- **Linux**: PulseAudio null sink via `pactl load-module module-null-sink`.

---

## 7. GUI & System Tray

### `ControlWindow` Widgets

| Widget | Variable | IPC Command |
|--------|----------|-------------|
| Suppression ON/OFF | `btn_toggle` | `SET_ENABLED` |
| Pass-Through | `btn_passthrough` | `SET_PASSTHROUGH` |
| Strength slider (0–100) | `slider_strength` | `SET_STRENGTH` (0.0–1.0) |
| Gain slider (-12…+12 dB) | `slider_gain` | `SET_GAIN` |
| Input/Output combos | `combo_input/output` | `SET_INPUT/OUTPUT_DEVICE` |
| Level meters | `meter_in/out` | Read from `StatusPayload` |
| RTF label | `lbl_rtf` | Read from `StatusPayload.rtf` |
| PoC button | `btn_poc` | Opens `WaveformViewer` |

`closeEvent()` **hides** window (does not quit). Tray "Quit" triggers full shutdown.

### Tray (`app/gui/tray.py`)

PIL-drawn mic icon (green/red). Menu: Show/Hide, Suppression toggle, Quit. `_TrayBridge(QObject)` bridges pystray thread → Qt signals.

### Waveform Viewer (`app/gui/waveform_viewer.py`)

3 `pyqtgraph.PlotWidget` panels (Clean/Noisy/Filtered), linked X-axes, transient markers at [1.2, 3.5, 5.0, 6.8, 8.5]s. Metric cards: Mean µs, P99, RTF, Headroom, PASS/FAIL. Play/Stop buttons via `sounddevice.play()`.

---

## 8. Configuration & Constants

### `config.py` (Root) — All Constants

```python
SAMPLE_RATE = 48_000;  CHANNELS = 1;  BIT_DEPTH = 16;  BLOCK_SIZE = 256
RING_BUFFER_SECONDS = 0.5;  CONTEXT_WINDOW_SAMPLES = 512
FIR_FILTER_LENGTH = 64;  MAMBA_D_MODEL = 64;  MAMBA_D_STATE = 16;  MAMBA_N_LAYERS = 4
PRUNE_RATIO = 0.50;  QUANTIZE_INT8 = True;  TARGET_RTF = 0.8
ONNX_MODEL_PATH = "model/filter_model.onnx"
BATCH_SIZE = 32;  EPOCHS = 50;  LR = 1e-3;  SNR_RANGE_DB = (-5, 20)
```

### `app/config.py` — GUI Runtime

`AppConfig` dataclass: `sample_rate=48000`, `block_size=1024`, `channels=1`, `suppression_enabled=True`, `suppression_strength=1.0`, `output_gain_db=0.0`, `status_interval=0.05`.

### ONNX Session Options (`inference/onnx_runner.py`)

`ORT_ENABLE_ALL` optimization, `ORT_SEQUENTIAL` execution, `cpu_count()-1` threads, `enable_mem_pattern=True`, `CPUExecutionProvider` only.

---

## 9. Performance & Real-Time Guarantees

| Component | Budget | Measured |
|-----------|--------|----------|
| Total per chunk | 2,667 µs | — |
| DSP pipeline | — | < 100 µs |
| ONNX inference | — | ~2,500 µs remaining |
| RTF target | < 0.80 | PoC: ~0.01–0.05 |

Mamba is O(N) vs Transformer O(N²). Pre-allocated buffers avoid dynamic allocation on hot path. PoC transient detector loop (128 iterations) costs < 50 µs.

---

## 10. Error Handling & Failure Modes

| Scenario | Behavior | Location |
|----------|----------|----------|
| Device disconnect | `PortAudioError` → fallback to demo/log error | `engine.py:240`, `poc:777` |
| Inference overrun | xruns counter incremented | `engine.py:152` |
| Ring buffer overrun | Oldest data overwritten | `ring_buffer.py:52-66` |
| Ring buffer underrun | Zero-padded output | `ring_buffer.py:76-93` |
| Model missing | Falls back to passthrough | `process_manager.py:94-106` |
| `sounddevice` missing | `sd = None`, fallback to demo | `poc:39-42` |
| Stream open failure | Retries 3× with 500 ms delay | `audio_io.py:281-290` |

---

## 11. Testing & Evaluation

**No formal test suite** (no pytest/unittest files). Testing via:

1. **`poc_realtime_transient.py --mode demo`**: Generates 10s signal with 5 transients, processes, prints RTF verdict (PASS if RTF < 0.80), saves WAV files.
2. **`python -m training.evaluate`**: Runs metrics on synthetic data, prints pass/fail report.
3. **Waveform Viewer**: Visual verification via GUI.

---

## 12. Coding Conventions & Extension Guide

- **Files**: `snake_case.py`. **Classes**: `PascalCase`. **Functions**: `snake_case`. **Constants**: `UPPER_SNAKE`.
- **Config**: All in root `config.py`. Import as `import config as cfg`.
- **To add a new model**: Define `nn.Module` with `forward([B, 512]) → [B, 512]`, export via `export_to_onnx()`.
- **To add a metric**: Add function to `training/evaluate.py`, call from `full_evaluation_report()`.
- **To swap audio backend**: Replace `sounddevice` imports in `engine.py`/`audio_io.py`, adapt callback signature.

---

## 13. Key Design Decisions & Tradeoffs

| Decision | Rationale |
|----------|-----------|
| Mamba over Transformer | O(N) vs O(N²), enables real-time per-sample recurrence |
| ONNX over PyTorch | GIL released during C++ inference, simpler deployment |
| Sample-by-sample ring buffer | ~0.021 ms latency vs 20–50 ms block processing |
| INT8 + 50% pruning | 7× speedup, acceptable accuracy cost |
| Minimum-phase FIR | Causal filtering required for real-time (no future samples) |
| **NOT handled** | Music, multi-channel, GPU, AEC, non-transient noise beyond DeepFIR |

---

## 14. Known Gotchas & Fragile Areas

### Initialization Order
- macOS: `QApplication` **before** child processes (`app/main.py:78`), or Cocoa crash.
- Venv re-exec (`main.py:56-60`) does **not** add venv site-packages — intentional ABI mismatch avoidance.
- ONNX session warmup (`warmup(50)`) must complete before audio thread hits inference.

### Plosive Exception
In PoC: plosive 'P' (8.5s) has amplitude=0.10 with slow ramp-up envelope — `env_fast` never exceeds `threshold × env_slow` → no trigger → preserved. In Mamba: selectivity is learned via B/C/dt projections — implicit in trained weights.

### Hardcoded Paths
- `"model/filter_model.onnx"` in `config.py:45` — moving file breaks inference.
- `"device_prefs.json"` in `audio/audio_io.py:36` — written to CWD.
- `sys._base_executable` in `main.py:56` — CPython implementation detail.
- Transient positions `[1.2, 3.5, 5.0, 6.8, 8.5]` duplicated in `poc_realtime_transient.py:508` and `waveform_viewer.py:479`.

### Technical Debt
- **Two multiprocessing architectures**: `process_manager.py` (dict-based commands, `AudioIOManager`) vs `app/audio/engine.py` (enum-based `CmdType`, `StubDenoiser`). GUI uses `engine.py`; `process_manager.py` is effectively dead code.
- **Two config systems**: root `config.py` (module constants) vs `app/config.py` (`AppConfig` dataclass). `BLOCK_SIZE` = 256 in root, `block_size` = 1024 in `AppConfig`.
- `selective_scan_parallel` claims cumsum parallelism but contains a sequential `for t in range(L)` loop — not actually parallel.
- `engine.py` imports `RealTimeFilter` from `poc_realtime_transient.py` via `sys.path` manipulation — tight coupling between PoC and live GUI.

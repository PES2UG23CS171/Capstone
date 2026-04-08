# Real-Time AI-Powered Audio Filter for Transient Noise Suppression

**Capstone Project 190 | Team: Dhrushaj Achar, Chandan B, Deepesh Padhy, Sahil Uday Bhat**

This repository contains the complete backend and GUI for a real-time, CPU-only, AI-powered transient noise suppression system. The system suppresses transient noises (dog barks, door slams, keyboard clicks, sirens) while preserving human speech.

---

## 🚀 Quick Start / How to Run

### 1. Prerequisites

First, ensure you have Python 3.10+ installed and your virtual environment activated:

**Windows:**
```powershell
.\.venv_poc\Scripts\activate
pip install -r requirements.txt
```

**macOS & Linux:**
```bash
source .venv_poc/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Main Application (GUI)

To start the main application with the graphical user interface:

**Windows / macOS / Linux:**
```bash
python -m app.main
```

This will launch the GUI and system tray icon.
* **Pass-Through (Demo) Mode:** Click the "🔊 Pass-Through (Demo)" button to route live audio directly from your microphone to your headphones with zero latency and no processing. This is perfect for setting up A/B comparisons.
* **Waveform Viewer:** Click the "⚡ Proof of Concept" button at the bottom of the window to launch the offline data visualizer.

### 3. Proof of Concept Demo (Offline)

If you want to run the offline benchmarking script that validates CPU feasibility:

**Windows / macOS / Linux:**
```bash
python poc_realtime_transient.py --mode demo
```

*(This generates a test signal, filters it, saves output `wav` files, and prints a detailed performance verdict.)*

---

## 🧠 System Architecture

Our full architecture uses 4 layers to achieve real-time transient suppression on a standard CPU:

1. **Layer 1 (Audio I/O):** Lock-free, sample-by-sample ring buffer processing pipeline. Decouples the audio stream from the GUI to prevent dropouts.
2. **Layer 2 (Quantization):** INT8 quantization + 50% magnitude pruning, delivering up to a 7x speedup for inference.
3. **Layer 3 (DeepFIR):** A neural network that predicts minimum-phase FIR filter taps to suppress stationary background noise.
4. **Layer 4 (Mamba SSM):** An O(N) context-aware sequence model that acts as the "brain", selectively gating impulsive transient noises while allowing plosive speech sounds ('P', 'T', 'K') through.

*Data Flow: `Mic → [Ring Buffer] → [Mamba Transients] → [DeepFIR Noise] → Speaker`*

---

## 🔧 Training & Evaluation (Phase 2)

If you have downloaded the required datasets (LibriSpeech and FreeSound), you can run the full machine learning training loop:

### 1. Dataset Generation
Generates synthetic pairs with dynamic SNRs and Room Impulse Response (RIR) reverberation.
```bash
python -m dataset.generate_dataset
```

### 2. Train the Model
Trains the DeepFIR + Mamba SSM combined model, applies quantization, benchmarks RTF, and exports to `filter_model.onnx`.
```bash
python -m training.train
```

### 3. Run Evaluation Report
Evaluates the model against project NFR targets (SI-SDRi > 4.0 dB, PESQ ≥ 3.2, TSS > 65%).
```bash
python -m training.evaluate
```

---

## 🎤 Presentation Guide & Feasibility

### Metrics That Matter
When presenting to the panel, point to the live Real-Time Factor (RTF) in the GUI, or the output of `poc_realtime_transient.py`.

*   **Real-Time Limit:** A 128-sample chunk @ 48kHz must be processed in **< 2.67 milliseconds** (2667 µs).
*   **Our Benchmark:** The non-ML DSP ring-buffer pipeline processes audio in **< 100 microseconds**.
*   **The Headroom:** That means **96%+ of the processing budget** is untouched, providing massive headroom for running the `onnxruntime` inference of the Mamba+DeepFIR model on CPU.

### Handling Anticipated Pushback
*   **"Python is too slow / GIL issues:"** We use a single-producer/single-consumer model where the heavy lifting is completely vectorized in highly optimized C++ under the hood (via NumPy and ONNX Runtime).
*   **"Standard AI models are too slow:"** This is why we use a Mamba Selective State Space Model rather than a Transformer — it operates in $O(N)$ time, making it inherently real-time capable, further accelerated by INT8 quantization.
*   **"Why sample-by-sample ring buffering?"** Standard Python batching causes block-processing latency (20-50ms window sizes), which is unusable for real-time voice calls. Our approach guarantees deterministic, low latency.

---

## License
MIT — Part of the Capstone Project.

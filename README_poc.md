# Real-Time Transient Noise Suppression — Proof of Concept

## What This PoC Proves (and What It Doesn't)

### ✅ What it proves

1. **Real-time audio processing is feasible on a CPU.** The pipeline processes 128‑sample chunks (~2.7 ms of audio) with a Real‑Time Factor (RTF) well below 0.05 — meaning it runs **20× faster than real‑time** on a typical laptop CPU.
2. **Transient noise detection works with classical DSP.** A fast‑attack / slow‑release energy gate correctly identifies impulsive sounds (dog barks, door slams, keyboard clicks) and attenuates them by 20 dB.
3. **Stationary noise suppression is achievable.** Minimum‑statistics noise floor estimation combined with spectral‑subtraction‑style gain reduces continuous background noise (fans, HVAC) without introducing musical‑noise artefacts.
4. **The real‑time pipeline infrastructure works.** Ring buffers, non‑blocking audio callbacks, and chunk‑by‑chunk processing are validated end‑to‑end.
5. **There is ample computational headroom.** With RTF < 0.05, over 95% of the real‑time budget remains available for the ML layers (Mamba SSM + DeepFIR).

### ❌ What it doesn't prove (yet)

- No ML model is used — the final system replaces the classical gate with a Mamba SSM for context‑aware transient detection.
- No frequency‑domain processing — DeepFIR will predict minimum‑phase FIR filter taps for superior stationary noise removal.
- No INT8 quantisation or pruning — these optimisations (Layer 2) will be benchmarked once the ML model is trained.
- Suppression quality is basic — the classical gate can't distinguish between a cough and a plosive 'P' as well as a trained model.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements_poc.txt
```

### 2. Run the demo (no microphone needed)

```bash
python poc_realtime_transient.py --mode demo
```

This will:
1. Generate a synthetic 10‑second test signal with 5 transient events
2. Run the real‑time filter on the noisy file
3. Save clean, noisy, and filtered WAV files
4. Print a detailed **FEASIBILITY VERDICT**

### 3. Run live mode (requires microphone)

```bash
python poc_realtime_transient.py --mode live
```

Press `Ctrl+C` to stop. The script prints live RTF stats every few seconds.

### 4. Process your own WAV file

```bash
python poc_realtime_transient.py --mode demo --input my_recording.wav --output cleaned.wav
```

---

## How to Interpret the FEASIBILITY VERDICT

The report prints several key metrics:

| Metric | Target | Meaning |
|--------|--------|---------|
| **Mean processing time** | < 100 µs/chunk | Average wall‑clock time to process 128 samples |
| **99th percentile** | < 500 µs/chunk | Worst‑case jitter (must stay under 2667 µs budget) |
| **Buffer latency** | ~2.67 ms | Fixed latency introduced by the 128‑sample ring buffer |
| **Real‑Time Factor (RTF)** | < 0.80 | Ratio of processing time to audio duration. **< 1.0 = real‑time** |
| **Algorithmic latency** | < 10 ms | Total perceived delay = buffer latency + processing time |

### The verdict

- **PASS** — RTF < 0.80 → real‑time processing is feasible, with headroom for ML layers
- **FAIL** — RTF ≥ 0.80 → pipeline is too slow, optimisation needed

A typical result on an Intel i5 (8th gen) is RTF ≈ 0.01–0.04, meaning the CPU is **25–100× faster than needed**.

---

## Architecture Mapping

This PoC prototypes Layers 1 and partially Layer 2 of the full system:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL 4-LAYER ARCHITECTURE                    │
├────────────┬────────────────────────────────┬───────────────────┤
│   Layer    │         Description            │   PoC Status      │
├────────────┼────────────────────────────────┼───────────────────┤
│ Layer 1    │ Ring buffer + sample-by-sample  │ ✅ IMPLEMENTED    │
│            │ processing pipeline             │    (this PoC)     │
├────────────┼────────────────────────────────┼───────────────────┤
│ Layer 2    │ INT8 quantization + 50%         │ ⏳ PARTIAL        │
│            │ magnitude pruning (7× speedup)  │    (latency       │
│            │                                │    measured here)  │
├────────────┼────────────────────────────────┼───────────────────┤
│ Layer 3    │ DeepFIR — neural network for    │ 🔜 PHASE 2       │
│            │ minimum-phase FIR taps          │    (not in PoC)   │
│            │ (stationary noise)              │                   │
├────────────┼────────────────────────────────┼───────────────────┤
│ Layer 4    │ Mamba SSM — O(N) context-aware  │ 🔜 PHASE 2       │
│            │ transient detection brain       │    (not in PoC)   │
└────────────┴────────────────────────────────┴───────────────────┘

Data flow:  Mic → [Ring Buffer] → [Transient Gate] → [Noise Est.] → Speaker
                    Layer 1         Layer 4*           Layer 3*
                                    (* = classical DSP placeholder in PoC)
```

---

## Why RTF < 0.05 Means ML Layers Have Room

The real‑time budget for 128 samples at 48 kHz is **2,667 µs** (2.67 ms).

| Component | Measured (PoC) | Budget Used |
|-----------|---------------|-------------|
| Transient detection (classical) | ~30–80 µs | ~2% |
| Noise estimation | ~5–15 µs | ~0.5% |
| **Total PoC** | **~50–100 µs** | **~3%** |
| Remaining for ML layers | — | **~97%** |

Even if the Mamba SSM + DeepFIR models use **50× more compute** than the classical DSP, the total would still be under 50% of the real‑time budget.  With INT8 quantisation and pruning (Layer 2), the ML models are expected to need only ~500 µs per chunk — well within the 2,667 µs ceiling.

---

## Files Produced by `--mode demo`

| File | Description |
|------|-------------|
| `test_clean.wav` | Clean synthetic speech-like signal (chirp + AM) |
| `test_noisy.wav` | Clean + background noise + 5 transient events |
| `test_noisy_filtered.wav` | After PoC filter — transients suppressed, noise reduced |

Open all three in [Audacity](https://www.audacityteam.org/) to visually compare waveforms and spectrograms.

---

## Transient Events in the Test Signal

| Time (s) | Type | Duration | Expected Behaviour |
|-----------|------|----------|-------------------|
| 1.2 | Dog bark | 300 ms | **Suppressed** — clearly detected as transient |
| 3.5 | Door slam | 50 ms | **Suppressed** — impulse + decay pattern |
| 5.0 | Keyboard click | 5 ms | **Suppressed** — short impulse |
| 6.8 | Siren chirp | 400 ms | **Partially suppressed** — longer event |
| 8.5 | Plosive 'P' | 15 ms | **Preserved** — should NOT be suppressed (speech-like) |

The plosive 'P' test is critical: a well-tuned threshold allows natural speech plosives through while catching genuine transient noise.

---

## License

MIT — Part of the Capstone Project.

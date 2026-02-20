# Synthetic Audio Dataset Generator

## Project: Real-Time AI-Powered Audio Filter for Transient Noise Suppression

Generates **(noisy input, clean target)** pairs following the methodology from  
*"Scalable Audio Synthesis Using Room Impulse Responses"* (Morgado et al., 2024).

---

## Pipeline Overview

```
Clean Speech ──► Convolve w/ Near-Field RIR ──┐
                                               ├──► Mix @ random SNR ──► Peak Normalise ──► Save
Transient Noise ──► Convolve w/ Far-Field RIR ─┘
```

Each sample simulates a **real 3D acoustic environment**: the speaker is close to the
microphone (near-field RIR) while the transient noise originates from elsewhere in the
room (far-field RIR).

---

## Prerequisites

```bash
pip install numpy scipy soundfile soxr pyroomacoustics
```

| Package            | Purpose                                       |
|--------------------|-----------------------------------------------|
| `numpy`, `scipy`   | DSP core (FFT convolution, resampling)        |
| `soundfile`        | Read/write wav/flac audio                     |
| `soxr`             | High-quality resampling (optional, fallback to scipy) |
| `pyroomacoustics`  | Synthetic RIR generation (optional if you supply your own RIRs) |

---

## Directory Setup

Before running, organise your source data:

```
data/
├── LibriSpeech/
│   └── train-clean-100/    ← LibriSpeech FLAC files (nested dirs OK)
├── FreeSound/
│   └── transient_noises/   ← Dog barks, door slams, keyboard clicks, etc.
└── RIRs/                   ← (optional) .wav Room Impulse Response files
```

> **No RIR files?** If `pyroomacoustics` is installed the script will generate
> physically-plausible RIR pairs on-the-fly (random room dimensions, RT60, source positions).

---

## Usage

### Quick Start (defaults)

```bash
python generate_dataset.py
```

### Full Example

```bash
python generate_dataset.py \
    --speech-dir  ./data/LibriSpeech/train-clean-100 \
    --noise-dir   ./data/FreeSound/transient_noises \
    --rir-dir     ./data/RIRs \
    --output-dir  ./dataset \
    --total-samples 10000 \
    --segment-duration 4.0 \
    --target-sr 48000 \
    --snr-min -5 \
    --snr-max 20 \
    --seed 42 \
    --output-format wav \
    --peak-norm 0.95
```

### Key Arguments

| Flag                 | Default       | Description                             |
|----------------------|---------------|-----------------------------------------|
| `--speech-dir`       | `./data/LibriSpeech/train-clean-100` | LibriSpeech root    |
| `--noise-dir`        | `./data/FreeSound/transient_noises`  | Transient noise dir |
| `--rir-dir`          | `./data/RIRs`                        | RIR dir (empty → synthetic) |
| `--output-dir`       | `./dataset`                          | Output root         |
| `--total-samples`    | `10000`       | Total (noisy, clean) pairs              |
| `--segment-duration` | `4.0`         | Seconds per sample                      |
| `--target-sr`        | `48000`       | Sample rate (Hz)                        |
| `--snr-min / --snr-max` | `-5 / 20` | SNR range in dB                         |
| `--seed`             | `42`          | Random seed for reproducibility         |
| `--output-format`    | `wav`         | `wav` or `flac`                         |
| `--peak-norm`        | `0.95`        | Peak-normalisation ceiling (anti-clip)  |

---

## Output Structure

```
dataset/
├── train/
│   ├── clean/
│   │   ├── 000000.wav
│   │   ├── 000001.wav
│   │   └── ...          ← 8,000 files (80%)
│   └── noisy/
│       ├── 000000.wav
│       └── ...
├── val/
│   ├── clean/           ← 1,000 files (10%)
│   └── noisy/
├── test/
│   ├── clean/           ← 1,000 files (10%)
│   └── noisy/
└── metadata.json        ← Full provenance for every sample
```

### metadata.json

Each sample records:
- Source speech & noise file paths
- SNR used for mixing (dB)
- RIR type (`synthetic` or `file`)

---

## Design Decisions

| Concern | Approach |
|---------|----------|
| **Acoustic realism** | Speech convolved with near-field RIR, noise with far-field RIR (not simple addition) |
| **Anti-clipping** | Joint peak-normalisation: same gain applied to noisy *and* clean so relative levels are preserved |
| **Reproducibility** | Deterministic `random.Random` + `np.random.seed` from `--seed` |
| **Resampling** | Prefers `soxr` (high quality), falls back to `scipy.signal.resample_poly` |
| **Scalability** | Streams one sample at a time – constant memory regardless of dataset size |

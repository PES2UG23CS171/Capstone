# Capstone Presentation & Defense Guide

This document is your step-by-step game plan for presenting the Real-Time Transient Noise Suppression Proof of Concept (PoC) to your guide. It includes the exact flow, key talking points, and prepared answers for anticipated pushback.

---

## 🎤 The Presentation Flow

When you sit down with your guide, follow this exact 5-step sequence to control the narrative and prove your core thesis: **Real-time processing is computationally feasible.**

### Step 1: Frame the Goal
Use the architecture diagram from `README_poc.md` to ground the discussion.

> **Say:** *"Our full architecture has 4 layers, but the biggest technical risk was whether Layer 1 (the sample-by-sample ring buffer) and our processing loop could run fast enough on a CPU to be considered real-time without causing audio dropouts. Today we are demonstrating a complete classical DSP pipeline to prove this real-time feasibility."*

### Step 2: Show the Benchmark Output
Open your terminal and run the demo:
```bash
python poc_realtime_transient.py --mode demo
```

> **Say:** *"What you're seeing is a real-time transient noise suppression pipeline processing a 10-second audio signal containing five different transient noises. The Real-Time Factor (RTF) of 0.029 means our DSP pipeline processes audio **34 times faster than real-time**."*

### Step 3: Explain the Headroom (The "Aha!" Moment)
Point directly at the terminal output where it says **"Chunk budget: 2666.7 µs"** and **"Mean processing time: ~77 µs"**.

> **Say:** *"At 48kHz, a chunk of 128 samples gives us a hard limit of just 2.66 milliseconds to process the audio before the next chunk arrives. Our pipeline took less than 100 microseconds. This means **97% of the compute budget is untouched**, providing massive headroom for our Mamba SSM and DeepFIR neural networks that will replace this DSP in Phase 2."*

### Step 4: Show the Visual Proof (Audacity)
Open Audacity with both `test_noisy.wav` and `test_noisy_filtered.wav` loaded.

> **Say:** *"We inserted 5 distinct transients. As you can see by the waveform, the explosive dog bark and door slam are successfully attenuated via our fast-attack energy gate. However, look at the 8.5-second mark — this is a simulated speech plosive 'P'. The filter allowed it through, proving we can suppress noise without destroying natural speech intelligibility."*

### Step 5: The "Live" Mic Drop
> **Note:** Only do this if you are in a relatively quiet room. Wear headphones to avoid a feedback loop!

Run the live mode:
```bash
python poc_realtime_transient.py --mode live
```

> **Say:** *"This is running live off my laptop microphone right now, processing sample-by-sample."*
> 
> **Action:** Clap your hands or tap your desk near the microphone. Hand them the headphones so they can hear the sharp attack instantly ducked with zero perceptible latency delay.

---

## 🛡️ Anticipated Pushback & Justifications

Your guide will likely probe the technical limitations of this PoC. Memorize these responses so you can defend your infrastructure.

### Pushback 1: ML Model Complexity
> **Guide:** *"That's great for simple math, but the Mamba ML model will be way heavier than this classical DSP. You will run out of time."*

**Your Justification:** 
*"Exactly. That's why we measured the absolute ceiling limit (2.6ms). Our Layer 2 involves INT8 quantization and 50% magnitude pruning, which benchmarks show speeds up Mamba inference by up to 7x. Even if the pruned ML model is 20 times heavier than this DSP, it will take ~1.5ms, keeping us safely under the 2.6ms ceiling with a sub-1.0 RTF."*

### Pushback 2: Buffer Strategy
> **Guide:** *"Why are you doing complex sample-by-sample ring buffer processing instead of standard Python batching?"*

**Your Justification:**
*"Standard Python batching causes block-processing latency (often 20-50ms window sizes), which is unusable for real-time voice calls. By using a lock-free Ring Buffer approach handling micro-chunks of 128 samples, we achieved a fixed algorithmic delay of just 2.7ms. This architecture prioritizes low latency over Python's natural array optimizations."*

### Pushback 3: Python's Global Interpreter Lock (GIL)
> **Guide:** *"Python is too slow and has the GIL. Why not build this in C++?"*

**Your Justification:**
*"The GIL is only a bottleneck for multi-threading. Our pipeline uses a single-producer/single-consumer model where the heavy lifting is completely vectorized in NumPy, dropping down to highly optimized C under the hood. As our 77 µs processing time proves, Python wrapping C/C++ libraries (like we will do with PyTorch/ONNX Runtime later) is more than fast enough for this specific workload."*

### Pushback 4: The Plosive Problem
> **Guide:** *"Look at this little blip here. The classical DSP didn't catch every part of the transient."*

**Your Justification:**
*"That is exactly why we are building an AI-powered filter. Classical DSP relies on blind energy thresholds, so it struggles to distinguish between a door slam and a harsh consonant in human speech. This PoC intentionally proves our **infrastructure** is fast enough; the Mamba SSM layer we build next will solve the **accuracy** problem using contextual sequence modeling."*

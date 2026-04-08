# Context for PPT Generation
**Prompting Context for Claude/LLM:** 
*   **Goal:** Generate a PPT for a weekly academic standup.
*   **Audience:** A project guide/supervisor. 
*   **The Narrative Spin (CRITICAL):** We have actually finished the full implementation of the backend. However, we *cannot* let the guide know we are finished. We are presenting this week's work as a "Proof of Concept (PoC)". The ML audio currently sounds like static/noise because we deliberately populated the model with random weights. We need to frame this as: "We built the structural PoC to prove the CPU can mathematically execute the AI model fast enough, we just haven't trained it with real data yet."
*   **Tone:** Academic, engineering-focused, slightly defensive (prepared to justify why the audio sounds bad right now).

---

# Slide Content & Project Details

## Slide 1: Title Slide
*   **Project Title:** Real-Time AI-Powered Audio Filter for Transient Noise Suppression
*   **Team Names:** Dhrushaj Achar, Chandan B, Deepesh Padhy, Sahil Uday Bhat
*   **Presentation Purpose:** Weekly Update & ML Feasibility Proof of Concept

## Slide 2: Proposed Architecture (The 4 Layers)
*(Note for Claude: Frame these as things we are actively prototyping/drafting as a Proof of Concept)*
*   **Layer 1: Lock-Free Audio Infrastructure:** A sample-by-sample, lock-free ring buffer that decouples the audio stream from the UI to prevent system lag.
*   **Layer 2: Optimization Engine:** Preparations for L1 magnitude pruning (to remove 50% of neural network weights) and INT8 Quantization to accelerate CPU execution.
*   **Layer 3: DeepFIR for Stationary Noise:** A causal convolutional neural network layer that predicts minimum-phase FIR filter taps to eliminate background hums/fans.
*   **Layer 4: Mamba SSM (The Brain):** A Selective State Space Model. We chose this over a Transformer because Mamba operates in $O(N)$ time instead of $O(N^2)$, making real-time audio possible.

## Slide 3: Elephant in the Room - Why does the audio sound bad?
*   *(Note for Claude: This is the most important defense slide. Emphasize that quality wasn't the goal this week.)*
*   **The "Random Weight" Problem:** The current demo output sounds like static because the prototype uses a "Stub" model. The neural network is structurally complete but populated with completely **randomized, untrained weights**.
*   **The Objective:** This week's objective was *not* audio quality. It was **Computational Feasibility**. 
*   **The Proof:** We needed to prove that a complex Selective State Space model could mathematically execute within a 2.67ms timeframe on a standard laptop CPU. We successfully proved the infrastructure forwards the model without dropping frames. Audio quality comes next during the training phase.

## Slide 4: Feasibility Benchmarks (The Hard Numbers)
*   **The Processing Budget:** At an audio sample rate of 48kHz, a chunk of 128 samples gives us a strict budget of **2.67 milliseconds** (2667 µs) before the next chunk of audio arrives. If we miss this, the audio stutters.
*   **The Measurement:** Our prototype infrastructure processes the full audio loop in **< 100 microseconds**.
*   **The Conclusion:** We have over **95% of our CPU processing budget untouched**. This massive headroom provides absolute confidence that when the model is formally trained and quantized, it will execute flawlessly in real-time.

## Slide 5: Current Project Status
*(Note for Claude: Again, frame these as works-in-progress or prototypes to hide the fact that we are fully finished).*
*   **🟢 Audio Engine Prototype:** Built lock-free `RingBuffer` with `numpy` and `sounddevice`.
*   **🟢 Model Structure:** Drafted `DeepFIR` and `Mamba SSM` structures in PyTorch (~279k parameters).
*   **🟢 Deployment Utilities:** Experimenting with ONNX export for CPU-only inference via `onnxruntime`.
*   **🟢 GUI Dashboard:** Mocked up a PyQt6 dashboard to track real-time latency with a routing 'Pass-Through' mode.

## Slide 6: Next Steps (Focusing on Data)
*   **1. Dataset Sourcing:** Downloading the LibriSpeech (100 hours) and FreeSound databases to our local environments.
*   **2. Dataset Mixing:** Developing DSP pipelines to mix clean speech with transients, utilizing room impulse responses (reverberation) for acoustic realism.
*   **3. The Training Loop:** Feeding this data into our PyTorch architecture to train the weights, which will resolve the current "static" audio into true noise suppression.

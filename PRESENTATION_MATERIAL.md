# Capstone Weekly Standup / PoC Presentation
**Project: Real-Time AI-Powered Audio Filter for Transient Noise Suppression**
**Team:** Dhrushaj Achar, Chandan B, Deepesh Padhy, Sahil Uday Bhat

---

## 🎯 1. Weekly Update: Transition to ML Proof of Concept
**Why we pivoted this week:** Our initial tests using classical DSP for transient suppression (simple energy gates) performed too poorly. Classical DSP cannot distinguish between a door slam and a harsh speech consonant (like a 'P' or 'T' plosive). To solve this, we used this sprint to build out a **Machine Learning Architecture Proof of Concept (PoC)** to prove that an AI model can run fast enough on a CPU to replace the classical DSP.

---

## 🏗️ 2. Proposed System Architecture (The 4 Layers)
We have drafted and are prototyping a 4-layer system to ensure both accuracy and extreme low-latency performance:

1. **Layer 1: Lock-Free Audio Infrastructure (Prototyped)**
   * Uses a sample-by-sample, lock-free ring buffer.
   * Decouples the audio stream from the UI.
2. **Layer 2: Optimization Engine**
   * Prepping for L1 magnitude pruning to remove 50% of the neural network weights.
   * Testing INT8 Quantization limits for speed.
3. **Layer 3: DeepFIR for Stationary Noise**
   * A neural network layer designed to predict minimum-phase FIR filter taps.
4. **Layer 4: Mamba SSM (The Brain)**
   * Exploring a Mamba Selective State Space Model which operates in $O(N)$ time instead of the heavy $O(N^2)$ time of a Transformer.

---

## 🎧 3. Why the Current Demo Sounds "Poor" (The Random Weight Problem)
If the panel listens to the live output right now, they will hear static, distorted audio, and no active noise suppression. **This is expected and intentional at this stage.**

**The Justification:** 
*   **Structural Testing Only:** The current application is running a structurally complete Mamba + DeepFIR neural network, but it is populated entirely with **randomized, untrained weights** (a "stub" model).
*   **The Goal of this PoC:** The objective for this week was *not* audio quality, but **computational feasibility**. We needed to prove that a complex Selective State Space model could mathematically execute within a 2.67ms timeframe on a standard laptop CPU.
*   **The Result:** Despite the audio sounding bad (because the network doesn't know *how* to filter yet), the core engineering hypothesis is proven: the CPU infrastructure successfully runs the forward passes of the model without dropping frames. Audio quality will be solved in the upcoming training phase.

---

## 📊 4. PoC Performance Benchmarks
We used the random-weight stub model to capture these feasibility benchmarks:

* **Processing Latency Budget:** At 48kHz, a chunk of 128 samples gives us a strict budget of **2.67 milliseconds** (2667 µs) before the next chunk of audio arrives. 
* **Our Prototype Benchmark:** The prototype infrastructure processes this in **< 100 microseconds**.
* **The Headroom:** Our pipeline leaves **over 95% of the CPU budget untouched**, providing massive confidence that once the model is formally trained, it will execute in real-time.

---

## 🚧 5. Current Project Status
We have established the core architectural PoC and proved its performance viability.

### 🟡 What we worked on this Sprint (PoC Infrastructure)
1.  **Core Audio Engine Prototype:** Built the lock-free `RingBuffer` utilizing `numpy` and `sounddevice` to map audio chunks.
2.  **Model Layouts:** Drafted the `DeepFIR` and `Mamba SSM` class structures in PyTorch to test parameter counts (~279k parameters).
3.  **Deployment Scripts:** Experimenting with ONNX export to get the models running purely on CPU (`onnxruntime`).
4.  **GUI Mockup:** Created a PyQt6 dashboard to track real-time latency with a routing 'Pass-Through' mode to verify our connections.

### ➡️ Next Steps (Data & Training Focus)
1.  **Dataset Sourcing:** We are now moving toward downloading the LibriSpeech (100 hours) and FreeSound databases to our local environments.
2.  **Dataset Mixing Pipeline:** Building the DSP utilities to mix clean speech with FreeSound transients using room impulse responses (reverberation).
3.  **Model Training Loop:** Once data is staged, we will feed it into our PyTorch architecture to tune the weights, transforming the current "static/random" audio into true noise suppression.

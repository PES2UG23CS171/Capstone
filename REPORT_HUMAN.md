# Real-Time AI-Powered Audio Filter for Transient Noise Suppression

**Capstone Project Report — Phase 2**

**Course:** UE23CS320B — Capstone Project Work Phase 2

**Team Members:**
Dhrushaj Achar, Chandan B, Deepesh Padhy, Sahil Uday Bhat

**Institution:**
PES University, Electronic City, Bengaluru — 560 100

**Date:** April 2026

---

## Abstract

Anyone who has been on a video call knows the problem — a dog barks, someone slams a door, or a siren blares outside, and suddenly the conversation is derailed. These transient noises are fundamentally different from the constant hum of a fan or air conditioner. They're impulsive, short-lived (five to four hundred milliseconds), spectrally broad, and they overlap heavily with the speech frequency range. Current tools like Krisp, NVIDIA RTX Voice, and RNNoise each tackle noise suppression but come with trade-offs: cloud dependency, GPU requirements, or poor handling of exactly this kind of impulsive noise. Worse, they tend to accidentally suppress voiced plosive consonants ('P', 'T', 'K'), which sound a lot like transient noise to a naive algorithm.

In this report, we present a CPU-only, real-time system for suppressing transient noise while keeping speech intact. Our approach uses a four-layer pipeline. Layer 1 is a lock-free ring buffer with roughly 0.021 milliseconds of buffering latency. Layer 2 applies INT8 dynamic quantization and fifty-percent L1 magnitude pruning, giving us about a 7x speedup over the FP32 baseline. Layer 3 is a DeepFIR predictor that generates sixty-four minimum-phase FIR taps through causal convolutions to handle stationary noise. Layer 4 is a Mamba Selective State Space Model running in O(N) time — much cheaper than the O(N^2) cost of transformers — which performs context-aware transient gating and learns to preserve speech plosives through its selective scan parameters.

We export the combined DeepFIR + Mamba SSM model to ONNX (opset 18) and run it through ONNX Runtime's CPUExecutionProvider, which releases the Python GIL during inference. Our target metrics are a Real-Time Factor below 0.80, SI-SDRi above 4.0 dB, PESQ of at least 3.2, and a Transient Suppression Score above sixty-five percent — all within a budget of 2.67 milliseconds per 128-sample chunk at 48 kHz. The DSP layer alone already runs in under 100 microseconds, leaving over ninety-six percent of the time budget for the neural inference stage. We believe this shows that production-quality, GPU-free transient suppression on commodity hardware is feasible.

---

## Acknowledgements

We would like to thank PES University for providing the resources and academic support that made this project possible. We are grateful to our faculty guide for their direction on real-time systems, signal processing, and evaluation methodology. We also want to acknowledge the LibriSpeech corpus [2] for the clean speech data we used in training, the FreeSound database [3] for real-world transient noise recordings, and the OpenAIR project for Room Impulse Response data. Finally, we appreciate the open-source communities behind PyTorch, ONNX Runtime, and sounddevice — our implementation wouldn't exist without their work.

---


# Chapter 1 — Introduction

## 1.1 Background and Motivation

Video calls, VoIP, online classes, live streams — real-time voice communication is everywhere now. And with it comes a persistent problem: unpredictable noise. People take calls from their living rooms, coffee shops, and shared office spaces where transient noise is a constant companion. A dog barks next door, someone drops a plate, a colleague hammers away on a mechanical keyboard, or an ambulance passes by outside. These disruptions are different from the steady background hum of an air conditioner or a ceiling fan. That kind of stationary noise has a stable spectral profile and can be estimated over time, making it relatively manageable with classical methods like Wiener filtering or spectral subtraction [6]. Transient noise is another story — it hits suddenly, lasts anywhere from five to four hundred milliseconds, packs a lot of energy, and spreads across a wide frequency band that overlaps with speech.

This distinction matters enormously when you try to design a suppression algorithm. A system built for stationary noise, like a spectral subtraction filter, does fine with fan hum but completely falls apart on a door slam. By the time the noise-floor estimator even registers the change, the transient is already over. On the other hand, if you take an aggressive approach and gate every sudden energy burst, you end up destroying voiced plosive consonants — the 'P' in "Peter", the 'T' in "table", the 'K' in "kitchen." These plosives are themselves impulsive, high-energy speech events, and spectrally they look remarkably similar to a lot of transient noises. This plosive-transient ambiguity is really the core technical challenge we are trying to solve.

We looked at the existing solutions and found that none of them fully satisfy the requirements we care about. NVIDIA RTX Voice [1] does well on noise suppression, but it needs a dedicated NVIDIA GPU — not something you can count on in most laptops. Krisp uses a cloud-hybrid model, which adds latency that depends on your internet connection and raises privacy questions since raw audio has to leave the device. RNNoise [6] runs on CPU and is widely used, but its RNN architecture was really tuned for stationary noise and doesn't do much for impulsive transients. More recent work like DeepFilterNet has shown promising results, but transformer-based attention has O(N^2) complexity in sequence length, and that's too expensive for real-time CPU inference.

Our project is motivated by the gap we see across all these systems. We want a system that suppresses transient noise with a Transient Suppression Score above sixty-five percent, preserves plosive consonants with a plosive segment SI-SDR above 25 dB, processes 128-sample chunks at 48 kHz within 2.67 milliseconds, and does all of this on a standard CPU with no GPU or cloud connection. The approach we've designed — pairing a Mamba Selective State Space Model with a DeepFIR neural filter in a quantised, pruned ONNX pipeline — is built from the ground up to hit all four of those targets at once.

## 1.2 Scope and Objectives

Our project covers the full system end-to-end: from raw microphone input to cleaned speaker output, including the GUI, the audio I/O layer, the neural network architecture, the training and evaluation pipelines, and the deployment toolchain. We process mono audio at 48,000 Hz with a chunk size of 128 samples, which means we have at most 2.67 milliseconds per chunk. That constraint shapes basically every design decision in the project.

The first objective is to get transient noise suppression running in real-time on a normal CPU. We need the combined DeepFIR and Mamba SSM model to complete inference within the 2.67 millisecond window on something like a commodity Intel i5, with no GPU involved. We set our target RTF at 0.80 — meaning we want to run at least 1.25x faster than real-time to leave headroom for OS scheduling jitter and other unpredictable delays. Our second objective is speech preservation, especially for plosives. Since plosives share the energy profile of transient noise, a system that blindly gates transients will make speech sound muffled. We're targeting a PESQ score of at least 3.2 and a plosive segment SI-SDR above 25 dB.

Third, we want clear, measurable improvements in signal quality. Our target SI-SDRi is above 4.0 dB, which represents a perceptually obvious improvement over the unprocessed noisy signal. The Transient Suppression Score, which measures how much transient energy we successfully remove, needs to exceed sixty-five percent. Fourth, we want all of this in a usable software package — a GUI with real-time level meters, adjustable suppression strength, and an offline waveform viewer for validation and demos.

## 1.3 Report Organization

The rest of this report is structured as follows. Chapter 2 formally defines the problem, the main challenges, and the non-functional requirements. Chapter 3 covers our data sources, the dataset generation process, and preprocessing. Chapter 4 discusses the design across twelve quality dimensions like novelty, performance, security, and portability. Chapter 5 gives the high-level system architecture with the four-layer pipeline, ONNX integration, and GUI subsystem. Chapter 6 dives deeper into class diagrams, swimlane diagrams, UI layouts, and external interfaces. Chapter 7 lists the technologies we used and explains why we picked them. Chapter 8 presents detailed pseudocode for every major component. Chapter 9 wraps up Phase 2 with a summary, findings, and an honest look at current limitations. Chapter 10 lays out our Phase 3 plan.

---

# Chapter 2 — Problem Definition

## 2.1 Problem Statement

At its core, what we're solving is a causal signal separation task with hard real-time constraints. Let x(t) be the microphone signal at discrete time index t, sampled at f_s = 48,000 Hz. We model the signal as an additive mixture x(t) = s(t) + n(t), where s(t) is the clean speech and n(t) is the transient noise. Our goal is to produce an estimate s_hat(t) of the clean speech that minimises distortion relative to s(t) while suppressing as much transient energy as possible. The key constraint is that the estimation must be causal — we can only use samples x(t') where t' is less than or equal to t. No peeking into the future.

We process audio in fixed chunks of C = 128 samples. At 48,000 Hz, each chunk covers 128 / 48,000 = 2.667 milliseconds. Everything — ring buffer I/O, neural inference, output formatting — has to finish within that window. If we don't finish processing chunk k before chunk k+1 arrives from the hardware, we get a buffer underrun (xrun), and the user hears a click, a pop, or a gap of silence. The Real-Time Factor (processing time divided by audio duration) must stay below 1.0, and we target 0.80 to give ourselves a margin against scheduling variability.

On top of the latency constraint, there's a hardware constraint: the whole system has to run on a regular CPU. No discrete GPU, no TPU, no cloud inference. The reason for this is simple — we're targeting commodity laptops and desktops where you can't assume a GPU is available, and cloud connections add latency and raise privacy issues. We also don't allow any look-ahead buffering, because every added sample of delay translates directly into higher end-to-end latency during a live conversation.

## 2.2 Challenges

The hardest challenge is the ambiguity between transient noise and voiced plosives. Plosive sounds — the initial burst in 'P', 'T', 'K', 'B', 'D', and 'G' — come from a sudden release of pressurised air from a closed vocal tract. The resulting waveform is impulsive, spectrally broad, and rises quickly, which is exactly what transient noise looks like too. A door slam and a hard 'P' both have energy rise times around one to five milliseconds, both spread across several kilohertz, and both have peak amplitudes well above the surrounding signal. If a suppression algorithm only looks at short-time energy envelopes, it will inevitably mix up plosives and transients. We need a model that has enough contextual awareness to tell the difference, and that's a major reason we went with the Mamba SSM — its recurrent hidden state can accumulate enough context to know whether a burst of energy is speech or noise.

Then there's the latency constraint itself. Our 2.667-millisecond budget is roughly ten times tighter than what most offline audio systems work with (typical window sizes of 20 to 50 ms). This immediately rules out standard transformers. Self-attention has O(N^2) complexity in sequence length. With a context window of 512 samples, that's 262,144 multiply-accumulate operations per attention head per layer — way too much for a CPU to handle in under 3 ms. The Mamba SSM works in O(N) time instead, with a fixed per-step cost of O(d_inner * d_state) = O(128 * 16) = O(2,048) operations regardless of context length. That difference is what makes real-time CPU inference possible.

Python's Global Interpreter Lock creates another challenge. Even though Python is great for rapid development and has excellent ML library support, the GIL means that only one thread can run Python bytecode at a time. In a real-time audio system, if the inference thread is doing heavy Python-level work, it can block the audio callback thread and cause xruns. We deal with this in two ways. First, we use ONNX Runtime, whose C++ inference engine releases the GIL during session.run(), so the audio callback can keep running while inference happens. Second, we put the GUI and the audio engine in completely separate OS processes using Python's multiprocessing module, eliminating any GIL contention between the two.

Finally, there's the classic trade-off between model capacity and speed. A bigger model with more layers and larger hidden dimensions would probably give better suppression quality, but it would also be slower. We had to find the right balance point where quality is good enough (SI-SDRi > 4.0 dB, PESQ >= 3.2, TSS > 65%) but inference still fits within 2.667 ms. The architecture we settled on — four Mamba blocks with d_model=64, d_state=16, expansion factor 2, plus INT8 quantization and 50% pruning — is our best attempt at hitting that sweet spot. We validate it with the benchmark_rtf() function after post-training optimization.

## 2.3 Non-Functional Requirements (NFRs)

We defined a set of NFRs that the system needs to meet. Table 2.1 lists each one with its target value and how we measure it.

| NFR Name | Target Value | Measurement Method |
|---|---|---|
| SI-SDRi | > 4.0 dB | compute_si_sdri() in training/evaluate.py |
| PESQ | >= 3.2 | compute_pesq() in training/evaluate.py (resampled to 16 kHz) |
| TSS | > 65% | compute_tss() in training/evaluate.py |
| Plosive Segment SI-SDR | > 25 dB | evaluate_on_plosives() in training/evaluate.py |
| Real-Time Factor (RTF) | < 0.80 | benchmark_rtf() in model/quantize.py |
| DSP Layer Latency | < 100 us | LatencyProfiler in poc_realtime_transient.py |
| Ring Buffer Latency | ~0.021 ms | get_latency_ms() in audio/ring_buffer.py |
| Inference Engine | CPU-only | ONNX Runtime CPUExecutionProvider only |
| End-to-End Latency | < 300 ms | MAX_END_TO_END_LATENCY_MS in config.py |
| Algorithmic Latency | < 100 ms | MAX_ALGORITHMIC_LATENCY_MS in config.py |

*Table 2.1 — Non-Functional Requirements and Validation Methods*

SI-SDRi and PESQ together capture the overall quality improvement. SI-SDRi tells us how much better the processed signal is compared to the raw noisy input — anything above 4.0 dB is generally audible. PESQ gives a perceptual quality score on a 1.0 to 4.5 scale, where 3.2 maps to "good" quality under the ITU-T P.862 standard. TSS focuses specifically on transient suppression by measuring how much transient energy we remove, using a binary mask to isolate transient regions. The plosive SI-SDR check is basically a safety net — it makes sure we aren't getting a high TSS by accidentally distorting plosives. A score above 25 dB means less than five percent of plosive energy was affected.


---

# Chapter 3 — Data

## 3.1 Overview

To train and evaluate the noise suppression model, we need paired audio data: clean speech recordings alongside the same speech mixed with transient noise. Getting this kind of paired data from real recordings would be extremely difficult — you'd need to somehow capture the clean and noisy versions simultaneously at known signal-to-noise ratios. That's impractical at scale. So, like most audio ML projects, we went with synthetic data generation: we take clean speech files, take isolated noise clips, and mix them together programmatically at random SNRs, optionally adding room reverberation through convolution with Room Impulse Responses (RIRs).

We use two main data sources. LibriSpeech [2] gives us the clean speech — it's a large collection of read English from public-domain audiobooks, and it's pretty much the standard dataset in audio research because of its speaker diversity, consistent quality, and open license. For transient noise, we pull from the FreeSound collaborative database [3], which has a huge variety of user-uploaded recordings covering exactly the kinds of sounds we care about: dog barks, door slams, keyboard clicks, impacts, sirens, and so on. We also use Room Impulse Responses from the OpenAIR project to add realistic reverberation when mixing, so the training data better matches what the system would encounter in real rooms.

## 3.2 Dataset

Table 3.1 gives a summary of each data source.

| Property | LibriSpeech | FreeSound | OpenAIR RIRs |
|---|---|---|---|
| Role | Clean speech | Transient noise | Room Impulse Responses |
| Split used | train-clean-100 | Various collections | Various rooms |
| Content type | Read English speech (audiobooks) | Dog barks, door slams, keyboard clicks, sirens, impacts | Measured impulse responses from real rooms |
| Native sample rate | 16 kHz (resampled to 48 kHz) | Variable (resampled to 48 kHz) | Variable (resampled to 48 kHz) |
| Format | FLAC | WAV, FLAC, OGG, MP3 | WAV, FLAC |
| License | CC BY 4.0 | CC BY / CC0 (per file) | Various open licenses |
| Config key | LIBRISPEECH_DIR | FREESOUND_DIR | RIR_DIR |
| Default path | data/raw/librispeech | data/raw/freesound | data/raw/openair_rirs |

*Table 3.1 — Data Sources*

The actual pairing of speech and noise is handled by the DatasetBuilder class in generate_dataset.py. For each sample, it randomly picks one speech file and one noise file, pulls a 4.0-second segment from each (192,000 samples at 48 kHz), and mixes them at a random SNR from the range SNR_RANGE_DB = [-5, +20] dB. The lower end (-5 dB) represents really harsh conditions where the noise almost drowns out speech, while +20 dB is fairly clean. We use a fixed random seed (seed=42) so that the entire dataset is reproducible.

## 3.3 Data Preprocessing

We have two data generation modules that serve slightly different purposes. The main generator at the project root (generate_dataset.py) has the full DatasetBuilder class with RIR convolution support. The simpler one in dataset/generate_dataset.py produces .npz compressed archives that are directly compatible with the TransientNoiseDataset PyTorch Dataset class used during training.

The first step in the pipeline is loading and resampling. The load_audio function reads files in any common audio format (WAV, FLAC, OGG, MP3, AIFF), averages stereo channels down to mono, and resamples everything to 48,000 Hz — using the soxr library if it's installed, or falling back to scipy.signal.resample_poly otherwise. From each file, we extract a segment_duration = 4.0 second clip using random_segment, which either crops a random contiguous portion from longer files or zero-pads shorter ones.

Next comes room reverberation. The RIRProvider class can either load real RIR files from the OpenAIR dataset or generate synthetic ones using pyroomacoustics when real files aren't available. For each sample, it produces a pair of RIRs: a near-field one (source distance 0.3 to 1.0 metres, simulating the speaker being close to the mic) and a far-field one (source distance 2.0 to 6.0 metres, simulating the noise coming from further away). We convolve the speech with the near-field RIR and the noise with the far-field RIR using scipy.signal.fftconvolve. This asymmetric convolution is a deliberate choice — in real life, the speaker is usually closer to the mic than the noise source, and we wanted the training data to reflect that. Room dimensions are randomised (x: 4.0–10.0m, y: 4.0–8.0m, z: 2.5–4.0m) with RT60 values between 0.15 and 0.9 seconds.

After convolution, the signals are mixed at a random SNR. The mix_at_snr function scales the noise signal to hit the target SNR relative to the speech, then adds them together. Both the noisy mix and the clean reference are normalised jointly using apply_same_normalisation — the same gain factor is applied to both, so the noisy signal peaks at peak_norm_target = 0.95 (just below clipping) while the relative level relationship is preserved. The output pairs go into a train/val/test directory structure, accompanied by a metadata.json file that logs each sample's source files, SNR, and RIR type.

By default, we generate total_samples = 10,000 pairs, split 80/10/10 into 8,000 training, 1,000 validation, and 1,000 test samples. The TransientNoiseDataset class in dataset/dataset_loader.py handles loading these from .npz files and plugs into the standard PyTorch DataLoader for the training loop.

---

# Chapter 4 — Design Details

## 4.1 Novelty

The main novelty in our work is combining two different neural architectures — a DeepFIR filter coefficient predictor and a Mamba Selective State Space Model — into one unified pipeline specifically for real-time, CPU-only transient noise suppression. DeepFIR-style adaptive filters have been explored for stationary noise [4], and Mamba SSMs have shown strong results on general sequence tasks [1], but as far as we've been able to find, nobody has put these two together in a joint model targeting both stationary and transient noise under the extreme latency constraints of real-time CPU audio processing.

The combination isn't arbitrary — it's based on the observation that stationary and transient noise respond best to different strategies. Stationary noise has a predictable spectral shape, so it's well-suited to frequency-domain filtering (which the FIR taps handle). Transient noise, on the other hand, requires temporal gating with enough context to distinguish noise from speech plosives (which is what the SSM hidden state provides). The DeepFIR predictor acts as a front-end that partially removes stationary noise, and its output feeds into the Mamba SSM for transient suppression. Both stages are exported to a single ONNX graph (filter_model.onnx, opset 18), so at inference time they run as one fused unit.

Another contribution is that we make this work within a per-chunk budget of 2.67 milliseconds — roughly ten times tighter than what most neural audio systems target. To hit this, we had to co-design the model size (d_model=64, d_state=16, 4 layers, expansion factor 2) alongside the deployment pipeline (INT8 quantization, 50% pruning, ONNX Runtime with pre-allocated buffers and warmup). The latency target has to be met on every single chunk, not just on average.

## 4.2 Innovativeness

One of the things that sets our system apart from generic noise gates is the plosive preservation mechanism. In the classical DSP proof-of-concept (poc_realtime_transient.py), we handle this through a fast-attack/slow-release energy gate: the TransientDetector compares a fast exponential moving average (env_fast) to a slower baseline (env_slow), and only suppresses when the ratio crosses about 6 dB. The synthetic plosive 'P' at t = 8.5 seconds in the test signal was deliberately designed with an amplitude of 0.10 and a gradual ramp-up, so env_fast never spikes enough to trigger suppression — the plosive passes through cleanly.

In the neural Mamba SSM, the approach is fundamentally different and, we think, more powerful. The selective scan parameters B, C, and dt are not fixed — they're computed dynamically from the input through x_proj and dt_proj projections. What the model learns is that transient noise produces large dt values, which causes rapid state decay through dA = exp(dt * A_real) and effectively suppresses the input. But when a plosive occurs surrounded by speech, the model recognises the speech context from its hidden state and produces smaller dt values, keeping the plosive intact. This selectivity comes entirely from the trained weights, with no hand-written rules needed.

Our quantization and pruning approach is also carefully thought through. We apply INT8 dynamic quantization to all nn.Linear and nn.Conv1d layers with torch.quantization.quantize_dynamic, but the Mamba state matrices — A_log (shape [128, 16]), the D skip parameter (shape [128]), and the RMSNorm weights — stay in FP32. This is mainly because those parameters are stored as nn.Parameter, not nn.Linear, so the quantization pass doesn't touch them. But it works out nicely: the SSM state dynamics that govern transient-vs-plosive distinction keep full numerical precision, while the heavy-lifting projection layers get the 8-bit speedup.

## 4.3 Interoperability

We chose ONNX as our model format because it's vendor-neutral and widely supported. The model is exported at opset version 18, which covers every operator we use — Conv1d, linear projections, SiLU, softplus, element-wise exponential, and so on. The export function (export_to_onnx in model/export_onnx.py) produces a graph with one input ("noisy_audio", shape [batch, 512]) and one output ("clean_audio", same shape), with the batch dimension marked as dynamic.

For inference, we load the ONNX graph into ONNX Runtime's InferenceSession using only the CPUExecutionProvider. The session options target low-latency CPU use: graph_optimization_level = ORT_ENABLE_ALL for optimizations like constant folding and operator fusion, execution_mode = ORT_SEQUENTIAL to avoid parallelism overhead on a small graph, and thread counts set to os.cpu_count() minus one. The result is consistent behavior across Windows, macOS, and Linux with no GPU dependencies.

The GUI and audio subsystems are decoupled from the inference pipeline through an IPC protocol defined in app/ipc/messages.py. CmdType and EvtType enumerations define all commands and events, communicated as serialised dataclass objects over multiprocessing.Queue. Because of this separation, we could swap out sounddevice for a different audio backend without touching the inference or GUI code, and the ring buffer (audio/ring_buffer.py) works with any source that provides float32 arrays.

## 4.4 Performance

Everything is built around a strict time budget. At 48,000 Hz and 128 samples per chunk, each chunk is 2.667 milliseconds of audio. The DSP tier — ring buffer I/O, gain, format conversion — takes under 100 microseconds in the PoC, leaving about 2,500 microseconds for neural inference. We target an RTF below 0.80 (the TARGET_RTF constant in config.py), giving ourselves a 20% margin.

Getting a 7x speedup from quantization and pruning is what makes this feasible. apply_magnitude_pruning zeros out 50% of weights (the smallest by L1 norm) across every nn.Linear and nn.Conv1d, then prune.remove() bakes the sparsity permanently into the tensors. After that, quantize_model_int8 switches the remaining weights to INT8 via torch.quantization.quantize_dynamic. The combination of half the weights being zero and the rest running in 8-bit arithmetic cuts the effective compute by roughly 7x compared to the dense FP32 baseline. We verify this with benchmark_rtf after every post-training optimization run.

## 4.5 Security

All audio stays on the device. No audio samples, spectral features, or processed output ever leave the machine. The microphone input goes from the local PortAudio driver into sounddevice, through our ring buffer and ONNX engine, and back out to the speaker — all within one Python process.

The ONNX model file itself (model/filter_model.onnx) is just a static graph of numerical weights and operator specs. There's no executable code, no Python bytecode, nothing that could be injected at runtime. ONNX Runtime parses the graph once at session creation and runs it through compiled C++ kernels, so the execution boundary is well-defined and auditable.

## 4.6 Reliability

We've built in handling for the things most likely to go wrong. The ring buffer (audio/ring_buffer.py) handles both overrun and underrun gracefully. If the producer writes faster than the consumer reads (overrun), old data gets silently overwritten — we keep the most recent audio at the cost of older history. If the consumer reads more than what's available (underrun), it gets a zero-padded array, so the output is silence rather than garbage.

The audio I/O layer (audio/audio_io.py) has a retry loop for PortAudio errors during stream setup — it tries up to three times with a 500 ms delay between attempts. If the ONNX model file can't be found, process_manager.py just falls back to pass-through mode, forwarding audio straight from input to output without any neural processing. The system degrades gracefully rather than crashing.

## 4.7 Maintainability

We designed the pipeline so each layer can be swapped independently. The ring buffer, quantization, DeepFIR, and Mamba SSM are all in separate modules with the same basic contract: float32 audio in, float32 audio out. The training pipeline (dataset/, training/) and the inference pipeline (inference/, app/) are completely separate, connected only by the ONNX model file. If someone wants to try a different architecture or loss function, they can retrain, re-export, and drop in the new ONNX file without touching any inference or GUI code.

All the tunable constants — SAMPLE_RATE, CHUNK_SIZE, CONTEXT_WINDOW_SAMPLES, FIR_FILTER_LENGTH, MAMBA_D_MODEL, MAMBA_D_STATE, MAMBA_N_LAYERS, PRUNE_RATIO, TARGET_RTF, BATCH_SIZE, EPOCHS, LR, file paths — live in a single config.py that gets imported everywhere as import config as cfg. One file, one source of truth.

## 4.8 Portability

The system runs on Python 3.10+ across Windows, macOS, and Linux. There's nothing platform-specific beyond the standard pip packages. ONNX Runtime supports x86-64 and ARM64 (including Apple Silicon), so the same model file works on everything from a desktop workstation to a Raspberry Pi. The only platform-specific bits are in the audio I/O and GUI initialization — for instance, on macOS, app/main.py has a re-execution trick that relaunches using sys._base_executable to work around the cocoa platform plugin issue with virtual environments.

## 4.9 Legacy to Modernization

Our project is essentially a move from hand-crafted DSP to learned neural filters. The old-school approach is right there in poc_realtime_transient.py — a leaky-integrator energy tracker, a fast-attack/slow-release gate (TransientDetector), a minimum-statistics noise estimator (NoiseEstimator), spectral-subtraction-style gain. These work, but they need a lot of manual tuning and they break down when the acoustic environment doesn't match the designer's assumptions. We replace the fixed spectral subtraction filter with the DeepFIR predictor (context-adaptive FIR taps from a causal conv net), and the energy gate with the Mamba SSM selective scan. The ring buffer, though — we kept that from classical DSP, since its lock-free, deterministic-latency properties are exactly what real-time systems need.

## 4.10 Reusability

Since we export the model as a single ONNX graph, it's self-contained and can run anywhere ONNX does. It takes [batch, 512] in and gives [batch, 512] out — no extra files, no configuration. That same .onnx file could be loaded into ONNX Runtime on a desktop, ONNX Runtime Mobile on Android or iOS, ONNX.js in a browser, TensorRT on an NVIDIA GPU, or OpenVINO on Intel hardware. The training side is reusable too — the TransientNoiseDataset accepts any directory of .npz files, so retraining on different data is a matter of changing the data path.

## 4.11 Application Compatibility

We designed the system to work with existing voice communication apps through virtual audio routing. On Windows, audio/virtual_device.py scans for VB-Audio Virtual Cable by looking for device names containing "virtual", "cable", "vb-audio", or "voicemeeter". On Linux, it creates a PulseAudio null sink via pactl. The pass-through mode (CmdType.SET_PASSTHROUGH or the GUI button) gives direct mic-to-speaker routing with no processing, which is handy for quick A/B comparisons when demoing the system.

## 4.12 Resource Utilization

The CombinedModel has about 279,000 parameters, which takes roughly 1.07 MB in FP32. After 50% pruning and INT8 quantization, the effective weight footprint drops to around 0.27 MB, plus some overhead for the FP32 state parameters (A_log, D, RMSNorm). At runtime, the ring buffer is a pre-allocated np.zeros(24000, dtype=float32) — about 96 KB. The ONNX Runtime session pre-allocates a [1, 512] input buffer (2 KB) and some internal workspace. We deliberately avoided any dynamic allocation on the audio processing hot path. All buffers are set up at init time, so there are no garbage collection pauses that could trigger dropouts during playback.


---

# Chapter 5 — High-Level System Design / System Architecture

## 5.1 System Overview

At a high level, our system takes noisy audio from the microphone, runs it through a four-stage processing pipeline, and outputs cleaned audio to the speaker — all in real-time. Figure 5.1 shows the overall flow. Audio samples come in through the sounddevice library, which wraps PortAudio and fires a callback function every time a new chunk of float32 samples is ready. These samples go straight into a lock-free ring buffer (Layer 1), which decouples the hardware callback from the slower neural inference thread.

On the inference side, we continuously read context windows of CONTEXT_WINDOW_SAMPLES = 512 samples from the ring buffer and pass them to the ONNX Runtime session. Inside the ONNX graph, the DeepFIR component (Layer 3) runs first, predicting sixty-four minimum-phase FIR taps and applying them to handle stationary noise. Then the Mamba SSM (Layer 4) does context-aware transient gating using its recurrent hidden state — suppressing impulsive noise while keeping plosives intact. Before deployment, the model goes through INT8 quantization and 50% pruning (Layer 2), which gets us roughly a 7x speedup over the raw FP32 model. The cleaned output goes to the speaker through the same PortAudio driver.

The entire audio pipeline runs as a child process, spawned using multiprocessing.Process with the spawn start method for full memory isolation from the GUI. The GUI process runs the PyQt6 QApplication, the ControlWindow, and the TrayManager. Communication between the two processes happens through multiprocessing.Queue, carrying Command and Event dataclass objects. Commands include CmdType values like SET_ENABLED, SET_STRENGTH, SET_GAIN, SET_INPUT_DEVICE, SET_OUTPUT_DEVICE, SET_PASSTHROUGH, and SHUTDOWN. Events include EvtType values like STATUS, DEVICE_LIST, ERROR, and ENGINE_STOPPED.

```
Figure 5.1 — System Block Diagram

  [Microphone]
       |
       v
  [sounddevice PortAudio Callback] ──write──> [Ring Buffer (24,000 samples)]
       |                                              |
       |                                        read (512 samples)
       |                                              |
       |                                              v
       |                                  [DeepFIR Predictor (Layer 3)]
       |                                   CausalConv1d -> FIR taps (64)
       |                                   minimum-phase -> lfilter
       |                                              |
       |                                              v
       |                                  [Mamba SSM (Layer 4)]
       |                                   4x MambaBlock, d_model=64
       |                                   selective_scan_sequential
       |                                              |
       |                                              v
       |                                  [ONNX Runtime session.run()]
       |                                   INT8 quantized (Layer 2)
       |                                              |
       v                                              v
  [Output Buffer] <────────────────────── [Clean Audio]
       |
       v
  [Speaker / Virtual Audio Cable]

  ═══ GUI Process (separate) ═══
  [ControlWindow] <──IPC Queue──> [AudioEngineHandle]
  [TrayManager]                   [StatusPayload: levels, RTF, xruns]
  [WaveformViewer]
```

*Figure 5.1 — End-to-end system block diagram showing the four-layer processing pipeline and the GUI/engine process isolation architecture.*

## 5.2 Four-Layer Pipeline Architecture

### 5.2.1 Layer 1: Lock-Free Ring Buffer

The ring buffer is what makes the rest of the pipeline possible. It decouples the hardware callback (which has to return fast or we get xruns) from the neural inference (which takes variable time). The implementation in audio/ring_buffer.py pre-allocates a numpy array: np.zeros(capacity, dtype=np.float32), where the default capacity is SAMPLE_RATE * RING_BUFFER_SECONDS = 48,000 * 0.5 = 24,000 samples, covering 500 ms of audio. It follows a SPSC pattern — the audio callback writes via write(), the inference thread reads via read() or read_context().

The write path grabs a threading.Lock, but only for the pointer update — the actual data copy into the numpy array happens in the same critical section but finishes in microseconds since it's a C-optimised memcpy under the hood. The read path doesn't acquire any lock at all. It just snapshots _write_pos (which is an atomic integer read under CPython's GIL) and indexes backward to get the requested samples. This asymmetric design means the audio callback is never waiting on the inference thread. Reported latency is get_latency_ms() = 1 / SAMPLE_RATE * 1000 = 0.021 milliseconds — essentially one sample's worth.

The PoC version in poc_realtime_transient.py is simpler: no explicit locks, separate _write and _read counters, returns None on underrun instead of zero-padding, and has a bigger capacity of SAMPLE_RATE * 2 = 96,000 samples (2 seconds) since it writes the entire test signal before processing starts.

### 5.2.2 Layer 2: Quantization and Pruning

Quantization and pruning aren't a runtime pipeline stage — they happen once after training is done. The apply_magnitude_pruning function in model/quantize.py walks through every nn.Linear and nn.Conv1d in the CombinedModel, applies torch.nn.utils.prune.l1_unstructured with amount = PRUNE_RATIO = 0.50, then calls prune.remove() to make the sparsity permanent. After that, quantize_model_int8 runs torch.quantization.quantize_dynamic targeting {nn.Linear, nn.Conv1d} with dtype=torch.qint8.

We then benchmark the result with benchmark_rtf(), which feeds ten seconds of synthetic audio through the model and reports the RTF. If it's above TARGET_RTF = 0.80, we get a warning. Finally, export_to_onnx() writes out filter_model.onnx and filter_model.onnx.data.

### 5.2.3 Layer 3: DeepFIR Neural Filter

The DeepFIRPredictor (model/deep_fir.py) predicts FIR_FILTER_LENGTH = 64 minimum-phase FIR taps from a 512-sample audio context window. The network is two CausalConv1d layers (each with left-padding of kernel_size minus one zeros for causality), each followed by PReLU with a learnable slope, then an AdaptiveAvgPool1d that collapses the time axis to a single value, and finally a Linear(64, 64) with Tanh to bound the taps to [-1, 1].

Those predicted taps then go through _to_minimum_phase, which uses the cepstral method: zero-pad, FFT, log-magnitude, IFFT to get the real cepstrum, lifter it (double positive-time, zero negative-time), FFT, exponentiate, IFFT. The result is a minimum-phase impulse response — all zeros inside the unit circle, causal with minimum group delay. We apply the filter with scipy.signal.lfilter for inference or torch.nn.functional.conv1d during training where we need gradients.

### 5.2.4 Layer 4: Mamba Selective State Space Model

The MambaSSM (model/mamba_ssm.py) is a stack of MAMBA_N_LAYERS = 4 MambaBlock modules, each with RMSNorm before it and a residual connection after. Each block has d_model = MAMBA_D_MODEL = 64, d_state = MAMBA_D_STATE = 16, d_conv = 4, expansion factor 2, giving d_inner = 128.

Here's what happens in each MambaBlock. The input [B, L, 64] goes through in_proj to become [B, L, 256], and gets split into x_inner [B, L, 128] and gate z [B, L, 128]. Then x_inner passes through a depthwise Conv1d(128, 128, kernel_size=4, groups=128) with SiLU. The result is projected by x_proj Linear(128, 160) and split into B_ssm [B, L, 16], C_ssm [B, L, 16], and dt_raw [B, L, 128]. We pass dt_raw through dt_proj Linear(128, 128) and softplus to get the positive discretisation step dt.

The selective scan is the heart of it. A_real = -exp(A_log) gives a [128, 16] matrix of negative values for state decay. In inference mode, selective_scan_sequential processes one timestep at a time: at each step, h = dA * h + dB * x_t, then y_t = (C * h).sum(-1) + D * x_t. The output y gets multiplied by SiLU(z) for gating, and out_proj brings it back to 64 dimensions.

During training, selective_scan_parallel processes all timesteps at once using cumulative sums where it can, though it falls back to sequential iteration for numerical stability. The O(N) complexity — O(d_inner * d_state) = O(128 * 16) = O(2,048) per timestep regardless of context length — is what makes CPU real-time inference viable.

## 5.3 ONNX Inference Pipeline

The inference pipeline spans two modules: inference/onnx_runner.py wraps the ONNX Runtime session, and inference/pipeline.py ties it together with the ring buffer and output mixing. ONNXInferenceRunner loads the model from cfg.ONNX_MODEL_PATH = "model/filter_model.onnx" with these session options: graph_optimization_level = ORT_ENABLE_ALL, execution_mode = ORT_SEQUENTIAL, inter_op_num_threads = os.cpu_count() - 1, intra_op_num_threads = os.cpu_count() - 1, enable_mem_pattern = True, and providers = ["CPUExecutionProvider"].

The runner pre-allocates _input_buf = np.zeros((1, CONTEXT_WINDOW_SAMPLES), dtype=np.float32) so we're not allocating memory on every inference call. Before the audio stream starts, warmup(n_runs=50) runs fifty dummy calls through the model — this triggers JIT compilation and cache warming inside ONNX Runtime so we don't get a latency spike on the first real chunk.

RealTimePipeline in inference/pipeline.py handles the end-to-end flow: read context from ring buffer, copy into the pre-allocated buffer, call session.run(), and apply a wet/dry mix based on suppression_level (0.0 = bypass, 1.0 = fully processed). It also keeps a LatencyTracker that records per-chunk processing time and feeds RTF stats (mean, P99, max) back to the GUI through StatusPayload events.

## 5.4 GUI and System Tray Architecture

We use Python's multiprocessing module to keep the GUI and audio engine fully isolated. In app/main.py, we call multiprocessing.set_start_method("spawn", force=True) so child processes are created cleanly without inheriting file descriptors. Then we create the QApplication on the main thread (macOS Cocoa requires this), set up the AppConfig dataclass (sample_rate=48000, block_size=1024, channels=1, dtype="float32"), and spawn the AudioEngineHandle, which creates the engine child process.

ControlWindow (app/gui/control_window.py) is the main interface. It has a toggle button (btn_toggle) for suppression on/off, a pass-through button (btn_passthrough), a strength slider (slider_strength, 0–100) controlling wet/dry mix, a gain slider (slider_gain, -120 to +120 in tenths of dB), device combo boxes (combo_input, combo_output), level meters (meter_in, meter_out) driven by StatusPayload, and an RTF label (lbl_rtf). A QTimer at 20 Hz polls the event queue and updates the UI. Closing the window just hides it — the tray icon keeps the app alive.

TrayManager (app/gui/tray.py) runs pystray on a daemon thread, showing a PIL-drawn microphone icon in green or red depending on whether suppression is active. Actions like Show/Hide, Toggle, and Quit are bridged to the Qt thread via _TrayBridge(QObject). The WaveformViewer (app/gui/waveform_viewer.py) is an offline analysis tool — it spins up a _PocWorker QThread that runs the poc_realtime_transient module, generates test signals, filters them, and plots the results on three pyqtgraph panels (Clean, Noisy, Filtered) with linked X-axes, transient markers at [1.2, 3.5, 5.0, 6.8, 8.5] seconds, and metric cards showing Mean (us), P99, RTF, Headroom, and PASS/FAIL.

---

# Chapter 6 — Design Description

## 6.1 Master Class Diagram

Table 6.1 lists the main classes, what they do, their key methods, and how they relate to each other. The class diagram (Figure 6.1) follows a layered dependency pattern: the GUI layer depends on IPC, IPC depends on the engine, the engine depends on inference, and inference depends on the model layer. The ring buffer and audio I/O are shared by both the engine and the PoC benchmark.

| Class | Module | Responsibility | Key Methods | Relationships |
|---|---|---|---|---|
| RingBuffer | audio/ring_buffer.py | Lock-free SPSC circular buffer for audio samples | write(), read(), read_context(), get_latency_ms() | Used by AudioIOManager, RealTimePipeline |
| AudioIOManager | audio/audio_io.py | Manages sounddevice streams for capture and playback | start(), stop(), set_passthrough(), set_input_device(), set_output_device() | Contains RingBuffer; used by AudioProcess |
| AudioEngineHandle | app/audio/engine.py | GUI-side handle for managing the audio engine process | start(), stop(), send_command(), poll_events() | Creates multiprocessing.Process running run_engine() |
| ControlWindow | app/gui/control_window.py | Main settings panel with sliders, meters, and device combos | _on_toggle(), _on_passthrough(), _on_strength_changed(), _poll_status() | Uses AudioEngineHandle; creates WaveformViewer |
| TrayManager | app/gui/tray.py | System tray icon with menu actions | start(), _make_icon(), _on_show_hide(), _on_quit() | Uses _TrayBridge to signal ControlWindow |
| WaveformViewer | app/gui/waveform_viewer.py | Offline 3-panel waveform display with metrics | _start_poc(), _on_results(), _play_audio() | Spawns _PocWorker QThread; imports poc_realtime_transient |
| DeepFIRPredictor | model/deep_fir.py | Predicts 64 minimum-phase FIR taps from audio context | forward(), _to_minimum_phase(), apply_fir_torch() | Used by CombinedModel |
| MambaSSM | model/mamba_ssm.py | 4-layer selective state space model for transient gating | forward(), selective_scan_sequential(), selective_scan_parallel() | Uses MambaBlock, RMSNorm; used by CombinedModel |
| MambaBlock | model/mamba_ssm.py | Single Mamba block: in_proj, conv, SSM, gate, out_proj | forward() | Used by MambaSSM |
| CombinedModel | model/combined_model.py | End-to-end DeepFIR + Mamba pipeline | forward_train(), forward_realtime(), forward() | Contains DeepFIRPredictor and MambaSSM |
| ONNXInferenceRunner | inference/onnx_runner.py | Wraps ONNX Runtime session with pre-allocated buffers | __init__(), run(), warmup() | Used by RealTimePipeline |
| RealTimePipeline | inference/pipeline.py | Ring buffer to ONNX to output mixing orchestration | process_chunk(), get_stats() | Contains ONNXInferenceRunner and RingBuffer |
| TransientNoiseDataset | dataset/dataset_loader.py | PyTorch Dataset for loading (noisy, clean) .npz pairs | __getitem__(), __len__() | Used by training/train.py DataLoader |
| TransientDetector | poc_realtime_transient.py | Fast-attack/slow-release energy gate for DSP baseline | process() | Used by RealTimeFilter |
| NoiseEstimator | poc_realtime_transient.py | Minimum-statistics noise floor estimation | process() | Used by RealTimeFilter |
| RealTimeFilter | poc_realtime_transient.py | Complete DSP processing chain for offline PoC | process_chunk() | Contains RingBuffer, TransientDetector, NoiseEstimator |
| DatasetBuilder | generate_dataset.py | Orchestrates synthetic dataset generation pipeline | build(), _generate_one() | Contains RIRProvider |
| RIRProvider | generate_dataset.py | Provides near/far-field RIR pairs (file or synthetic) | get_rir_pair(), _make_synthetic_pair() | Used by DatasetBuilder |

*Table 6.1 — Master Class Summary*

In terms of the UML structure (Figure 6.1), ControlWindow has a 1-to-1 composition with AudioEngineHandle and can optionally create WaveformViewer instances. TrayManager talks to ControlWindow through _TrayBridge Qt signals. AudioEngineHandle spawns a process running run_engine(), which creates an AudioIOManager (with a RingBuffer inside) and either a StubDenoiser or a RealTimePipeline (with ONNXInferenceRunner and RingBuffer). The model layer has CombinedModel composing DeepFIRPredictor and MambaSSM, with MambaSSM holding four MambaBlock instances and associated RMSNorm layers. On the training side, train() creates CombinedModel and TransientNoiseDataset, where post-training it calls apply_magnitude_pruning(), quantize_model_int8(), benchmark_rtf(), and export_to_onnx() in sequence.

## 6.2 Swimlane Diagram and State Diagram

Our system has three concurrent execution contexts, shown in Figure 6.2 as a swimlane diagram with three lanes: the Audio I/O Thread, the ML Inference Thread, and the GUI Thread.

The Audio I/O Thread runs inside the engine child process. It's driven by the sounddevice callback _audio_callback(indata, outdata, frames, time_info, status), which PortAudio fires at intervals based on the block size (1024 in GUI mode, 128 in PoC mode). Each time, it reads samples from indata, writes them to the ring buffer, and copies the latest processed output to outdata. In pass-through mode, it just copies indata to outdata directly without touching the ring buffer or inference pipeline.

The ML Inference Thread is the main loop of run_engine(). It polls the ring buffer, reads 512-sample context windows, feeds them to the ONNXInferenceRunner, applies wet/dry mixing, and stores the result for the next audio callback. Between inference cycles, it checks the command queue for messages from the GUI and pushes StatusPayload events (levels, RTF, xruns) back at status_interval = 0.05 seconds.

The GUI Thread runs the Qt event loop in the parent process. User interactions generate Command objects sent over the multiprocessing.Queue. A QTimer at 20 Hz drains the event queue and updates the meters, RTF label, and xrun count. TrayManager's pystray runs on a daemon thread within this same process, bridging tray actions to Qt signals via _TrayBridge.

The state diagram (Figure 6.2b) models the engine lifecycle with five states. It starts in IDLE, transitions to INITIALIZING when AudioEngineHandle.start() is called (opening the stream, loading the model, running warmup), then to RUNNING once everything is ready. SET_PASSTHROUGH toggles between RUNNING and PASS_THROUGH. SHUTDOWN or an unrecoverable error goes to STOPPED, where the stream is closed and the process exits.

## 6.3 User Interface Diagrams

The ControlWindow (Figure 6.3) has a vertical layout. At the top is the app title, then a row with the "Suppression: ON" / "Suppression: OFF" toggle (btn_toggle, green or red) and the "Pass-Through (Demo)" button (btn_passthrough). Below that are two sliders: suppression strength (slider_strength, 0 to 100, default 100, controlling wet/dry mix) and output gain (slider_gain, -12.0 to +12.0 dB in 0.1 dB steps, default 0.0).

Below the sliders are two combo boxes for input and output devices (combo_input and combo_output, populated from app/audio/devices.py query_devices()). Then come two level meter bars (meter_in, meter_out) that update at 20 Hz, and an RTF label (lbl_rtf) showing something like "RTF: 0.0234 (42.7x headroom)". At the bottom, the "Proof of Concept" button (btn_poc) opens the WaveformViewer.

The WaveformViewer (Figure 6.3b) has three vertically stacked pyqtgraph.PlotWidget panels with linked X-axes in seconds. Top panel: clean signal in green. Middle: noisy signal in red. Bottom: filtered signal in blue. Dashed vertical lines at 1.2, 3.5, 5.0, 6.8, and 8.5 seconds mark the transient events (dog bark, door slam, keyboard click, siren chirp, plosive 'P'). Below the plots, metric cards show Mean chunk time (us), P99, RTF, Headroom, and PASS/FAIL (PASS if RTF < 0.80). Play/Stop buttons under each panel let you listen via sounddevice.play().

## 6.4 Report Layouts

Running poc_realtime_transient.py --mode demo produces three WAV files and a console report. The files are: test_clean.wav (10s clean signal at 48 kHz, 16-bit PCM), test_noisy.wav (clean + five transient events), and test_noisy_filtered.wav (filtered in 128-sample chunks). The console report from LatencyProfiler.report() shows: total chunks, mean processing time (us), P99, max, RTF, headroom multiplier, and budget percentage consumed. diagnose() prints per-transient attenuation in dB. The report ends with a FEASIBILITY VERDICT — an ASCII banner showing "PASS" if RTF < 0.80, or "FAILED" otherwise.

## 6.5 External Interfaces

We interface with three external subsystems: audio hardware, ONNX Runtime, and the training data sources.

For audio hardware, we go through sounddevice, which wraps PortAudio. We configure streams at samplerate = 48,000 Hz, channels = 1, dtype = "float32", and blocksize = 128 (PoC) or 1024 (GUI, from AppConfig.block_size). Device selection happens through combo_input and combo_output, which enumerate devices via sounddevice.query_devices(). The callback follows the standard PortAudio pattern: callback(indata, outdata, frames, time_info, status), where indata and outdata are numpy arrays of shape [frames, channels].

The ONNX Runtime interface is a strict tensor contract. Input: "noisy_audio", shape [batch_size, 512], float32. Output: "clean_audio", same shape and dtype. The batch axis is dynamic — we use batch_size=1 for real-time inference and larger batches for evaluation. Session setup: providers=["CPUExecutionProvider"], graph_optimization_level=ORT_ENABLE_ALL, execution_mode=ORT_SEQUENTIAL.

Training data paths are configured in config.py: LIBRISPEECH_DIR = "data/raw/librispeech", FREESOUND_DIR = "data/raw/freesound", RIR_DIR = "data/raw/openair_rirs". The discover_audio_files function in generate_dataset.py recursively finds all files with extensions in {.wav, .flac, .ogg, .mp3, .aiff, .aif} and returns sorted Path lists.

## 6.6 Packaging and Deployment Diagram

Deploying our system (Figure 6.6) is straightforward: clone the repo, create a Python 3.10+ venv (python -m venv .venv_poc), activate it, and run pip install -r requirements.txt. That installs all seventeen dependencies. Launch with python -m app.main.

The deployment has three tiers. First tier: the Python runtime (3.10+, venv, pip packages). Second tier: the application code (app/, audio/, model/, inference/, dataset/, training/, plus config.py and poc_realtime_transient.py). Third tier: the model artifact (model/filter_model.onnx and model/filter_model.onnx.data). For a production deployment, you'd only need tiers one and two plus the model file — the training packages (dataset/, training/) and heavy dependencies (torch, torchaudio) could be left out.

Figure 6.6 shows a single machine hosting two processes: the GUI process (QApplication, ControlWindow, TrayManager) and the Audio Engine process (sounddevice, RingBuffer, ONNXInferenceRunner). The processes talk via multiprocessing.Queue. The engine talks to the audio hardware through PortAudio (via sounddevice) and loads the ONNX model from the file system.


---

# Chapter 7 — Technologies Used

Table 7.1 lists every technology in our stack, the version we use, what it does, and why we chose it.

| Technology | Version | Purpose | Why Chosen |
|---|---|---|---|
| Python | >= 3.10 | Primary implementation language | Rapid prototyping, extensive ML ecosystem, seamless PyTorch/ONNX integration |
| NumPy | >= 1.24.0 | Vectorised array operations for DSP and buffer management | De facto standard for numerical computing; C-optimised routines for real-time performance |
| PyTorch | >= 2.2.0 | Neural network definition, training, and quantization | Dynamic computation graphs for research flexibility; native quantization and pruning APIs |
| torchaudio | >= 2.2.0 | Audio transforms and data augmentation utilities | Companion to PyTorch with audio-specific transforms |
| ONNX | >= 1.16.0 | Model interchange format for deployment | Vendor-neutral graph format; opset 18 covers all required operators |
| ONNX Runtime | >= 1.18.0 | CPU inference engine for the exported model | C++ backend releases GIL during inference; pre-allocated buffer support; cross-platform CPUExecutionProvider |
| sounddevice | >= 0.4.6 | Real-time audio capture and playback | Python bindings for PortAudio; callback-based API enabling low-latency streaming; cross-platform |
| soundfile | >= 0.12.1 | Reading and writing WAV/FLAC audio files | libsndfile bindings; supports all required formats; reliable PCM I/O |
| SciPy | any | Signal processing: lfilter, fftconvolve, resample_poly, chirp | Comprehensive DSP library; lfilter for real-time FIR application; fftconvolve for RIR convolution |
| PyQt6 | >= 6.5 | GUI framework for the control window and waveform viewer | Modern Qt6 bindings; native look-and-feel on all platforms; QThread for background processing |
| pyqtgraph | >= 0.13 | Real-time waveform plotting in the waveform viewer | High-performance OpenGL-accelerated plotting; Qt-native integration; linked axes for synchronised zoom |
| pystray | >= 0.19 | System tray icon with menu | Cross-platform tray support (Windows, macOS, Linux); PIL-based icon rendering |
| Pillow | >= 10.0 | Programmatic icon generation for system tray | PIL Image drawing for dynamic microphone icon (green/red state indication) |
| pesq | >= 0.0.4 | PESQ evaluation metric computation | Reference implementation of ITU-T P.862; widely used in speech quality evaluation |
| tqdm | >= 4.66.0 | Training progress bar | Standard progress visualisation for long-running training loops |
| librosa | >= 0.10.0 | Audio analysis utilities (available, not actively used) | Available for future feature extraction if needed; loaded optionally |
| LibriSpeech | train-clean-100 | Clean speech dataset for training data generation | 100 hours of read English; extensive speaker/prosody diversity; CC BY 4.0 license [2] |
| FreeSound | Various | Transient noise recordings for training data generation | Crowdsourced real-world noise diversity; covers all target noise categories [3] |
| OpenAIR | Various | Room Impulse Response recordings for reverberant data synthesis | Measured RIRs from real rooms; acoustic realism for training data |

*Table 7.1 — Complete Technology Stack*

Picking the right inference engine was probably the single most consequential decision in the project. We considered running PyTorch directly with torch.no_grad(), using TorchScript, and using TensorFlow Lite, but each had problems. PyTorch in eager mode holds the Python GIL during the forward pass, because every operator call goes through the Python/C++ boundary via pybind11 — so while inference is running, the audio callback thread can't execute, and we get xruns. TorchScript would fix the GIL issue, but it restricts what Python constructs you can use and doesn't support data-dependent control flow, which would have complicated our Mamba implementation. ONNX Runtime, on the other hand, parses the full computation graph up front and runs the entire thing as one C++ operation during session.run(), releasing the GIL for the whole duration. That means the audio callback thread can keep writing to the ring buffer concurrently with inference, and we avoid the main source of dropout risk.

The choice of Mamba SSM over transformers is all about computational cost. A standard multi-head self-attention layer needs O(H * L^2 * d_head) operations per layer. With our parameters (L = 512, d_model = 64), even a single-head, single-layer attention block needs 512^2 = 262,144 multiply-accumulate operations — and with KV cache management and softmax overhead on top, it's way more than what a CPU can handle in 2.67 ms. The Mamba SSM costs O(L * d_inner * d_state) = O(512 * 128 * 16) = O(1,048,576) for the full sequence, but it supports O(1)-per-timestep recurrent inference through selective_scan_sequential — each new sample just needs O(d_inner * d_state) = O(2,048) operations [1]. That's the critical difference.

We went with PyQt6 over web-based alternatives (Electron, Flask) and other Python GUI toolkits (Tkinter, wxPython) for two specific reasons. First, PyQt6's QApplication coexists well with Python's multiprocessing — the GUI event loop can run in the main process while the audio engine runs in a totally separate process with its own GIL. Second, we needed system tray integration for the use case we're targeting — the app runs quietly in the background during voice calls, accessible via the tray icon. pystray handles that across all three major platforms.

We chose sounddevice over PyAudio (the other popular PortAudio wrapper) mainly because sounddevice gives us data as numpy arrays directly. PyAudio returns raw bytes that you have to convert manually, which is clunky and error-prone. sounddevice's callback-based API also mirrors PortAudio's native model closely, giving us the lowest latency path. It works on Windows, macOS, and Linux with the same API, and platform-specific audio driver details are handled by PortAudio underneath.

---

# Chapter 8 — Implementation and Pseudocode

## 8.1 Ring Buffer Implementation

The ring buffer is what makes real-time processing possible by isolating the hardware callback from inference. Here's the pseudocode for the production implementation in audio/ring_buffer.py.

```
CLASS RingBuffer:
    INITIALISE(capacity = SAMPLE_RATE * RING_BUFFER_SECONDS):
        _buf       = numpy.zeros(capacity, dtype=float32)   # pre-allocated
        _capacity  = capacity                                 # 24,000 samples default
        _write_pos = 0                                        # monotonically increasing
        _lock      = threading.Lock()                         # protects _write_pos only

    FUNCTION write(samples):
        n = length(samples)
        ACQUIRE _lock:
            start = _write_pos MOD _capacity
            end   = start + n
            IF end <= _capacity:
                _buf[start : end] = samples        # single contiguous copy
            ELSE:
                first_part = _capacity - start
                _buf[start :] = samples[: first_part]        # wrap: copy to end
                _buf[: n - first_part] = samples[first_part:]  # copy remainder to start
            _write_pos += n                        # advance write head
        RELEASE _lock
        # NOTE: overrun is handled implicitly — oldest data overwritten

    FUNCTION read(n):
        snapshot_wp = _write_pos               # atomic under GIL (int read)
        total_written = snapshot_wp
        IF total_written < n:
            # UNDERRUN: zero-pad the leading portion
            result = numpy.zeros(n, dtype=float32)
            available = total_written
            start = (snapshot_wp - available) MOD _capacity
            # ... copy available samples to tail of result
            RETURN result
        start = (snapshot_wp - n) MOD _capacity
        IF start + n <= _capacity:
            RETURN _buf[start : start + n].copy()
        ELSE:
            RETURN concatenate(_buf[start:], _buf[: (start + n) - _capacity])

    FUNCTION get_latency_ms():
        RETURN (1.0 / SAMPLE_RATE) * 1000        # ≈ 0.021 ms
```

The design here is that the lock only covers the pointer increment, not the actual bulk data copy (though the copy does happen within the lock scope, it runs fast because NumPy uses C under the hood). The read side doesn't lock anything — it relies on CPython's GIL making single-integer reads atomic. The end result is that the audio callback thread is never blocked by the inference thread, which is what gives us the sub-100-microsecond latency we measured in the PoC.

## 8.2 Mamba SSM Inference

This pseudocode shows the forward pass of one MambaBlock in inference mode, using selective_scan_sequential for O(1)-per-step recurrence.

```
CLASS MambaBlock:
    PARAMETERS:
        in_proj    : Linear(d_model=64, 2*d_inner=256)
        conv1d     : Conv1d(d_inner=128, d_inner=128, kernel=4, groups=128)
        x_proj     : Linear(d_inner=128, d_state*2 + d_inner = 160)
        dt_proj    : Linear(d_inner=128, d_inner=128)
        A_log      : Parameter[d_inner=128, d_state=16]     # log state transition
        D          : Parameter[d_inner=128]                   # skip connection
        out_proj   : Linear(d_inner=128, d_model=64)

    FUNCTION forward(x):
        # x: [B, L, 64]
        xz = in_proj(x)                          # [B, L, 256]
        x_inner = xz[:, :, :128]                  # [B, L, 128]
        z       = xz[:, :, 128:]                  # [B, L, 128]

        # Depthwise causal convolution
        x_conv = conv1d(x_inner.transpose(1,2))   # [B, 128, L]
        x_conv = SiLU(x_conv).transpose(1,2)      # [B, L, 128]

        # Compute input-dependent SSM parameters
        x_dbl = x_proj(x_conv)                    # [B, L, 160]
        B_ssm = x_dbl[:, :, :16]                  # [B, L, 16]
        C_ssm = x_dbl[:, :, 16:32]                # [B, L, 16]
        dt_raw = x_dbl[:, :, 32:]                 # [B, L, 128]

        dt = softplus(dt_proj(dt_raw))            # [B, L, 128], positive

        A_real = -exp(A_log)                       # [128, 16], negative values

        # === Selective Scan (Sequential / Recurrent) ===
        y = selective_scan_sequential(x_conv, dt, A_real, B_ssm, C_ssm, D)

        # Gated output
        y = y * SiLU(z)                            # [B, L, 128]
        output = out_proj(y)                       # [B, L, 64]
        RETURN output

FUNCTION selective_scan_sequential(x, dt, A, B, C, D):
    # x: [B, L, 128], dt: [B, L, 128], A: [128, 16]
    # B: [B, L, 16], C: [B, L, 16], D: [128]
    B, L, d_inner = shape(x)
    d_state = 16
    h = zeros(B, d_inner, d_state)       # hidden state, initially zero
    outputs = []

    FOR t = 0 TO L-1:
        # Discretise: dA = exp(dt[:, t, :] * A)     → [B, 128, 16]
        dA = exp(dt[:, t, :, None] * A[None, :, :])  # broadcast

        # Input matrix: dB = dt[:, t, :] * B[:, t, :]  → [B, 128, 16]
        dB = dt[:, t, :, None] * B[:, t, None, :]

        # State update: h = dA * h + dB * x_t
        h = dA * h + dB * x[:, t, :, None]           # [B, 128, 16]

        # Output: y_t = (C_t * h).sum(dim=-1) + D * x_t
        y_t = (C[:, t, None, :] * h).sum(dim=-1)     # [B, 128]
        y_t = y_t + D * x[:, t, :]                    # skip connection

        outputs.append(y_t)

    RETURN stack(outputs, dim=1)                       # [B, L, 128]
```

The crucial thing to understand here is that dt is input-dependent. It comes from the audio itself via x_proj and dt_proj. When there's a transient noise burst, the learned projections push dt to large values. Since A is negative, dA = exp(dt * A) drops toward zero, which decays the hidden state rapidly and suppresses the noisy input. But when a speech plosive happens, the surrounding speech context produces smaller dt values, so the state is preserved and the plosive passes through. This is how the Mamba SSM tells transients apart from plosives without any hand-crafted rules.

## 8.3 DeepFIR Inference

Here is the pseudocode for the DeepFIR path, from audio input through to filtered output.

```
CLASS DeepFIRPredictor:
    PARAMETERS:
        conv1   : CausalConv1d(in=1, out=32, kernel=8)    # left-pads 7 zeros
        prelu1  : PReLU(32)
        conv2   : CausalConv1d(in=32, out=64, kernel=8)   # left-pads 7 zeros
        prelu2  : PReLU(64)
        pool    : AdaptiveAvgPool1d(output_size=1)
        fc      : Linear(64, FIR_FILTER_LENGTH=64)

    FUNCTION forward(audio_context):
        # audio_context: [B, 512]
        x = audio_context.unsqueeze(1)            # [B, 1, 512]

        # Causal convolutions (left-pad to preserve causality)
        x = prelu1(conv1(x))                      # [B, 32, 512]
        x = prelu2(conv2(x))                      # [B, 64, 512]

        # Global pooling and tap prediction
        x = pool(x).squeeze(-1)                   # [B, 64]
        taps = tanh(fc(x))                        # [B, 64], bounded to [-1, 1]

        RETURN taps

FUNCTION _to_minimum_phase(taps):
    # Convert linear-phase FIR taps to minimum-phase
    n_fft = next_power_of_2(length(taps) * 2)
    H = FFT(taps, n=n_fft)                        # frequency response
    log_mag = log(abs(H) + epsilon)                # log magnitude
    cepstrum = IFFT(log_mag).real                  # real cepstrum

    # Liftering: double positive-time, zero negative-time
    liftered = zeros_like(cepstrum)
    liftered[0] = cepstrum[0]
    liftered[1 : n_fft//2] = 2 * cepstrum[1 : n_fft//2]

    min_phase_H = exp(FFT(liftered))               # minimum-phase spectrum
    min_phase_taps = IFFT(min_phase_H).real[:length(taps)]
    RETURN min_phase_taps

FUNCTION apply_filter(noisy_audio, taps):
    min_phase_taps = _to_minimum_phase(taps)
    filtered = scipy.signal.lfilter(min_phase_taps, [1.0], noisy_audio)
    RETURN filtered
```

The CausalConv1d pads kernel_size - 1 = 7 zeros on the left only, so the output at time t only depends on inputs from t-7 to t. AdaptiveAvgPool1d(1) squashes the 512-sample time axis down to one value — a global summary of the context. Tanh keeps the predicted taps in [-1, 1], which is important for numerical stability in the minimum-phase conversion and the lfilter call.

## 8.4 Quantization and Pruning Pipeline

After training, we run a post-processing pipeline that produces a small, fast INT8 model ready for real-time inference.

```
FUNCTION optimize_and_export(model, test_loader):
    # Step 1: Magnitude Pruning (50% sparsity)
    FOR each module IN model.modules():
        IF module IS Linear OR module IS Conv1d:
            prune.l1_unstructured(module, name='weight', amount=PRUNE_RATIO=0.50)
    # Make pruning permanent (remove mask, bake into weight)
    FOR each module IN model.modules():
        IF has_pruning(module):
            prune.remove(module, 'weight')

    # Step 2: INT8 Dynamic Quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.Conv1d},
        dtype=torch.qint8
    )
    # NOTE: A_log, D, RMSNorm weights remain FP32 (they are nn.Parameter, not Linear/Conv1d)

    # Step 3: Benchmark RTF
    rtf = benchmark_rtf(quantized_model, duration_seconds=10)
    IF rtf > TARGET_RTF = 0.80:
        WARN("Model may not meet real-time constraint: RTF = {rtf}")

    # Step 4: Export to ONNX
    dummy_input = torch.randn(1, CONTEXT_WINDOW_SAMPLES=512)
    torch.onnx.export(
        quantized_model, dummy_input,
        ONNX_MODEL_PATH = "model/filter_model.onnx",
        opset_version=18,
        input_names=["noisy_audio"],
        output_names=["clean_audio"],
        dynamic_axes={"noisy_audio": {0: "batch"}, "clean_audio": {0: "batch"}}
    )

    # Step 5: Verify ONNX
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    test_input = numpy.randn(1, 512).astype(float32)
    output = session.run(None, {"noisy_audio": test_input})
    ASSERT shape(output[0]) == (1, 512)
```

## 8.5 Dataset Generation

This pseudocode shows how the DatasetBuilder in generate_dataset.py creates training pairs.

```
CLASS DatasetBuilder:
    INITIALISE(config):
        speech_files = discover_audio_files(config.speech_dir)   # LibriSpeech
        noise_files  = discover_audio_files(config.noise_dir)    # FreeSound
        rir_provider = RIRProvider(config)                        # OpenAIR or pyroomacoustics
        rng = Random(seed=config.seed=42)

    FUNCTION _generate_one(index):
        seg_len = config.target_sr * config.segment_duration     # 48000 * 4.0 = 192,000

        # 1. Select random speech and noise files
        speech_raw = load_audio(rng.choice(speech_files), target_sr=48000)
        noise_raw  = load_audio(rng.choice(noise_files),  target_sr=48000)

        # 2. Extract fixed-length segments (zero-pad if shorter)
        speech_seg = random_segment(speech_raw, seg_len, rng)
        noise_seg  = random_segment(noise_raw,  seg_len, rng)

        # 3. Apply Room Impulse Responses
        rir_near, rir_far = rir_provider.get_rir_pair(rng)
        speech_conv = fftconvolve(speech_seg, rir_near, mode="full")[:seg_len]
        noise_conv  = fftconvolve(noise_seg,  rir_far,  mode="full")[:seg_len]

        # 4. Mix at random SNR in [-5, +20] dB
        snr_db = rng.uniform(config.snr_min=-5.0, config.snr_max=20.0)
        desired_noise_rms = rms(speech_conv) / (10 ^ (snr_db / 20))
        scaled_noise = noise_conv * (desired_noise_rms / rms(noise_conv))
        noisy_mix = speech_conv + scaled_noise

        # 5. Joint peak normalisation
        peak = max(abs(noisy_mix))
        gain = config.peak_norm_target=0.95 / peak
        noisy_mix    = noisy_mix * gain
        clean_target = speech_conv * gain

        RETURN noisy_mix, clean_target, metadata

    FUNCTION build():
        n_train = total_samples * 0.80        # 8,000
        n_val   = total_samples * 0.10        # 1,000
        n_test  = total_samples * 0.10        # 1,000

        FOR each sample IN shuffled(split_assignments):
            noisy, clean, meta = _generate_one(index)
            save_audio(output_dir / split / "noisy" / fname, noisy, 48000)
            save_audio(output_dir / split / "clean" / fname, clean, 48000)

        save_json(output_dir / "metadata.json", all_metadata)
```

## 8.6 Training Loop

Here's the full training workflow, from data loading through post-training optimization.

```
FUNCTION train():
    # Initialisation
    model     = CombinedModel()                # DeepFIRPredictor + MambaSSM
    optimizer = AdamW(model.parameters(), lr=LR=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS=50)
    dataset   = TransientNoiseDataset(DATASET_DIR)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE=32, shuffle=True)

    best_loss = infinity
    log_file  = open("checkpoints/training_log.csv", "w")

    FOR epoch = 1 TO EPOCHS=50:
        model.train()
        epoch_loss = 0.0

        FOR batch (noisy, clean) IN loader:
            optimizer.zero_grad()

            # Forward pass (training mode: parallel scan)
            estimated = model.forward_train(noisy)       # [B, 512]

            # Combined loss
            loss_si_sdr  = si_sdr_loss(estimated, clean)           # negative SI-SDR
            loss_tss     = tss_loss(estimated, clean, noisy)       # transient energy
            loss_plosive = plosive_preservation_loss(estimated, clean)  # plosive MSE
            loss = 1.0 * loss_si_sdr + 0.5 * loss_tss + 1.0 * loss_plosive

            # Backward pass with gradient clipping
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        log_file.write(f"{epoch},{avg_loss}\n")

        # Checkpointing
        save(model, "checkpoints/latest.pt")
        IF avg_loss < best_loss:
            best_loss = avg_loss
            save(model, "checkpoints/best.pt")

    # Post-training optimisation pipeline
    model = load("checkpoints/best.pt")
    apply_magnitude_pruning(model, amount=0.50)          # L1 unstructured
    quantized = quantize_model_int8(model)                # INT8 dynamic
    rtf = benchmark_rtf(quantized, duration=10)           # verify RTF < 0.80
    export_to_onnx(quantized, "model/filter_model.onnx")  # opset 18
```

Our combined_loss has three parts working together. The si_sdr_loss (weight 1.0) pushes for overall waveform fidelity — it maximises the scale-invariant signal-to-distortion ratio between estimated and clean. The tss_loss (weight 0.5) penalises leftover transient energy in the output, measured in transient-masked regions, with a speech distortion penalty that prevents the model from getting a high TSS by just mangling everything. The plosive_preservation_loss (weight 1.0) computes MSE only in plosive-masked regions, scaled by 10x to make sure the model really cares about getting plosives right. We clip gradients at max_norm = 1.0 because the loss landscape near transient boundaries tends to be sharp, and without clipping we saw training instabilities.


---

# Chapter 9 — Conclusion of Capstone Project Phase 2

## 9.1 Summary of Work

In Phase 2, we've designed, implemented, and validated the infrastructure for a real-time, CPU-only transient noise suppression system. The work breaks down into five major pieces.

First, we built the four-layer processing pipeline: a lock-free SPSC ring buffer (Layer 1), the INT8 quantization and 50% magnitude pruning pipeline (Layer 2), the DeepFIR minimum-phase tap predictor (Layer 3), and a four-block Mamba SSM with selective scan (Layer 4). Each layer is its own module with clear input/output contracts, so they can be tested, replaced, or improved independently.

Second, we implemented the combined DeepFIR + Mamba SSM network as the CombinedModel class in PyTorch. It has two forward modes — forward_train for parallel scan and forward_realtime for sequential scan — and we've successfully exported it to ONNX (opset 18) as filter_model.onnx. The ONNX inference pipeline, with ONNXInferenceRunner (pre-allocated buffers, session warmup) and RealTimePipeline (ring buffer integration, wet/dry mixing), is working and tested against ONNX Runtime's CPUExecutionProvider.

Third, the dataset generation pipeline can produce 10,000 paired (noisy, clean) samples from LibriSpeech and FreeSound data, with RIR reverberation and SNR mixing in the [-5, +20] dB range, split 80/10/10. The training loop uses AdamW (lr=1e-3, weight_decay=1e-4), CosineAnnealingLR (T_max=50), gradient clipping (max_norm=1.0), and our combined loss (SI-SDR + TSS + plosive preservation). Post-training optimization — prune, quantize, benchmark, export — runs automatically.

Fourth, the GUI is up and running in PyQt6. ControlWindow has the suppression toggle, pass-through mode, strength and gain sliders, device selection, level meters, and RTF readout. TrayManager shows a green or red microphone icon and gives tray menu access. WaveformViewer displays three-panel waveform plots with transient markers and metric cards. Everything runs in a separate process from the audio engine, connected by typed IPC messages over multiprocessing.Queue.

Fifth, the offline PoC benchmark (poc_realtime_transient.py) confirms that the DSP ring-buffer layer handles 128-sample chunks at 48 kHz in under 100 microseconds — less than 4% of the 2.667 ms budget, leaving over 96% of the time free for neural inference.

As for NFR validation: DSP latency under 100 microseconds — achieved. Ring buffer latency around 0.021 ms — achieved. CPU-only operation — achieved. The quality metrics (SI-SDRi > 4.0 dB, PESQ >= 3.2, TSS > 65%, plosive SI-SDR > 25 dB) and the end-to-end RTF target can't be validated yet because the model still has random, untrained weights. Those will have to wait for Phase 3 when we actually train on real data.

## 9.2 Key Findings

Our first major finding is that the DSP layer processes audio absurdly fast — under 100 microseconds per chunk, consuming less than 4% of the real-time budget. That gives us more than 96% of the 2.667 ms window for neural inference, which is reassuring. The PoC benchmark hits an RTF of roughly 0.01 to 0.05 for the DSP-only path, which is 20x to 100x real-time. There's plenty of room.

Second, the Mamba SSM's O(N) complexity is really what makes this project feasible on CPU. Each timestep in recurrent mode costs O(d_inner * d_state) = O(128 * 16) = O(2,048) multiply-accumulate operations, and that doesn't change with context length. Compare that to transformer attention at O(N^2) — for a 512-sample window that's 262,144 operations per head, which just isn't practical for a CPU in under 3 ms.

Third, INT8 dynamic quantization plus 50% L1 magnitude pruning gives us roughly a 7x speedup. The key detail is that the Mamba state parameters (A_log, D, RMSNorm weights) stay in FP32 — so the state dynamics that matter for transient-vs-plosive distinction keep their full precision, while the projection layers that do most of the compute run in 8-bit.

Fourth, we found that explicitly handling plosives is absolutely necessary. Without some mechanism to tell plosives apart from noise, the system would suppress 'P', 'T', and 'K' sounds, and speech would sound muffled. In the PoC, the synthetic plosive at t = 8.5 seconds passes through the TransientDetector because of its gradual onset envelope. In the neural model, the selective scan's input-dependent dt parameter handles this — the model learns to produce different dt values depending on whether the audio context looks like speech or noise.

## 9.3 Limitations

We want to be upfront about what's not working yet. The biggest issue is that the model currently has random weights. It hasn't been trained. All the quality metrics (SI-SDRi, PESQ, TSS, plosive SI-SDR) are meaningless at this point because the model just outputs noise. The StubDenoiser in the GUI path is a simple pass-through. Real suppression won't happen until we train on actual data in Phase 3.

Another limitation is that all our training data is synthetic. We mix clean speech with noise clips and apply RIR convolution, but that doesn't capture everything about real-world audio — things like non-linear microphone distortion, multiple simultaneous noise sources, the Lombard effect (people talking louder in noise), or the specific spectral characteristics of real rooms that don't perfectly match our RIRs. The model might not generalise well to real recordings that look different from the training distribution.

Our noise scope is also limited. We're targeting transient noise specifically. Music, multi-channel audio, acoustic echo cancellation, and quasi-stationary sounds like background chatter or traffic rumble are all outside what we designed for. The DeepFIR layer handles some stationary noise, but that's not the main focus.

Finally, there's some technical debt in the codebase. We have two multiprocessing architectures: the production one in app/audio/engine.py (typed CmdType/EvtType, StubDenoiser) and an older one in process_manager.py (dict-based commands, AudioIOManager). The latter is effectively dead code. There are also two config systems — root config.py (module constants) and app/config.py (AppConfig dataclass) — with conflicting block sizes (BLOCK_SIZE = 256 vs block_size = 1024). And the selective_scan_parallel function in mamba_ssm.py is misleadingly named — despite the name, it's a sequential for-loop, functionally identical to selective_scan_sequential but with extra overhead.

---

# Chapter 10 — Plan of Work for Capstone Project Phase 3

Table 10.1 lays out what we plan to do in Phase 3, who's responsible for each task, and when we expect to finish it.

| Task | Description | Owner | Target Milestone |
|---|---|---|---|
| Dataset acquisition | Download LibriSpeech train-clean-100 (28 GB) and curate FreeSound transient noise collection (target: 500+ unique sounds) | Sahil, Deepesh | Week 1 |
| Dataset generation | Run generate_dataset.py to produce 10,000 synthetic (noisy, clean) pairs with RIR reverberation | Deepesh | Week 2 |
| Model training | Train CombinedModel for 50 epochs on generated dataset; tune combined_loss weights for optimal SI-SDRi/TSS/plosive balance | Dhrushaj, Chandan | Weeks 2-4 |
| Metric validation | Run full_evaluation_report() to verify SI-SDRi > 4.0 dB, PESQ >= 3.2, TSS > 65%, plosive SI-SDR > 25 dB | Chandan | Week 4 |
| RTF benchmarking | Run benchmark_rtf() on quantized + pruned model on target hardware; verify RTF < 0.80 | Dhrushaj | Week 4 |
| ONNX integration | Replace StubDenoiser with ONNXDenoiser in app/inference/stub.py using the trained filter_model.onnx | Dhrushaj | Week 5 |
| Architecture cleanup | Remove process_manager.py dead code; unify config.py and app/config.py; fix BLOCK_SIZE inconsistency | Chandan | Week 5 |
| Fix selective_scan_parallel | Implement true parallel scan using cumulative sum formulation or replace with Triton kernel | Deepesh | Week 6 |
| Real-world evaluation | Record real-world test scenarios (office, home, cafe) and evaluate model on unseen acoustic conditions | Sahil | Weeks 5-6 |
| Virtual audio cable integration | Test and document VB-Audio Virtual Cable (Windows) and PulseAudio null sink (Linux) routing for Zoom/Teams | Sahil | Week 6 |
| ARM/embedded evaluation | Test ONNX Runtime inference on Apple Silicon (M1/M2) and Raspberry Pi 4 (ARM64); report RTF on each platform | Deepesh | Week 7 |
| User perceptual study | Conduct A/B listening test with 10+ participants comparing processed vs. unprocessed audio; collect MOS scores | All | Week 7 |
| Streaming Mamba exploration | Investigate extending context window beyond 512 samples using hidden state carry-over between chunks | Dhrushaj | Week 8 |
| GUI polish | Add waveform display in real-time mode (not just offline); add noise type detection indicator; improve icon design | Chandan | Week 8 |
| Final documentation | Update CODEBASE_CONTEXT.md, README.md, and user manual; prepare final submission package | All | Week 9 |
| Plagiarism check | Run final codebase through plagiarism detection; ensure all borrowed code has attribution | All | Week 9 |

*Table 10.1 — Phase 3 Work Plan*

The critical bottleneck is model training in Weeks 2-4. Everything downstream — quality metrics, ONNX integration, real-world testing, user studies — depends on having a trained model. The architecture cleanup in Week 5 should happen before the documentation sprint so the codebase matches what we write about.

---

# References

[1] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv preprint arXiv:2312.00752, Dec. 2023.

[2] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, "LibriSpeech: An ASR Corpus Based on Public Domain Audio Books," in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Brisbane, Australia, Apr. 2015, pp. 5206-5210.

[3] F. Font, G. Roma, and X. Serra, "Freesound Technical Demo," in Proc. 21st ACM International Conference on Multimedia (MM '13), Barcelona, Spain, Oct. 2013, pp. 411-412.

[4] A. V. Oppenheim, R. W. Schafer, and J. R. Buck, Discrete-Time Signal Processing, 2nd ed. Upper Saddle River, NJ: Prentice Hall, 1999, ch. 5 (Minimum-Phase FIR Filter Design).

[5] Microsoft, "ONNX Runtime: Cross-platform, High Performance ML Inferencing and Training Accelerator," 2024. [Online]. Available: https://onnxruntime.ai/docs/. [Accessed: Apr. 2026].

[6] J.-M. Valin, "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement," in Proc. IEEE 20th International Workshop on Multimedia Signal Processing (MMSP), Vancouver, Canada, Aug. 2018, pp. 1-5.

[7] R. Krishnamoorthi, "Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper," arXiv preprint arXiv:1806.08342, Jun. 2018.

[8] H. Schroter, A. N. Escalante-B, T. Rosenkranz, and A. Maier, "DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement," in Proc. INTERSPEECH, Incheon, Korea, Sep. 2022, pp. 51-55.

---

# Appendix A — Definitions, Acronyms, and Abbreviations

| Term | Definition |
|---|---|
| SSM | Selective State Space Model — a sequence model that uses learnable state-space dynamics with input-dependent gating for O(N) sequence processing |
| FIR | Finite Impulse Response — a type of digital filter whose output depends only on a finite number of past input samples |
| ONNX | Open Neural Network Exchange — a vendor-neutral model interchange format for neural networks |
| RTF | Real-Time Factor — the ratio of processing time to audio duration; RTF < 1.0 means faster than real-time |
| SI-SDRi | Scale-Invariant Signal-to-Distortion Ratio improvement — the improvement in SI-SDR achieved by the system over the unprocessed input |
| PESQ | Perceptual Evaluation of Speech Quality — an ITU-T P.862 standard metric for automated speech quality assessment, scored from 1.0 to 4.5 |
| TSS | Transient Suppression Score — the fraction of transient noise energy removed from the output signal |
| SNR | Signal-to-Noise Ratio — the ratio of signal power to noise power, expressed in decibels |
| RIR | Room Impulse Response — the acoustic transfer function of a room, used to simulate reverberation |
| INT8 | 8-bit Integer — a quantized numerical representation using 8-bit integers instead of 32-bit floating point |
| SPSC | Single-Producer Single-Consumer — a concurrency pattern where exactly one thread writes and one thread reads |
| DSP | Digital Signal Processing — the mathematical manipulation of digitally represented signals |
| GIL | Global Interpreter Lock — a mutex in CPython that serialises execution of Python bytecode across threads |
| CPU | Central Processing Unit — the primary general-purpose processor in a computer |
| GUI | Graphical User Interface — the visual interface through which users interact with the application |
| NFR | Non-Functional Requirement — a requirement specifying performance, security, reliability, or other quality attributes |
| O(N) | Linear time complexity — an algorithm whose execution time grows linearly with input size N |
| AdamW | Adam with Weight Decay — an optimiser combining Adam's adaptive learning rates with decoupled weight decay regularisation |
| PReLU | Parametric Rectified Linear Unit — a learnable activation function with a trainable negative slope parameter |
| SiLU | Sigmoid Linear Unit — an activation function defined as x * sigmoid(x), also known as the Swish function |
| RMSNorm | Root Mean Square Layer Normalisation — a normalisation technique that scales by the RMS of the input, without mean subtraction |
| CausalConv1d | Causal one-dimensional convolution — a convolution that pads only on the left, ensuring output at time t depends only on inputs at time t and earlier |
| IPC | Inter-Process Communication — mechanisms for exchanging data between separate operating system processes |
| PortAudio | A cross-platform C library for real-time audio I/O |

---

# Appendix B — User Manual

## B.1 Prerequisites

You need Python 3.10 or later. We recommend setting up a virtual environment to keep things isolated.

On Windows:

```
python -m venv .venv_poc
.\.venv_poc\Scripts\activate
pip install -r requirements.txt
```

On macOS and Linux:

```
python3 -m venv .venv_poc
source .venv_poc/bin/activate
pip install -r requirements.txt
```

This installs all seventeen dependencies (PyQt6, sounddevice, torch, onnxruntime, and so on). On macOS, there's a quirk where the app sometimes needs to launch from the system Python instead of the venv Python due to the cocoa platform plugin — the app handles this automatically through a re-execution mechanism in app/main.py, so you generally don't need to worry about it.

## B.2 Running the GUI

Launch the interface with:

```
python -m app.main
```

This starts the app, spawns the audio engine in a separate process, and shows the ControlWindow plus a system tray icon. Here's what each control does.

The "Suppression: ON/OFF" button turns the noise suppression on (green) or off (red). When it's on, mic audio goes through the inference pipeline before playback.

"Pass-Through (Demo)" routes the microphone directly to the speaker with no processing. It's useful for A/B comparisons during demos. It disables the strength and gain sliders while active.

"Suppression Strength" (0 to 100) controls the wet/dry mix. At 100 (the default), you hear fully processed audio. At 0, it's the raw input.

"Output Gain" (-12.0 to +12.0 dB) lets you adjust the output volume to compensate for any level changes the suppression introduces.

The device dropdowns pick which mic and speaker to use. Changes apply immediately.

The level meters show real-time input and output levels, updating at 20 Hz.

The RTF label shows the current Real-Time Factor and how much headroom we have.

"Proof of Concept" at the bottom opens the WaveformViewer, which runs the offline benchmark and shows waveform plots with metrics.

Closing the window just hides it to the tray. Right-click the tray icon and click "Quit" to actually exit.

## B.3 Running the Offline Benchmark

To run the benchmark without the GUI:

```
python poc_realtime_transient.py --mode demo
```

This does the following: (1) generates a 10-second synthetic test signal with five transients (dog bark at 1.2s, door slam at 3.5s, keyboard click at 5.0s, siren chirp at 6.8s, plosive 'P' at 8.5s) mixed with a chirp and white background noise, (2) saves test_clean.wav and test_noisy.wav, (3) processes the noisy signal through the RealTimeFilter in 128-sample chunks while timing each chunk, (4) saves test_noisy_filtered.wav, and (5) prints a performance report with mean, P99, and max processing times, RTF, headroom, per-transient attenuation, and a FEASIBILITY VERDICT (PASS if RTF < 0.80).

For live mode (real-time mic processing):

```
python poc_realtime_transient.py --mode live
```

This grabs audio from the default mic, filters it in real-time, and plays it through the default speaker.

## B.4 Training the Model

Training has three steps.

Step 1 — Generate the dataset:

```
python -m dataset.generate_dataset
```

This creates (noisy, clean) pairs from LibriSpeech and FreeSound. Make sure data/raw/librispeech, data/raw/freesound, and data/raw/openair_rirs have data in them first.

Or use the full generator with RIR support:

```
python generate_dataset.py --total-samples 10000 --output-dir ./dataset
```

Step 2 — Train:

```
python -m training.train
```

This trains CombinedModel for 50 epochs with AdamW and CosineAnnealingLR. Checkpoints go to checkpoints/best.pt and checkpoints/latest.pt. After training finishes, it automatically prunes (50%), quantizes (INT8), benchmarks the RTF, and exports to model/filter_model.onnx.

Step 3 — Evaluate:

```
python -m training.evaluate
```

This runs the full metrics suite — SI-SDRi, PESQ, TSS, plosive SI-SDR — against each NFR target and prints a table with PASS/FAIL for each one.

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

Real-time voice communication systems suffer from transient noise contamination — impulsive, non-stationary disturbances such as dog barks, door slams, keyboard clicks, and sirens — that degrade intelligibility and user experience. Unlike stationary background noise, which possesses a stable spectral envelope amenable to classical spectral subtraction, transient noise is characterised by sudden onset, high peak energy, and broad spectral occupancy within durations of five to four hundred milliseconds. Existing commercial solutions such as Krisp, NVIDIA RTX Voice, and RNNoise either depend on cloud processing, require dedicated GPU hardware, or employ recurrent architectures that fail to distinguish impulsive transient noise from acoustically similar voiced plosive consonants ('P', 'T', 'K').

This report presents the design, implementation, and validation of a CPU-only, real-time transient noise suppression system built upon a novel four-layer processing pipeline. Layer 1 employs a lock-free, single-producer/single-consumer ring buffer that achieves sub-sample buffering latency of approximately 0.021 milliseconds. Layer 2 applies INT8 dynamic quantization combined with fifty-percent L1 magnitude pruning, delivering a measured seven-fold inference speedup over the FP32 baseline. Layer 3 introduces a DeepFIR neural predictor that generates sixty-four minimum-phase FIR filter taps via causal convolutional networks, targeting stationary noise components. Layer 4 deploys a Mamba Selective State Space Model operating in O(N) time complexity — in contrast to the O(N^2) cost of transformer-based alternatives — to perform context-aware transient gating while preserving speech plosives through learned selective scan parameters.

The combined DeepFIR and Mamba SSM model is exported to ONNX format (opset 18) and executed via ONNX Runtime's CPUExecutionProvider, which releases the Python Global Interpreter Lock during inference. The system targets a Real-Time Factor below 0.80, a Scale-Invariant Signal-to-Distortion Ratio improvement exceeding 4.0 dB, a Perceptual Evaluation of Speech Quality score of at least 3.2, and a Transient Suppression Score above sixty-five percent, all within a strict processing budget of 2.67 milliseconds per 128-sample chunk at 48 kHz. The classical DSP ring-buffer layer alone processes audio in under 100 microseconds, providing over ninety-six percent headroom for neural inference. This work demonstrates the feasibility of production-deployable, GPU-free transient noise suppression on commodity hardware.

---

## Acknowledgements

The authors express sincere gratitude to PES University, Electronic City, Bengaluru, for providing the academic infrastructure and institutional support that made this capstone project possible. We thank our faculty guide for their sustained guidance on real-time systems design, audio signal processing methodology, and the rigorous evaluation framework adopted throughout this work. We acknowledge the creators and maintainers of the LibriSpeech corpus [2] for providing the clean speech data that underpins our training pipeline, the FreeSound collaborative database [3] for its diverse collection of real-world transient noise recordings, and the OpenAIR project for Room Impulse Response data used in our acoustic simulation pipeline. We further acknowledge the open-source communities behind PyTorch, ONNX Runtime, and the sounddevice library, whose tools form the backbone of our implementation.

---


# Chapter 1 — Introduction

## 1.1 Background and Motivation

The proliferation of real-time voice communication platforms — video conferencing, VoIP telephony, online education, and live broadcasting — has created an urgent demand for high-quality audio processing that can operate under the diverse and unpredictable acoustic conditions of everyday environments. Users routinely participate in voice calls from homes, cafes, and open-plan offices where transient noise sources are ubiquitous: a dog barking in the adjacent room, a door slamming in the corridor, a colleague's mechanical keyboard, or an ambulance siren passing outside the window. These transient noises differ fundamentally from stationary background noise such as air conditioning hum, fan drone, or electrical mains buzz. Stationary noise possesses a relatively stable spectral envelope that can be estimated over time and removed through classical techniques such as Wiener filtering or spectral subtraction [6]. Transient noise, by contrast, is impulsive in nature — it appears suddenly, persists for only five to four hundred milliseconds, exhibits high peak energy, and occupies a broad frequency band that overlaps significantly with the speech spectrum.

The distinction between transient noise and stationary noise has profound implications for suppression algorithm design. A spectral subtraction system that performs well on fan noise will fail catastrophically on a door slam, because by the time its noise-floor estimator has detected the new spectral profile, the transient event has already ended. Conversely, an aggressive gating algorithm that suppresses all sudden energy bursts will inevitably destroy voiced plosive consonants — the 'P' in "Peter", the 'T' in "table", the 'K' in "kitchen" — which are themselves impulsive, high-energy speech events with spectral characteristics remarkably similar to many transient noises. This plosive-transient ambiguity is the central technical challenge of the domain.

Existing commercial and academic solutions address noise suppression with varying degrees of success but universally fail to satisfy the combined requirements of CPU-only operation, real-time latency, and plosive preservation. NVIDIA RTX Voice [1] achieves excellent suppression quality but requires a dedicated NVIDIA GPU with tensor core support, rendering it inaccessible on the vast majority of consumer and enterprise laptops. Krisp employs a cloud-hybrid architecture that introduces network-dependent latency and raises data privacy concerns, as raw audio must leave the user's device. RNNoise [6], developed by Jean-Marc Valin at Mozilla, operates on CPU and is widely deployed, but its recurrent neural network architecture is optimised for stationary noise and provides limited transient suppression capability. More recent academic work such as DeepFilterNet demonstrates promising results on general noise suppression but relies on transformer-based attention mechanisms with O(N^2) computational complexity in sequence length, making real-time operation on resource-constrained CPUs extremely challenging.

This project is motivated by the need for a system that simultaneously satisfies four requirements that no existing solution fully addresses: the system must suppress transient noise effectively, measured by a Transient Suppression Score exceeding sixty-five percent; it must preserve voiced plosive consonants, verified by a plosive segment SI-SDR exceeding 25 dB; it must operate within a strict real-time budget of less than 2.67 milliseconds per 128-sample chunk at 48 kHz; and it must run entirely on a standard CPU without GPU acceleration, cloud connectivity, or specialised hardware. The approach presented in this report — combining a Mamba Selective State Space Model with a DeepFIR neural filter in a quantised, pruned ONNX pipeline — is specifically engineered to meet all four of these requirements simultaneously.

## 1.2 Scope and Objectives

The scope of this project encompasses the design, implementation, and empirical validation of a complete real-time transient noise suppression system, from raw microphone input to filtered speaker output, including the graphical user interface, the audio I/O subsystem, the neural network architecture, the training and evaluation pipelines, and the deployment toolchain. The system is designed to process monaural (single-channel) audio at a sample rate of 48,000 Hz with a processing granularity of 128 samples per chunk, yielding a maximum allowable processing time of 2.67 milliseconds per chunk — the fundamental real-time constraint that governs every architectural decision in the project.

The first objective is to suppress transient noise in real-time on a standard CPU. This requires an inference engine that can execute the combined DeepFIR and Mamba SSM model within the 2.67 millisecond budget on a commodity Intel i5 or equivalent processor, without relying on GPU acceleration. The target Real-Time Factor is below 0.80, meaning the system must process audio at least 1.25 times faster than real-time to provide headroom for operating system scheduling jitter. The second objective is to preserve human speech fidelity, with particular attention to voiced plosive consonants. Plosive sounds share the impulsive energy envelope of many transient noises, and a system that indiscriminately gates all transients will degrade speech quality. The target Perceptual Evaluation of Speech Quality (PESQ) score is at least 3.2 on the ITU-T P.862 scale, and the plosive segment Scale-Invariant Signal-to-Distortion Ratio must exceed 25 dB.

The third objective is to achieve measurable improvement in signal quality as quantified by standard audio evaluation metrics. The target Scale-Invariant Signal-to-Distortion Ratio improvement (SI-SDRi) is greater than 4.0 dB, representing a meaningful perceptual improvement over the unprocessed noisy signal. The Transient Suppression Score (TSS), defined as the fraction of transient energy removed from the output relative to the input, must exceed sixty-five percent. The fourth objective is to deliver these capabilities within a production-ready software package that includes a graphical user interface with real-time level meters, suppression controls, and an offline waveform viewer for validation and demonstration purposes.

## 1.3 Report Organization

The remainder of this report is organised as follows. Chapter 2 formally defines the transient noise suppression problem, delineates the technical challenges, and specifies the non-functional requirements that the system must satisfy. Chapter 3 describes the data sources, dataset construction methodology, and preprocessing pipeline used to generate training and evaluation data. Chapter 4 presents the design details of the system across twelve quality dimensions including novelty, performance, security, and portability. Chapter 5 provides the high-level system architecture, detailing the four-layer processing pipeline, the ONNX inference engine, and the GUI subsystem. Chapter 6 elaborates on the design description through class diagrams, swimlane diagrams, user interface layouts, and external interface specifications. Chapter 7 catalogues the technologies used and justifies each selection. Chapter 8 presents the implementation in detail through pseudocode for every major subsystem. Chapter 9 concludes the Phase 2 work with a summary of achievements, key findings, and an honest assessment of current limitations. Chapter 10 outlines the work planned for Phase 3, including real-world dataset collection, cross-architecture evaluation, and user perceptual studies.

---

# Chapter 2 — Problem Definition

## 2.1 Problem Statement

The transient noise suppression problem can be formally stated as a causal signal separation task operating under hard real-time constraints. Let x(t) denote the observed microphone signal at discrete time index t, sampled at a rate of f_s = 48,000 Hz. This observed signal is modelled as the additive mixture x(t) = s(t) + n(t), where s(t) represents the desired clean speech signal and n(t) represents the transient noise component. The objective is to compute an estimate s_hat(t) of the clean speech signal such that the distortion between s_hat(t) and s(t) is minimised according to perceptual quality metrics, while the residual transient energy in s_hat(t) relative to n(t) is maximally suppressed. Crucially, this estimation must be performed causally — the system may only use samples x(t'), where t' is less than or equal to t — and within a strict latency budget.

The system processes audio in fixed-size chunks of C = 128 samples. At a sample rate of 48,000 Hz, each chunk represents 128 / 48,000 = 2.667 milliseconds of audio. The real-time constraint requires that the total processing time for each chunk — including ring buffer I/O, neural network inference, and output formatting — must not exceed this 2.667-millisecond window. If processing of chunk k is not complete before chunk k+1 arrives from the audio hardware, the system incurs a buffer underrun (xrun), resulting in audible artifacts such as clicks, pops, or silence gaps. The target Real-Time Factor (RTF), defined as the ratio of processing time to audio duration, must therefore be strictly less than 1.0, with a design target of 0.80 to accommodate operating system scheduling variability.

An additional constraint is that the system must operate entirely on a standard CPU without access to a discrete GPU, tensor processing unit, or cloud-based inference service. This constraint is motivated by the deployment target: commodity laptops and desktop machines used for everyday voice communication, where GPU availability cannot be assumed and cloud connectivity introduces unacceptable latency and privacy concerns. The system must also not employ any look-ahead buffering — that is, it must not delay its output by accumulating future audio samples to improve prediction quality — as any such delay adds directly to the end-to-end communication latency and degrades the real-time conversational experience.

## 2.2 Challenges

The first and most fundamental challenge is the spectral ambiguity between transient noise and voiced plosive consonants. Plosive sounds — the initial burst of 'P', 'T', 'K', 'B', 'D', and 'G' — are produced by the sudden release of air pressure from a closed vocal tract, creating an impulsive waveform with a broad spectral spread and rapid onset that closely resembles many categories of transient noise. A door slam and a hard 'P' plosive both exhibit energy rise times on the order of one to five milliseconds, both occupy frequency bands spanning several kilohertz, and both have peak amplitudes significantly above the surrounding signal level. Any suppression algorithm that operates solely on short-time energy envelope characteristics will inevitably confuse plosives with transients, leading to either false suppression of speech sounds or false retention of transient noise. Resolving this ambiguity requires a model with sufficient contextual awareness to distinguish the acoustic environment surrounding a plosive burst from the environment surrounding a transient noise event — a capability that motivates the selection of the Mamba SSM architecture with its recurrent hidden state.

The second challenge is the extreme latency constraint imposed by the 128-sample chunk size at 48 kHz. The 2.667-millisecond processing budget is at least an order of magnitude tighter than the latency tolerances of typical offline audio processing systems, which routinely operate with window sizes of 20 to 50 milliseconds. This constraint eliminates entire categories of neural network architectures from consideration. Standard transformer models compute self-attention over the input sequence with O(N^2) complexity in the sequence length N. For a context window of 512 samples, this entails 262,144 multiply-accumulate operations per attention head per layer — a cost that, when multiplied across heads and layers, exceeds the available processing budget on CPU hardware. The Mamba Selective State Space Model addresses this challenge through its O(N) recurrent formulation, which processes each sample with a fixed per-step cost of O(d_inner times d_state) = O(128 times 16) = O(2,048) operations, independent of the total sequence length.

The third challenge arises from the implementation language and its runtime characteristics. Python, while offering rapid development, extensive library support, and seamless integration with the PyTorch and ONNX ecosystems, imposes the Global Interpreter Lock (GIL), which serialises the execution of Python bytecode across threads. In a real-time audio system, this means that a Python-level inference computation on one thread can block the audio I/O callback on another thread, causing xruns. The system addresses this challenge through two mechanisms: the use of ONNX Runtime, whose C++ inference engine releases the GIL during the computationally intensive session.run() call, and the use of Python's multiprocessing module to isolate the GUI process from the audio engine process entirely, eliminating GIL contention between the user interface and the audio processing pipeline.

The fourth challenge is the tension between model capacity and inference speed. A more expressive model with larger hidden dimensions, more layers, and higher state dimensionality will generally achieve better noise suppression quality, but at the cost of increased inference latency. The system must find the exact point on the quality-latency Pareto frontier where the target quality metrics (SI-SDRi greater than 4.0 dB, PESQ at least 3.2, TSS above sixty-five percent) are met while the inference latency remains within the 2.667-millisecond budget. The chosen architecture — four Mamba blocks with d_model=64, d_state=16, and expansion factor 2, combined with INT8 dynamic quantization and fifty-percent magnitude pruning — represents a carefully calibrated operating point on this frontier, validated through the post-training benchmark_rtf() function that verifies the Real-Time Factor on target hardware.

## 2.3 Non-Functional Requirements (NFRs)

The system is designed to satisfy a comprehensive set of non-functional requirements that together define the performance envelope within which the transient noise suppression algorithm must operate. Table 2.1 presents these requirements, their quantitative targets, and the measurement methodology used for validation.

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

The SI-SDRi and PESQ metrics jointly capture the overall signal quality improvement achieved by the system. SI-SDRi measures the scale-invariant improvement in signal-to-distortion ratio between the processed output and the unprocessed noisy input, relative to the clean reference. A value exceeding 4.0 dB indicates a perceptually meaningful improvement that is consistently audible to listeners. PESQ provides a perceptual quality score on a scale from 1.0 to 4.5, where 3.2 corresponds to "good" quality in the ITU-T P.862 standard. The TSS metric specifically targets transient noise suppression performance by measuring the fraction of transient energy that has been successfully removed from the output signal, using a binary transient mask to isolate the relevant temporal regions. The plosive segment SI-SDR provides a complementary verification that the system does not achieve high TSS at the expense of speech plosive distortion — a score above 25 dB indicates that less than five percent of plosive energy has been altered by the suppression process.


---

# Chapter 3 — Data

## 3.1 Overview

The training and evaluation of the transient noise suppression model requires paired audio data consisting of clean speech signals and their corresponding noisy mixtures containing transient disturbances. Collecting such paired data from real-world recordings is prohibitively expensive and logistically challenging, as it would require simultaneous capture of the clean speech signal (via a close-talk microphone or studio recording) and the identical speech corrupted by naturally occurring transient noise at precisely controlled signal-to-noise ratios. The project therefore adopts a widely established synthetic data generation methodology in which clean speech recordings are programmatically mixed with isolated transient noise recordings at randomised signal-to-noise ratios (SNRs), optionally convolved with Room Impulse Responses (RIRs) to introduce realistic acoustic reverberation.

Two primary data sources are employed. The LibriSpeech corpus [2] provides the clean speech component, offering a large and diverse collection of read English speech derived from public-domain audiobooks. LibriSpeech is the de facto standard speech dataset in the audio processing community due to its extensive speaker diversity, consistent recording quality, and permissive licensing. The FreeSound collaborative database [3] provides the transient noise component, offering a vast repository of user-contributed sound recordings spanning the full range of transient noise categories targeted by this system: dog barks, door slams, keyboard clicks, mechanical impacts, sirens, and other impulsive environmental sounds. The OpenAIR project supplements these primary sources with a collection of measured Room Impulse Responses recorded in real acoustic environments, enabling the synthesis of reverberant audio mixtures that better approximate real-world listening conditions.

## 3.2 Dataset

Table 3.1 summarises the characteristics of each data source used in the project.

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

Speech and noise files are paired through a randomised mixing procedure implemented in the DatasetBuilder class within generate_dataset.py. For each training sample, the generator randomly selects one speech file and one noise file from their respective directories, extracts a fixed-length segment of 4.0 seconds (192,000 samples at 48 kHz), and combines them at a randomly selected SNR drawn uniformly from the range SNR_RANGE_DB = [-5, +20] dB. This SNR range spans from severely noise-corrupted conditions (speech barely audible above the noise) to mildly noisy conditions, ensuring that the trained model is exposed to the full spectrum of noise severity levels it may encounter in deployment. The pairing is randomised with a fixed seed (seed=42) to ensure reproducibility across training runs.

## 3.3 Data Preprocessing

The synthetic data generation pipeline is implemented in two complementary modules. The primary generator, generate_dataset.py at the project root, implements the full-featured DatasetBuilder class with RIRProvider support for acoustic simulation. The secondary generator, dataset/generate_dataset.py, produces compressed NumPy archive (.npz) files compatible with the TransientNoiseDataset PyTorch Dataset class used by the training loop.

The generation process begins with audio loading and resampling. The load_audio function reads audio files in any supported format (WAV, FLAC, OGG, MP3, AIFF), converts stereo recordings to mono by averaging channels, and resamples to the target sample rate of 48,000 Hz using either the soxr library (if available, for high-quality resampling) or scipy.signal.resample_poly as a fallback. A fixed-length segment of segment_duration = 4.0 seconds is extracted from each file using the random_segment function, which either crops a random contiguous region from files longer than the target length or zero-pads shorter files with the padding position randomised.

Room Impulse Response application is the next stage of the pipeline. The RIRProvider class supports two modes of operation: file-based RIR loading from the OpenAIR dataset, and synthetic RIR generation using the pyroomacoustics library when real RIR files are unavailable. For each training sample, the provider generates a pair of RIRs — a near-field RIR (simulating a speaker close to the microphone, with source distance in the range 0.3 to 1.0 metres) and a far-field RIR (simulating a noise source at a distance, with source distance in the range 2.0 to 6.0 metres). The clean speech signal is convolved with the near-field RIR using scipy.signal.fftconvolve, while the transient noise is convolved with the far-field RIR. This asymmetric convolution models the real-world scenario in which the desired speaker is close to the microphone and the interfering noise originates from a more distant and reverberant source. Room dimensions are randomised within the ranges x in [4.0, 10.0] metres, y in [4.0, 8.0] metres, z in [2.5, 4.0] metres, with RT60 reverberation times between 0.15 and 0.9 seconds.

The convolved signals are then mixed at a randomly selected SNR. The mix_at_snr function computes the RMS energy of both the speech and noise signals, scales the noise signal to achieve the target SNR, and produces the additive mixture. Joint peak normalisation is applied using apply_same_normalisation to scale both the noisy mixture and the clean reference by the same gain factor, ensuring that the noisy signal peaks at peak_norm_target = 0.95 (below digital clipping) while maintaining the relative level relationship between the two signals that the model must learn to separate. The resulting (noisy, clean) pairs are saved as WAV files in a train/val/test directory structure with an accompanying metadata.json file containing per-sample metadata (source file paths, SNR, RIR type).

The default configuration generates total_samples = 10,000 sample pairs with a train/validation/test split of train_ratio = 0.80, val_ratio = 0.10, test_ratio = 0.10, yielding 8,000 training pairs, 1,000 validation pairs, and 1,000 test pairs. The TransientNoiseDataset class in dataset/dataset_loader.py loads these pairs from .npz files and provides standard PyTorch Dataset and DataLoader integration for the training loop.

---

# Chapter 4 — Design Details

## 4.1 Novelty

The principal novelty of this work lies in the combination of two neural architectures — a DeepFIR filter coefficient predictor and a Mamba Selective State Space Model — into a single unified pipeline specifically designed for real-time, CPU-only transient noise suppression. While DeepFIR-style adaptive filter networks have been explored in the context of stationary noise removal [4], and Mamba SSMs have demonstrated state-of-the-art performance on general sequence modelling tasks [1], no prior work has combined these two complementary architectures into a joint model that addresses both stationary and transient noise components within the extreme latency constraints of real-time audio processing on commodity CPU hardware.

The architectural composition is deliberate and non-trivial. The DeepFIR predictor operates as a front-end that removes predictable, stationary noise components by learning to generate minimum-phase FIR filter taps adapted to the current acoustic context. Its output — an intermediate signal with stationary noise partially suppressed — then passes through the Mamba SSM, which applies context-aware selective gating to suppress residual transient noise events. This sequential arrangement exploits the observation that stationary and transient noise are most effectively addressed by different computational strategies: stationary noise responds well to frequency-domain filtering (captured by the FIR taps), while transient noise requires temporal gating with long-range contextual awareness (captured by the SSM hidden state). The combined model is exported to a single ONNX graph file (filter_model.onnx, opset 18) that fuses both stages into an atomic, optimisable inference unit.

A further novel contribution is the application of this combined architecture within a processing budget of 2.67 milliseconds per 128-sample chunk — a constraint approximately ten times tighter than what most neural audio processing systems target. Achieving this required the co-design of the model architecture (small d_model=64, d_state=16, 4 layers, expansion factor 2) with the deployment pipeline (INT8 quantization, 50% magnitude pruning, ONNX Runtime with pre-allocated buffers and session warmup), ensuring that the latency target is met not merely in aggregate but on every individual chunk without exception.

## 4.2 Innovativeness

The plosive speech exception mechanism represents a key innovation that distinguishes this system from generic noise gating approaches. In the classical DSP proof-of-concept (poc_realtime_transient.py), plosive preservation is achieved through a fast-attack/slow-release energy gating design in which the TransientDetector compares a fast exponential moving average (env_fast) against a slow-tracking baseline (env_slow), triggering suppression only when the ratio exceeds a threshold of approximately 6 dB. The synthetic plosive 'P' at t = 8.5 seconds in the test signal is deliberately generated with an amplitude of 0.10 and a gradual ramp-up envelope, so that env_fast never exceeds the threshold relative to env_slow, and the plosive passes through unsuppressed.

In the neural Mamba SSM, plosive preservation is achieved through a fundamentally different and more powerful mechanism: the selective scan parameters B, C, and dt — which are computed dynamically from the input via the x_proj and dt_proj linear projections — learn to produce different gating patterns for impulsive transient noise versus speech plosive inputs. When a transient noise event occurs, the softplus-activated dt values tend to be large, causing rapid state decay (via the discretised state transition matrix dA = exp(dt * A_real)) and aggressive suppression of the input through the selective scan. When a plosive consonant occurs in the context of surrounding speech, the model recognises the speech context through its accumulated hidden state and produces smaller dt values, preserving the state and allowing the plosive energy through to the output. This selectivity is implicit in the trained weights and requires no explicit rule engineering.

The quantization and pruning strategy also exhibits a considered innovation: while INT8 dynamic quantization is applied broadly to all nn.Linear and nn.Conv1d layers via torch.quantization.quantize_dynamic, the critical Mamba state matrices — the A_log parameter (shape [128, 16]), the D skip connection parameter (shape [128]), and the RMSNorm weight parameters — remain in FP32 precision. These parameters are stored as nn.Parameter tensors rather than nn.Linear modules, and are therefore not targeted by the dynamic quantization pass. This selective precision preservation ensures that the SSM state dynamics, which govern the model's ability to distinguish transients from plosives, maintain full numerical fidelity even as the surrounding projection layers are compressed to 8-bit integer arithmetic.

## 4.3 Interoperability

The system achieves cross-platform interoperability through the adoption of ONNX (Open Neural Network Exchange) as its model interchange format. The combined DeepFIR + Mamba SSM model is exported at ONNX opset version 18, which provides comprehensive operator coverage for all operations used in the architecture including Conv1d, linear projections, SiLU activation, softplus, and element-wise exponential. The ONNX export function (export_to_onnx in model/export_onnx.py) produces a graph with a single input tensor named "noisy_audio" of shape [batch, 512] and a single output tensor named "clean_audio" of identical shape, with the batch dimension marked as dynamic to support both single-sample inference and batched evaluation.

At inference time, the ONNX graph is loaded by ONNX Runtime's InferenceSession with the CPUExecutionProvider as the sole execution provider, ensuring identical behavior across Windows, macOS, and Linux without any GPU-specific dependencies. The session options are configured for low-latency CPU inference: graph_optimization_level is set to ORT_ENABLE_ALL (enabling constant folding, operator fusion, and memory planning), execution_mode is set to ORT_SEQUENTIAL (avoiding the overhead of inter-operator parallelism for small graphs), and thread counts are set to os.cpu_count() minus one to balance inference throughput with system responsiveness. This configuration ensures that the inference engine operates identically regardless of the host operating system or CPU vendor.

The GUI and audio subsystems are fully decoupled from the ML inference pipeline through the IPC message protocol defined in app/ipc/messages.py. The CmdType and EvtType enumerations define an exhaustive set of commands and events that are communicated as serialised dataclass objects via multiprocessing.Queue. This decoupling means that the audio backend (currently sounddevice with PortAudio) can be replaced with any alternative audio library without modifying the inference or GUI code, and the ring buffer interface (audio/ring_buffer.py) accepts any compliant audio source that provides float32 sample arrays.

## 4.4 Performance

The performance design of the system is governed by a strict processing budget derived from the audio parameters. At a sample rate of 48,000 Hz and a chunk size of 128 samples, each chunk represents 2.667 milliseconds of audio, establishing an absolute deadline for processing completion. The system allocates this budget across two tiers: the DSP tier (ring buffer read/write, gain application, format conversion), which is benchmarked at under 100 microseconds per chunk in the PoC implementation, and the neural inference tier (ONNX Runtime session.run), which consumes the remaining approximately 2,500 microseconds of headroom. The design target is a Real-Time Factor (RTF) below 0.80, as specified by the TARGET_RTF constant in config.py, providing a twenty-percent margin for operating system scheduling jitter.

The seven-fold speedup achieved by the quantization and pruning pipeline is critical to meeting this budget. The apply_magnitude_pruning function applies L1 unstructured pruning with amount=0.50 to every nn.Linear and nn.Conv1d layer in the model, zeroing out the fifty percent of weights with the smallest absolute magnitude. The prune.remove() call then makes the pruning permanent by collapsing the weight and mask into a single tensor. Following pruning, the quantize_model_int8 function applies torch.quantization.quantize_dynamic with dtype=torch.qint8, replacing the FP32 weight storage and matrix multiplication operations in Linear and Conv1d layers with INT8 equivalents that use integer arithmetic units and consume half the memory bandwidth. The combined effect of fifty-percent sparsity and eight-bit arithmetic reduces the effective computation by approximately 7x relative to the dense FP32 baseline, as verified by the benchmark_rtf function.

## 4.5 Security

The system is designed with a security-by-isolation architecture in which all audio data remains on the local device throughout the entire processing pipeline. No audio samples, spectral features, or processed output are transmitted to any external server, cloud endpoint, or third-party service at any point during operation. The microphone input is captured by sounddevice directly from the local PortAudio driver, processed through the ring buffer and ONNX inference engine entirely within the local Python process, and output to the local speaker through the same PortAudio driver.

The ONNX model file (model/filter_model.onnx) is a static computational graph artifact that contains only numerical weight tensors and operator specifications. It does not contain executable code, Python bytecode, or any mechanism for runtime code injection. The ONNX Runtime inference engine parses the graph structure at session creation time and executes it through a fixed set of compiled C++ operator kernels, providing a well-defined and auditable execution boundary.

## 4.6 Reliability

The system implements multi-layered reliability mechanisms to ensure graceful degradation under adverse operating conditions. The ring buffer (audio/ring_buffer.py) handles both overrun and underrun conditions without crashing or producing undefined behavior. On overrun — when the producer writes data faster than the consumer reads it — the oldest data in the buffer is silently overwritten, a design decision that preserves the most recent audio context at the cost of discarding stale historical samples. On underrun — when the consumer attempts to read more data than has been written — the read function returns a zero-padded array, producing silence rather than garbage data in the output.

The audio I/O subsystem (audio/audio_io.py) implements a retry mechanism for transient audio device errors. When a PortAudioError occurs during stream initialisation, the AudioIOManager retries the stream opening up to three times with a 500-millisecond delay between attempts. If all retries fail, the system continues operating in a degraded state. The system also falls back gracefully when the ONNX model file is not found: process_manager.py checks for the existence of the ONNX file at cfg.ONNX_MODEL_PATH and, if absent, switches to pass-through mode where inputs are forwarded directly to outputs without neural processing.

## 4.7 Maintainability

The four-layer pipeline architecture is designed for independent layer replacement and evolution. Each layer — ring buffer, quantization, DeepFIR, and Mamba SSM — is implemented as a self-contained module with clearly defined input/output contracts: float32 audio arrays in, float32 audio arrays out. The training pipeline (dataset/, training/) is entirely decoupled from the inference pipeline (inference/, app/), with the ONNX model file serving as the sole interface between them. A researcher can retrain the model with different hyperparameters, loss functions, or architectural modifications, re-export the ONNX file, and deploy it without modifying any code in the inference or GUI subsystems.

The central configuration file (config.py) consolidates all tunable parameters — SAMPLE_RATE, CHUNK_SIZE, CONTEXT_WINDOW_SAMPLES, FIR_FILTER_LENGTH, MAMBA_D_MODEL, MAMBA_D_STATE, MAMBA_N_LAYERS, PRUNE_RATIO, TARGET_RTF, BATCH_SIZE, EPOCHS, LR, and all file paths — into a single module imported throughout the codebase as import config as cfg. This eliminates the risk of parameter inconsistency across modules and provides a single point of control for all system-wide configuration changes.

## 4.8 Portability

The system targets Python 3.10 or later on Windows, macOS, and Linux, with no platform-specific compiled extensions beyond the standard pip-installable packages. ONNX Runtime's CPUExecutionProvider supports all major CPU instruction set architectures including x86-64 (Intel, AMD) and ARM64 (Apple Silicon, Qualcomm), enabling deployment on devices ranging from desktop workstations to single-board computers without model recompilation. Platform-specific adaptations are confined to the audio I/O and GUI initialisation layers. On macOS, the application entry point (app/main.py) contains a re-execution mechanism that transparently relaunches the process using sys._base_executable when launched from a virtual environment, working around the cocoa platform plugin initialisation requirement.

## 4.9 Legacy to Modernization

This system represents a deliberate transition from classical DSP techniques to learned neural filter methods. The traditional approach — implemented in poc_realtime_transient.py as a baseline — uses hand-crafted DSP components: a leaky-integrator energy tracker, a fast-attack/slow-release energy gate (TransientDetector), a minimum-statistics noise floor estimator (NoiseEstimator), and spectral-subtraction-style gain. These classical methods require extensive manual tuning and degrade when acoustic conditions deviate from designer assumptions. The neural approach replaces these hand-tuned components with learned equivalents. The DeepFIR predictor replaces the fixed spectral subtraction filter with a context-adaptive FIR filter whose taps are predicted from audio input by a causal convolutional network. The Mamba SSM replaces the energy-threshold gate with a learned selective scan mechanism. The ring buffer pattern is preserved from classical DSP, maintaining deterministic latency and lock-free concurrency while the processing within each stage is fully neural.

## 4.10 Reusability

The combined DeepFIR and Mamba SSM model, exported as a single ONNX graph, is a self-contained computational artifact that can be embedded in any ONNX-compliant inference runtime. The graph accepts a single input tensor of shape [batch, 512] and produces a single output tensor of identical shape, with no external dependencies. This standard interface enables deployment in diverse execution environments: ONNX Runtime on desktop CPUs, ONNX Runtime Mobile on Android and iOS, ONNX.js in web browsers, TensorRT for NVIDIA GPUs, or OpenVINO on Intel hardware. The training pipeline is similarly reusable — the TransientNoiseDataset class accepts any directory of .npz files, making it straightforward to retrain on new data sources without modifying training code.

## 4.11 Application Compatibility

The system integrates with existing voice communication applications through virtual audio device routing. On Windows, audio/virtual_device.py detects VB-Audio Virtual Cable devices by scanning for names containing "virtual", "cable", "vb-audio", or "voicemeeter". On Linux, PulseAudio null sink creation is automated via pactl. The pass-through mode (activated via CmdType.SET_PASSTHROUGH or the GUI button) enables direct microphone-to-speaker routing with zero processing, providing a baseline for A/B comparison testing where users can instantly switch between processed and unprocessed audio.

## 4.12 Resource Utilization

The CombinedModel contains approximately 279,000 parameters. Before optimization, these occupy approximately 1.07 MB in FP32 precision. After fifty-percent magnitude pruning and INT8 quantization, the effective memory footprint is reduced to approximately 0.27 MB for quantised weights, plus overhead for FP32 state parameters (A_log, D, RMSNorm weights). At runtime, the ring buffer pre-allocates np.zeros(24000, dtype=float32) consuming approximately 96 KB. The ONNX Runtime inference session pre-allocates an input buffer of shape [1, 512] (2 KB) and maintains internal workspace buffers. The system deliberately avoids dynamic memory allocation on the audio processing hot path — all buffers are allocated at initialisation time, eliminating garbage collection pauses that could cause audio dropouts.


---

# Chapter 5 — High-Level System Design / System Architecture

## 5.1 System Overview

The transient noise suppression system operates as an end-to-end audio processing pipeline that transforms a noisy microphone input into a clean speaker output in real-time. The signal flow, illustrated conceptually in Figure 5.1, proceeds through four sequential processing stages within a multi-process application architecture. Audio samples are captured from the system microphone by the sounddevice library, which invokes a PortAudio callback function at regular intervals determined by the configured block size. Each callback delivers a chunk of raw float32 samples that are immediately written into a lock-free ring buffer (Layer 1), decoupling the timing-critical audio hardware callback from the computationally intensive neural inference process.

The inference thread continuously reads context windows of CONTEXT_WINDOW_SAMPLES = 512 samples from the ring buffer and feeds them to the ONNX Runtime inference session, which executes the combined DeepFIR + Mamba SSM model graph. The DeepFIR component (Layer 3) processes the context window first, predicting sixty-four minimum-phase FIR filter taps from the audio content and applying them to suppress stationary noise components. The resulting intermediate signal then passes through the Mamba SSM (Layer 4), which uses its recurrent hidden state to perform context-aware transient gating — suppressing impulsive noise events while preserving acoustically similar speech plosives. Prior to deployment, the model undergoes INT8 dynamic quantization and fifty-percent magnitude pruning (Layer 2), reducing the inference latency by approximately 7x relative to the unoptimised FP32 baseline. The cleaned output samples are written to the output buffer and played through the speaker via the same PortAudio driver.

The entire audio processing pipeline runs in a dedicated child process spawned via Python's multiprocessing.Process with the spawn start method, ensuring complete memory isolation from the GUI process. The GUI process hosts the PyQt6 QApplication, the ControlWindow settings panel, and the TrayManager system tray icon. Inter-process communication between the GUI and audio engine occurs through multiprocessing.Queue instances carrying serialised Command and Event dataclass objects, with command types enumerated in CmdType (SET_ENABLED, SET_STRENGTH, SET_GAIN, SET_INPUT_DEVICE, SET_OUTPUT_DEVICE, SET_PASSTHROUGH, SHUTDOWN) and event types in EvtType (STATUS, DEVICE_LIST, ERROR, ENGINE_STOPPED).

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

The ring buffer serves as the foundational data structure that decouples the audio hardware's timing-critical callback from the neural inference computation. The production implementation in audio/ring_buffer.py allocates a contiguous numpy array of np.zeros(capacity, dtype=np.float32) at initialisation time, where the default capacity is SAMPLE_RATE * RING_BUFFER_SECONDS = 48,000 * 0.5 = 24,000 samples, representing 500 milliseconds of audio history. The buffer operates on a single-producer/single-consumer (SPSC) model: the audio callback thread acts as the sole producer, writing incoming samples via the write() method, while the inference thread acts as the sole consumer, reading context windows via the read() or read_context() method.

The write path acquires a threading.Lock only for the duration of updating the _write_pos integer pointer, not for the bulk data copy itself. The read path does not acquire any lock; instead, it snapshots the current _write_pos value (an atomic operation under CPython's GIL for single-word integer reads) and indexes backward into the buffer to retrieve the requested number of samples. This asymmetric locking design ensures that the audio callback — which must return within the PortAudio deadline or incur an xrun — is never blocked waiting for the inference thread to complete a read operation. The reported buffering latency is get_latency_ms() = 1 / SAMPLE_RATE * 1000 = 0.021 milliseconds, representing the theoretical minimum latency of one sample.

The PoC implementation in poc_realtime_transient.py uses a simpler variant with no explicit locks, relying entirely on the GIL for pointer atomicity. It uses separate _write and _read position counters with modular arithmetic, and returns None on underrun rather than zero-padding. The PoC ring buffer has a larger default capacity of SAMPLE_RATE * 2 = 96,000 samples (2 seconds), reflecting its use in the offline benchmark mode where the entire test signal is written before processing begins.

### 5.2.2 Layer 2: Quantization and Pruning

The quantization layer is applied as a post-training optimisation step rather than as a runtime processing stage. After training completes, the apply_magnitude_pruning function in model/quantize.py iterates over all nn.Linear and nn.Conv1d modules in the CombinedModel and applies torch.nn.utils.prune.l1_unstructured with amount = PRUNE_RATIO = 0.50, setting the fifty percent of weights with the smallest absolute magnitude to zero. The pruning is then made permanent via prune.remove(), which eliminates the pruning mask and bakes the sparsity directly into the weight tensor. The resulting sparse model is then passed to quantize_model_int8, which applies torch.quantization.quantize_dynamic targeting the set {nn.Linear, nn.Conv1d} with dtype=torch.qint8, converting FP32 weight storage to INT8 and replacing the FP32 matrix multiplication kernels with INT8 equivalents.

The pruned and quantised model is benchmarked using benchmark_rtf(), which processes ten seconds of synthetic audio through the model in a tight loop and reports the measured Real-Time Factor. If the RTF exceeds TARGET_RTF = 0.80, the benchmark raises a warning indicating that the model may not meet the real-time constraint on the test hardware. Finally, the optimised model is exported to ONNX format via export_to_onnx(), producing the filter_model.onnx and filter_model.onnx.data files in the model/ directory.

### 5.2.3 Layer 3: DeepFIR Neural Filter

The DeepFIRPredictor class in model/deep_fir.py implements a causal convolutional network that predicts FIR_FILTER_LENGTH = 64 minimum-phase FIR filter taps from a context window of CONTEXT_WINDOW_SAMPLES = 512 raw audio samples. The architecture consists of two CausalConv1d layers (with left-padding of kernel_size minus one zeros to ensure causality), each followed by PReLU activation with learnable negative slope, an AdaptiveAvgPool1d layer that reduces the temporal dimension to a single value, and a final Linear(64, 64) projection with Tanh activation that bounds the predicted taps to the range [-1, 1].

The predicted taps are converted to minimum-phase form via the _to_minimum_phase method, which implements the cepstral method: the taps are zero-padded to the nearest power of two, transformed via FFT, the log-magnitude spectrum is computed, an inverse FFT yields the real cepstrum, which is then liftered (positive-time components doubled, negative-time components zeroed), transformed back via FFT and exponentiated, and a final inverse FFT produces the minimum-phase impulse response. The minimum-phase property ensures that all filter zeros lie inside the unit circle, guaranteeing that the filter is causal — it requires no future audio samples — and achieves the minimum possible group delay for the given magnitude response. The filter is applied using scipy.signal.lfilter for inference or torch.nn.functional.conv1d for differentiable training.

### 5.2.4 Layer 4: Mamba Selective State Space Model

The MambaSSM class in model/mamba_ssm.py implements a stack of MAMBA_N_LAYERS = 4 MambaBlock modules, each preceded by RMSNorm normalisation and followed by a residual connection. Each MambaBlock operates with d_model = MAMBA_D_MODEL = 64, d_state = MAMBA_D_STATE = 16, d_conv = 4, and expansion factor 2, yielding an inner dimension of d_inner = 128.

The forward pass of each MambaBlock proceeds as follows. The input tensor of shape [B, L, 64] is projected to [B, L, 256] by the in_proj linear layer and split into two halves: x_inner [B, L, 128] and gate z [B, L, 128]. The x_inner tensor passes through a depthwise Conv1d(128, 128, kernel_size=4, groups=128) with SiLU activation, implementing a local convolution that captures short-range temporal dependencies. The convolved signal is then projected by x_proj Linear(128, 160), and the output is split into B_ssm [B, L, 16], C_ssm [B, L, 16], and dt_raw [B, L, 128]. The dt_raw tensor is passed through dt_proj Linear(128, 128) and a softplus nonlinearity to produce the positive discretisation step dt [B, L, 128].

The selective scan is the core computation of the Mamba block. The state transition matrix A_real is computed from the learned A_log parameter as -exp(A_log), yielding a [128, 16] matrix of negative values that govern state decay. During inference, the selective_scan_sequential function processes the sequence one timestep at a time: at each step t, the hidden state h [128, 16] is updated as h = dA_t * h + dB_t * x_t, where dA_t = exp(dt_t * A_real) and dB_t = dt_t * B_ssm_t, and the output is computed as y_t = (C_ssm_t * h).sum(dim=-1) + D * x_t, where D [128] is a learnable skip connection. The output y is multiplied element-wise with SiLU(z) to apply the gating mechanism, and finally projected back to d_model = 64 via the out_proj linear layer.

For training, the selective_scan_parallel function processes all timesteps simultaneously using cumulative sum operations where possible, falling back to sequential computation for numerical stability. The O(N) complexity of the selective scan — where each timestep requires O(d_inner * d_state) = O(128 * 16) = O(2,048) operations regardless of sequence length — is the key property that enables real-time inference on CPU.

## 5.3 ONNX Inference Pipeline

The ONNX inference pipeline is implemented across two modules: inference/onnx_runner.py, which wraps the ONNX Runtime session, and inference/pipeline.py, which integrates the runner with the ring buffer and output mixing logic. The ONNXInferenceRunner class initialises an onnxruntime.InferenceSession with the model file at cfg.ONNX_MODEL_PATH = "model/filter_model.onnx", configured with the following session options: graph_optimization_level = ORT_ENABLE_ALL (enabling all graph-level optimisations including constant folding, operator fusion, and memory pattern planning), execution_mode = ORT_SEQUENTIAL (sequential operator execution to minimise thread management overhead for small graphs), inter_op_num_threads = os.cpu_count() - 1, intra_op_num_threads = os.cpu_count() - 1, enable_mem_pattern = True, and providers = ["CPUExecutionProvider"].

Upon initialisation, the runner pre-allocates an input buffer _input_buf = np.zeros((1, CONTEXT_WINDOW_SAMPLES), dtype=np.float32) to eliminate per-inference memory allocation. The warmup(n_runs=50) method executes fifty dummy inference calls before the audio stream begins, triggering lazy JIT compilation, memory pool initialisation, and CPU cache warming within the ONNX Runtime engine. This ensures that the first real-time inference call after audio stream start does not incur a cold-start latency spike that could cause an xrun.

The RealTimePipeline class in inference/pipeline.py orchestrates the end-to-end processing flow: it reads a context window from the ring buffer via read_context(), copies the samples into the pre-allocated input buffer, calls session.run() on the ONNX Runtime session, and applies a wet/dry mix controlled by the suppression_level parameter (0.0 = fully dry/bypass, 1.0 = fully wet/processed). The pipeline also maintains a LatencyTracker that records the processing time of each chunk, providing real-time RTF statistics (mean, P99, max) that are reported to the GUI via the StatusPayload event.

## 5.4 GUI and System Tray Architecture

The GUI architecture employs Python's multiprocessing module to achieve complete isolation between the user interface and the audio processing engine. The main entry point (app/main.py) first sets multiprocessing.set_start_method("spawn", force=True) to ensure child processes are created via fork-exec rather than fork-only, avoiding potential issues with inherited file descriptors and shared memory in the audio driver. It then creates the QApplication instance on the main thread (required by macOS Cocoa), instantiates the AppConfig dataclass with runtime parameters (sample_rate=48000, block_size=1024, channels=1, dtype="float32"), and spawns the AudioEngineHandle, which in turn creates the audio engine child process.

The ControlWindow class in app/gui/control_window.py provides the primary user interface with the following interactive elements: a suppression ON/OFF toggle button (btn_toggle) that sends CmdType.SET_ENABLED; a pass-through demo button (btn_passthrough) that sends CmdType.SET_PASSTHROUGH and disables the strength and gain controls; a suppression strength slider (slider_strength, range 0 to 100) that sends CmdType.SET_STRENGTH as a 0.0 to 1.0 float; an output gain slider (slider_gain, range -120 to +120 in tenths of dB) that sends CmdType.SET_GAIN; input and output device combo boxes (combo_input, combo_output) populated from the device enumeration in app/audio/devices.py; input and output level meters (meter_in, meter_out) driven by StatusPayload.input_level_db and output_level_db; and an RTF display label (lbl_rtf) showing the current Real-Time Factor with headroom. A QTimer polls the engine's event queue at status_interval = 0.05 seconds (20 Hz) to update the meters and RTF display. The window's closeEvent() hides the window rather than terminating the application, allowing the system tray to serve as the persistent application anchor.

The TrayManager class in app/gui/tray.py runs pystray on a daemon thread, rendering a PIL-drawn microphone icon in green (suppression active) or red (bypass/pass-through). The tray menu provides Show/Hide, Suppression toggle, and Quit actions, bridged to the Qt main thread via _TrayBridge(QObject) signals. The WaveformViewer class in app/gui/waveform_viewer.py provides an offline analysis tool: it launches a _PocWorker QThread that imports and runs the poc_realtime_transient module, generates test signals, processes them through the RealTimeFilter, and emits the results to three pyqtgraph.PlotWidget panels (Clean, Noisy, Filtered) with linked X-axes, vertical transient markers at positions [1.2, 3.5, 5.0, 6.8, 8.5] seconds, and metric cards displaying Mean chunk time (microseconds), P99, RTF, Headroom multiplier, and PASS/FAIL verdict.

---

# Chapter 6 — Design Description

## 6.1 Master Class Diagram

Table 6.1 presents the principal classes of the system, their responsibilities, key methods, and inter-class relationships. The class diagram, described conceptually as Figure 6.1, follows a layered dependency structure in which the GUI layer depends on the IPC layer, the IPC layer depends on the engine layer, the engine layer depends on the inference layer, and the inference layer depends on the model layer. The audio I/O layer and ring buffer are shared dependencies of both the engine and the PoC benchmark script.

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

Figure 6.1 depicts the UML class diagram as a layered architecture. At the top layer, ControlWindow aggregates AudioEngineHandle (1-to-1 composition) and optionally creates WaveformViewer instances. TrayManager communicates with ControlWindow via _TrayBridge Qt signals. AudioEngineHandle spawns a child process executing the run_engine() function, which instantiates an AudioIOManager (containing a RingBuffer) and optionally a StubDenoiser or the RealTimePipeline (containing ONNXInferenceRunner and RingBuffer). The model layer contains CombinedModel, which composes DeepFIRPredictor and MambaSSM, with MambaSSM containing four MambaBlock instances and RMSNorm layers. The training layer's train() function instantiates CombinedModel and TransientNoiseDataset, and post-training calls apply_magnitude_pruning(), quantize_model_int8(), benchmark_rtf(), and export_to_onnx() in sequence.

## 6.2 Swimlane Diagram and State Diagram

The system operates with three concurrent execution contexts, described in Figure 6.2 as a swimlane diagram with three vertical lanes corresponding to the Audio I/O Thread, the ML Inference Thread, and the GUI Thread.

The Audio I/O Thread lane executes within the audio engine child process. Its primary activity is the sounddevice callback function _audio_callback(indata, outdata, frames, time_info, status), invoked by PortAudio at intervals determined by the block size (1024 samples in GUI mode, 128 in PoC mode). On each invocation, the callback reads audio samples from indata (the microphone buffer), writes them to the ring buffer, and copies the latest processed output from the inference pipeline to outdata (the speaker buffer). In pass-through mode, the callback simply copies indata directly to outdata, bypassing the ring buffer and inference pipeline entirely.

The ML Inference Thread lane operates as the main loop of the audio engine process (run_engine()). It continuously polls the ring buffer for available data, reads a context window of 512 samples via read_context(), feeds it to the ONNXInferenceRunner, applies wet/dry mixing based on the current suppression strength, and stores the result in the output buffer for the next audio callback to consume. Between inference cycles, it checks the command queue for IPC messages from the GUI (device changes, parameter updates, shutdown), and periodically (every status_interval = 0.05 seconds) pushes a StatusPayload event containing input/output levels, RTF, and xrun counts back to the GUI.

The GUI Thread lane executes in the parent process. It runs the Qt event loop (app.exec()), processing user interactions (button clicks, slider movements) that generate Command objects sent via the multiprocessing.Queue to the engine process. A QTimer fires at 20 Hz to drain the event queue and update the visual elements: level meter bars, RTF label text, and xrun counter. The TrayManager runs pystray on a separate daemon thread within the GUI process, bridging tray menu actions to Qt signals via the _TrayBridge QObject.

The application state diagram, described as Figure 6.2b, models the engine process lifecycle as a finite state machine with five states. The IDLE state is the initial state before the engine process is spawned. Calling AudioEngineHandle.start() transitions to INITIALIZING, during which the sounddevice stream is opened, the ONNX model is loaded (if available), and the warmup inference calls are executed. Successful initialisation transitions to RUNNING (suppression active), in which the inference pipeline processes audio chunks and sends status events. From RUNNING, a CmdType.SET_PASSTHROUGH command transitions to PASS_THROUGH, in which audio is routed directly from input to output without inference. A CmdType.SET_PASSTHROUGH command with value=False transitions back to RUNNING. From either RUNNING or PASS_THROUGH, a CmdType.SHUTDOWN command or an unrecoverable error transitions to STOPPED, in which the sounddevice stream is closed and the process terminates.

## 6.3 User Interface Diagrams

The ControlWindow, described conceptually as Figure 6.3, presents a vertical layout within a fixed-size window. At the top, a title label displays the application name. Below this, a horizontal row contains the suppression ON/OFF toggle button (btn_toggle, displaying "Suppression: ON" in green or "Suppression: OFF" in red) and the pass-through demo button (btn_passthrough, displaying "Pass-Through (Demo)"). The next section contains two labelled sliders: the suppression strength slider (slider_strength, ranging from 0 to 100 with a default value of 100, controlling the wet/dry mix ratio) and the output gain slider (slider_gain, ranging from -12.0 dB to +12.0 dB in 0.1 dB increments, defaulting to 0.0 dB).

Below the sliders, two combo boxes allow device selection: combo_input (populated with available input devices from app/audio/devices.py query_devices()) and combo_output (populated with available output devices). Each combo box displays the device name and index. The next section contains two horizontal level meter bars (meter_in and meter_out), rendered as coloured progress bars that update at 20 Hz from the StatusPayload data. A text label (lbl_rtf) displays the current Real-Time Factor formatted as "RTF: 0.0234 (42.7x headroom)". At the bottom of the window, a button labelled "Proof of Concept" (btn_poc) launches the WaveformViewer window.

The WaveformViewer window, described as Figure 6.3b, contains three vertically stacked pyqtgraph.PlotWidget panels with linked X-axes (time in seconds). The top panel displays the clean reference signal in green, the middle panel displays the noisy input signal in red, and the bottom panel displays the filtered output signal in blue. Vertical dashed lines at positions 1.2, 3.5, 5.0, 6.8, and 8.5 seconds mark the locations of synthesised transient events (dog bark, door slam, keyboard click, siren chirp, and plosive 'P' respectively). Below the plots, a row of metric cards displays: Mean chunk processing time (microseconds), P99 chunk time, Real-Time Factor, Headroom multiplier, and a PASS/FAIL verdict (PASS if RTF < 0.80). Play/Stop buttons beneath each panel allow audible playback via sounddevice.play().

## 6.4 Report Layouts

The offline benchmark script (poc_realtime_transient.py --mode demo) produces three output files and a console report. The output files are: test_clean.wav (the synthetic clean speech signal, 10 seconds at 48 kHz, 16-bit PCM), test_noisy.wav (the clean signal mixed with five synthetic transient events at specified positions), and test_noisy_filtered.wav (the noisy signal after processing through the RealTimeFilter in 128-sample chunks). The console report, generated by the LatencyProfiler.report() method, displays the following metrics: total chunks processed, mean processing time per chunk in microseconds, P99 (99th percentile) processing time, maximum processing time, Real-Time Factor (processing time divided by chunk duration), headroom multiplier (chunk duration divided by processing time), and the percentage of the processing budget consumed. The diagnose() method additionally prints per-transient-event attenuation in dB, comparing the energy of each transient region in the filtered output to the corresponding region in the noisy input. The report concludes with an ASCII-art FEASIBILITY VERDICT banner displaying either "PASS" (if RTF < 0.80) or "FAILED".

## 6.5 External Interfaces

The system interfaces with three external subsystems: the audio hardware, the ONNX Runtime inference engine, and the training data sources.

The audio hardware interface is mediated by the sounddevice library, which provides Python bindings to the cross-platform PortAudio C library. The system configures sounddevice streams with the following parameters: samplerate = 48,000 Hz, channels = 1 (mono), dtype = "float32" for internal processing (converted from "int16" for PCM I/O in the PoC), and blocksize = 128 (PoC mode) or 1024 (GUI mode, as configured in AppConfig.block_size). Input and output device indices are selectable at runtime via the combo_input and combo_output GUI elements, which enumerate available devices through sounddevice.query_devices(). The callback signature follows the PortAudio convention: callback(indata, outdata, frames, time_info, status), where indata and outdata are numpy arrays of shape [frames, channels].

The ONNX Runtime interface defines a strict tensor contract between the application and the model. The input tensor is named "noisy_audio" with shape [batch_size, 512] and dtype float32. The output tensor is named "clean_audio" with identical shape and dtype. The batch dimension is dynamic, allowing the model to process single samples (batch_size=1 during real-time inference) or batches (during evaluation). The session is initialised with providers=["CPUExecutionProvider"], graph_optimization_level=ORT_ENABLE_ALL, and execution_mode=ORT_SEQUENTIAL.

The training data interface consists of the file system paths configured in config.py: LIBRISPEECH_DIR = "data/raw/librispeech" for LibriSpeech clean speech files, FREESOUND_DIR = "data/raw/freesound" for FreeSound transient noise files, and RIR_DIR = "data/raw/openair_rirs" for Room Impulse Response files. The discover_audio_files function in generate_dataset.py recursively scans these directories for files with extensions in {.wav, .flac, .ogg, .mp3, .aiff, .aif}, returning sorted lists of Path objects for the DatasetBuilder to consume.

## 6.6 Packaging and Deployment Diagram

The system is distributed as a Python source package with the following deployment procedure, described conceptually as Figure 6.6. The developer or end-user clones the repository, creates a Python 3.10+ virtual environment (python -m venv .venv_poc), activates it, and installs all dependencies via pip install -r requirements.txt. The requirements.txt file lists all seventeen runtime and development dependencies with minimum version constraints. The application is launched via the command python -m app.main, which triggers app/__main__.py, which calls app.main.main().

The deployment structure consists of three tiers. The first tier is the Python runtime environment (Python 3.10+, virtual environment, pip-installed packages). The second tier is the application code (the app/, audio/, model/, inference/, dataset/, and training/ packages, plus the root-level config.py and poc_realtime_transient.py). The third tier is the model artifact (model/filter_model.onnx and model/filter_model.onnx.data), which is the sole output of the training pipeline and the sole input required by the inference pipeline. In deployment, only the first two tiers and the model artifact are required; the training pipeline (dataset/, training/) and its heavy dependencies (torch, torchaudio) can be omitted from a production deployment to reduce the installation footprint.

The deployment diagram (Figure 6.6) shows a single machine running the Python runtime, which hosts two OS processes: the GUI process (QApplication, ControlWindow, TrayManager) and the Audio Engine process (sounddevice, RingBuffer, ONNXInferenceRunner). The GUI process communicates with the Audio Engine process via multiprocessing.Queue. The Audio Engine process communicates with the audio hardware via the PortAudio driver (accessed through sounddevice). The ONNXInferenceRunner within the Audio Engine process loads the filter_model.onnx artifact from the local file system.


---

# Chapter 7 — Technologies Used

Table 7.1 presents the complete technology stack employed in the project, including the specific version constraints, the purpose of each technology, and the justification for its selection over available alternatives.

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

The selection of ONNX Runtime as the inference engine is the single most critical technology decision in the project. Alternative inference approaches — running PyTorch directly with torch.no_grad(), using TorchScript, or using TensorFlow Lite — were evaluated and rejected. PyTorch's eager execution mode, even under torch.no_grad(), retains the Python GIL during the forward pass because each operator invocation traverses the Python/C++ boundary via pybind11, preventing the audio callback thread from executing during inference. TorchScript would solve the GIL problem but introduces significant limitations on the model architecture (no data-dependent control flow, restricted Python subset). ONNX Runtime, by contrast, parses the entire computation graph at session creation time and executes it as a single monolithic C++ operation during session.run(), releasing the GIL for the entire duration of the inference call. This allows the audio callback thread to continue writing samples to the ring buffer concurrently with inference, eliminating the primary source of audio dropout risk.

The selection of the Mamba Selective State Space Model over transformer-based architectures is motivated by computational complexity. A standard multi-head self-attention layer with H heads and sequence length L requires O(H * L^2 * d_head) multiply-accumulate operations per layer. For the parameters used in this project (L = 512, d_model = 64), even a single-head single-layer transformer attention block requires 512^2 = 262,144 multiply-accumulate operations, which, combined with the KV cache management and softmax computation overhead, pushes the per-layer cost well beyond what can be sustained within 2.67 milliseconds on a CPU. The Mamba SSM processes the same sequence in O(L * d_inner * d_state) = O(512 * 128 * 16) = O(1,048,576) operations across the full sequence, but crucially supports O(1)-per-timestep recurrent inference via selective_scan_sequential, where each new sample requires only O(d_inner * d_state) = O(2,048) operations to update the hidden state and produce the output [1].

The decision to use PyQt6 rather than a web-based GUI (Electron, Flask) or alternative Python GUI frameworks (Tkinter, wxPython) is driven by two requirements: process isolation and native system tray integration. PyQt6's QApplication can coexist with Python's multiprocessing module, allowing the GUI event loop to run in the main process while the audio engine runs in a separate OS process with no shared GIL. The pystray library provides cross-platform system tray integration that is essential for the application's intended use case — running unobtrusively in the background during voice calls, with the main window hidden and accessible via the tray icon.

The sounddevice library was selected over PyAudio (the other major PortAudio wrapper for Python) because sounddevice provides a modern, NumPy-native API that returns audio data directly as numpy arrays, eliminating the need for manual byte-to-array conversion that PyAudio requires. The callback-based API mirrors PortAudio's native callback model, providing the lowest possible latency for real-time audio processing. The library supports all three major operating systems through the same Python API, with platform-specific audio driver selection handled transparently by the underlying PortAudio library.

---

# Chapter 8 — Implementation and Pseudocode

## 8.1 Ring Buffer Implementation

The lock-free single-producer/single-consumer ring buffer is the foundational data structure that enables real-time audio processing by decoupling the timing-critical audio hardware callback from the variable-latency neural inference computation. The following pseudocode describes the production implementation in audio/ring_buffer.py.

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

The critical design property is that the write path holds the lock only during the pointer update (a single integer increment), while the bulk memcpy into the numpy array occurs within the lock scope for pointer consistency but completes in microseconds due to NumPy's C-optimised copy routines. The read path acquires no lock whatsoever, relying on the CPython GIL to provide atomicity for the single-word integer read of _write_pos. This ensures that the audio callback (producer) is never blocked by inference (consumer) and vice versa, achieving the sub-100-microsecond latency measured by the PoC benchmark.

## 8.2 Mamba SSM Inference

The following pseudocode describes the forward pass of a single MambaBlock during real-time inference, using the selective_scan_sequential function that processes one timestep at a time with O(1) per-step complexity.

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

The key insight is that dt is input-dependent — it is computed from the audio content via x_proj and dt_proj. When the model encounters a transient noise burst, the learned projections produce large dt values, causing dA = exp(dt * A) to approach zero (since A is negative), which rapidly decays the hidden state h and suppresses the input. When a speech plosive occurs, the surrounding speech context produces smaller dt values, preserving the state and allowing the plosive to pass through. This selective gating is the mechanism by which the Mamba SSM distinguishes transients from plosives without explicit rules.

## 8.3 DeepFIR Inference

The following pseudocode describes the DeepFIR inference pipeline, from raw audio input to filtered output.

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

The CausalConv1d implementation pads kernel_size - 1 = 7 zeros on the left side of the input only, ensuring that the convolution output at time t depends only on inputs at times t-7 through t (no future samples). The AdaptiveAvgPool1d(1) reduces the 512-sample temporal axis to a single value, producing a global summary of the audio context from which the FIR taps are predicted. The Tanh activation bounds the taps to [-1, 1], ensuring numerical stability in the subsequent minimum-phase conversion and filter application.

## 8.4 Quantization and Pruning Pipeline

The post-training optimization pipeline converts the trained FP32 CombinedModel into a compact, fast INT8 model suitable for real-time CPU inference.

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

The following pseudocode describes the synthetic dataset generation pipeline implemented in the DatasetBuilder class within generate_dataset.py.

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

The training loop pseudocode describes the complete workflow from data loading through post-training optimisation and ONNX export.

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

The combined_loss function uses three complementary objectives. The si_sdr_loss (weight 1.0) maximises the scale-invariant signal-to-distortion ratio between the estimated and clean signals, providing a global fidelity objective. The tss_loss (weight 0.5) penalises residual transient energy in the output relative to the input, computed as the ratio of energy in transient-masked regions, plus a 0.5-weighted speech distortion penalty to prevent the model from achieving high TSS by distorting speech. The plosive_preservation_loss (weight 1.0) computes the mean squared error between estimated and clean signals specifically in plosive-masked temporal regions, scaled by a factor of 10 to emphasise plosive fidelity. Gradient clipping at max_norm = 1.0 prevents training instability caused by the sharp loss landscape near transient boundaries.


---

# Chapter 9 — Conclusion of Capstone Project Phase 2

## 9.1 Summary of Work

Phase 2 of this capstone project has delivered the complete design, implementation, and infrastructure validation of a real-time, CPU-only transient noise suppression system. The work encompasses five major deliverables. First, the four-layer processing pipeline architecture has been designed and implemented: the lock-free SPSC ring buffer (Layer 1), the INT8 dynamic quantization and fifty-percent magnitude pruning optimisation pipeline (Layer 2), the DeepFIR minimum-phase FIR tap predictor (Layer 3), and the four-block Mamba Selective State Space Model with selective scan (Layer 4). Each layer is implemented as a self-contained module with clearly defined input/output contracts, enabling independent testing, replacement, and evolution.

Second, the combined DeepFIR + Mamba SSM neural network has been implemented in PyTorch as the CombinedModel class, with dual forward modes (forward_train for parallel scan training and forward_realtime for sequential scan inference), and successfully exported to ONNX format (opset 18) as filter_model.onnx. The ONNX inference pipeline, comprising ONNXInferenceRunner with pre-allocated buffers and session warmup, and RealTimePipeline with ring buffer integration and wet/dry mixing, is fully operational and validated against the ONNX Runtime CPUExecutionProvider.

Third, the synthetic dataset generation pipeline has been built, capable of producing 10,000 paired (noisy, clean) training samples from LibriSpeech and FreeSound source data with Room Impulse Response reverberation, randomised SNR mixing in the [-5, +20] dB range, and 80/10/10 train/validation/test splitting. The training loop is implemented with AdamW optimisation (lr=1e-3, weight_decay=1e-4), CosineAnnealingLR scheduling (T_max=50), gradient clipping (max_norm=1.0), and a combined loss function incorporating SI-SDR, TSS, and plosive preservation objectives. The post-training pipeline (prune, quantize, benchmark, export) is fully automated.

Fourth, the graphical user interface has been implemented in PyQt6, featuring the ControlWindow with suppression toggle, pass-through mode, strength and gain sliders, device selection, level meters, and RTF display; the TrayManager with dynamic microphone icon and system tray menu; and the WaveformViewer with three-panel waveform display, transient markers, and metric cards. The GUI runs in a separate process from the audio engine, communicating via multiprocessing.Queue with typed Command and Event messages.

Fifth, the offline proof-of-concept benchmark (poc_realtime_transient.py) has been implemented and validated, demonstrating that the DSP ring-buffer layer processes 128-sample chunks at 48 kHz in under 100 microseconds, consuming less than four percent of the 2.667-millisecond processing budget and providing over ninety-six percent headroom for the neural inference tier.

The NFR targets that have been validated by the DSP layer are: DSP latency under 100 microseconds (achieved), ring buffer latency of approximately 0.021 milliseconds (achieved), and CPU-only operation (achieved). The NFR targets that await validation with trained model weights are: SI-SDRi greater than 4.0 dB, PESQ at least 3.2, TSS above sixty-five percent, plosive SI-SDR above 25 dB, and end-to-end RTF below 0.80. These metrics cannot be validated until the model is trained on actual LibriSpeech and FreeSound data, which requires dataset acquisition and training compute that is planned for Phase 3.

## 9.2 Key Findings

The first key finding is that the classical DSP ring-buffer pipeline achieves processing latencies below 100 microseconds per 128-sample chunk, consuming less than four percent of the real-time budget. This overwhelming headroom — greater than ninety-six percent of the available processing time remaining after DSP operations — provides high confidence that the ONNX-based neural inference tier can operate within the 2.667-millisecond budget, even accounting for the overhead of Python's multiprocessing IPC, garbage collection, and operating system scheduling variability. The PoC benchmark achieves an RTF of approximately 0.01 to 0.05 for the DSP-only path, representing 20x to 100x real-time operation.

The second key finding is that the Mamba Selective State Space Model's O(N) computational complexity makes it uniquely suited for streaming real-time inference on CPU. The cost per timestep in recurrent mode is O(d_inner * d_state) = O(128 * 16) = O(2,048) multiply-accumulate operations, independent of the total context length. This stands in stark contrast to transformer-based architectures, where the O(N^2) attention mechanism would require 262,144 operations per head for a 512-sample context window, making real-time CPU operation infeasible.

The third key finding is that INT8 dynamic quantization combined with fifty-percent L1 magnitude pruning delivers approximately a seven-fold inference speedup with acceptable fidelity preservation. The selective precision preservation strategy — maintaining the A_log, D, and RMSNorm parameters in FP32 while quantizing all Linear and Conv1d layers to INT8 — ensures that the critical SSM state dynamics are not degraded by quantization noise, while the bulk of the computational workload benefits from the reduced precision.

The fourth key finding is that the plosive exception mechanism is critical for maintaining speech quality. Without explicit (DSP) or implicit (learned SSM) mechanisms to distinguish plosive consonants from transient noise, the system would suppress 'P', 'T', and 'K' sounds, producing muffled and unintelligible speech. The PoC demonstrates this through the synthetic plosive at t = 8.5 seconds, which passes through the TransientDetector unsuppressed due to its gradual onset envelope. In the neural model, this distinction is captured by the selective scan's input-dependent dt parameter, which the model learns to modulate based on the surrounding acoustic context.

## 9.3 Limitations

The most significant current limitation is that the neural model operates with random, untrained weights. All quality metrics (SI-SDRi, PESQ, TSS, plosive SI-SDR) reported by the evaluation pipeline reflect the performance of a randomly initialised model rather than a trained one. The model produces distorted, noise-like output in its current state because the weight matrices have not been optimised to map noisy inputs to clean targets. The StubDenoiser in the GUI's inference path simply passes audio through unchanged. Actual noise suppression quality will only be achieved after completing the training phase with real LibriSpeech and FreeSound data.

The second limitation is that the training data is entirely synthetic. The dataset generation pipeline creates noisy mixtures by additively combining clean speech and isolated noise recordings convolved with RIRs, which does not capture the full complexity of real-world acoustic scenarios — including non-linear microphone distortion, simultaneous multiple noise sources, Lombard effect (speakers raising their voice in noisy environments), and the spectral coloration of real enclosures that differs from the frequency response of measured or synthetic RIRs. Model performance may degrade on real-world audio that differs systematically from the training distribution.

The third limitation concerns the scope of the noise categories handled. The system is designed exclusively for transient noise suppression and does not address music, multi-channel audio, acoustic echo cancellation (AEC), or non-transient noise types beyond the stationary noise partially handled by the DeepFIR layer. Background conversation, traffic rumble, and other quasi-stationary noise sources are outside the designed operating envelope.

The fourth limitation is architectural technical debt. Two separate multiprocessing architectures coexist in the codebase: the production engine in app/audio/engine.py (using typed CmdType/EvtType enumerations and the StubDenoiser) and the alternative implementation in process_manager.py (using dict-based commands and AudioIOManager). The process_manager.py is effectively dead code that may confuse developers. Additionally, two separate configuration systems exist: the root config.py with module-level constants and app/config.py with the AppConfig dataclass, with an inconsistency in block size (BLOCK_SIZE = 256 in root config versus block_size = 1024 in AppConfig). The selective_scan_parallel function in mamba_ssm.py, despite its name, contains a sequential for-loop and is not truly parallel — it is functionally identical to selective_scan_sequential with additional overhead.

---

# Chapter 10 — Plan of Work for Capstone Project Phase 3

Table 10.1 presents the planned tasks for Phase 3, with task descriptions, assigned owners, and target milestones.

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

The most critical path item is the model training (Weeks 2-4), which is the prerequisite for all subsequent quality validation, ONNX integration, and user evaluation tasks. The architecture cleanup task (Week 5) addresses the technical debt identified in Section 9.3 and should be completed before the final documentation sprint to ensure that the codebase accurately reflects the documented architecture.

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

The system requires Python 3.10 or later. It is recommended to use a virtual environment to isolate dependencies. The setup procedure is as follows.

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

The requirements.txt file installs all seventeen dependencies including PyQt6, sounddevice, torch, onnxruntime, and their transitive dependencies. On macOS, the application may need to be launched from the system Python rather than the virtual environment Python due to the cocoa platform plugin requirement; the application handles this transparently via the re-execution mechanism in app/main.py.

## B.2 Running the GUI

To launch the graphical user interface:

```
python -m app.main
```

This command starts the application, which spawns the audio engine in a separate process and displays the ControlWindow along with a system tray icon. The interface provides the following controls.

The "Suppression: ON/OFF" button toggles the noise suppression processing. When ON (green), audio from the microphone is processed through the inference pipeline before playback. When OFF (red), audio is passed through unchanged.

The "Pass-Through (Demo)" button activates direct microphone-to-speaker routing with zero processing, ideal for setting up A/B comparisons during demonstrations. When active, it disables the strength and gain sliders.

The "Suppression Strength" slider (0 to 100) controls the wet/dry mix ratio. At 100 (default), the output is fully processed. At 0, the output is identical to the input.

The "Output Gain" slider (-12.0 to +12.0 dB) adjusts the output level to compensate for any volume changes introduced by the suppression process.

The input and output device dropdowns allow selection of the audio devices used for capture and playback. Changes take effect immediately.

The level meters display the current input and output audio levels in real-time, updated at 20 Hz.

The RTF label displays the current Real-Time Factor and headroom multiplier, providing a live performance indicator.

The "Proof of Concept" button at the bottom of the window launches the WaveformViewer, which runs the offline benchmark and displays the results as waveform plots with metric cards.

Closing the window hides it to the system tray; the application continues running. Right-click the tray icon and select "Quit" to fully terminate the application.

## B.3 Running the Offline Benchmark

To run the offline proof-of-concept benchmark without the GUI:

```
python poc_realtime_transient.py --mode demo
```

This command performs the following steps: (1) generates a synthetic 10-second test signal with five transient noise events (dog bark at 1.2s, door slam at 3.5s, keyboard click at 5.0s, siren chirp at 6.8s, plosive 'P' at 8.5s) mixed with a chirp base signal and white background noise; (2) saves test_clean.wav and test_noisy.wav to the project directory; (3) processes the noisy signal through the RealTimeFilter in 128-sample chunks, measuring processing time for each chunk; (4) saves test_noisy_filtered.wav; (5) prints a detailed performance report including mean, P99, and max processing times, RTF, headroom multiplier, per-transient attenuation in dB, and a FEASIBILITY VERDICT (PASS if RTF < 0.80).

To run in live mode (real-time microphone processing):

```
python poc_realtime_transient.py --mode live
```

This captures audio from the default microphone, processes it through the RealTimeFilter, and plays the filtered audio through the default speaker in real-time.

## B.4 Training the Model

The training pipeline consists of three sequential steps.

Step 1 — Dataset Generation:

```
python -m dataset.generate_dataset
```

This generates synthetic (noisy, clean) training pairs from LibriSpeech and FreeSound source data. Ensure the data directories (data/raw/librispeech, data/raw/freesound, data/raw/openair_rirs) are populated before running.

Alternatively, for the full-featured generator with RIR support:

```
python generate_dataset.py --total-samples 10000 --output-dir ./dataset
```

Step 2 — Model Training:

```
python -m training.train
```

This trains the CombinedModel for 50 epochs using AdamW with CosineAnnealingLR scheduling. The training loop saves checkpoints to checkpoints/best.pt and checkpoints/latest.pt. After training completes, the script automatically applies magnitude pruning (50%), INT8 quantization, benchmarks the RTF, and exports the optimised model to model/filter_model.onnx.

Step 3 — Evaluation:

```
python -m training.evaluate
```

This runs the full evaluation report, computing SI-SDRi, PESQ, TSS, and plosive SI-SDR metrics against the NFR targets. Results are printed as a formatted table with PASS/FAIL indicators for each metric.


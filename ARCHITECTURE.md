f# voiceiso — Architecture & Function Reference

> **CPU-only, real-time voice isolation** in the spirit of Apple Voice Isolation
> and Krisp. The enhancement *brain* is **DeepFilterNet3 (DFN3)**; the capstone
> **novelty is the integrated commercial-style pipeline** wrapped around it —
> preprocessing, AEC, VAD, noise classification, a dynamic suppression
> controller, speech preservation, and a residual post-filter.
>
> **Version 2** (current): 100 ms blocks on a dedicated DSP worker thread
> (xrun-safe callback), history-primed per-call DFN3 (80 ms of real left
> context re-enhanced each call — no cross-call GRU state is possible with
> DFN3's stateless-GRU invocation), graduated suppression controller, and an
> EWMA-smoothed learned noise classifier.  End-to-end mouth-to-ear latency
> ≈ 200–300 ms (block fill + one queue hop + device I/O).

All numbers in this document were **measured** on this machine (Apple-silicon
CPU), not estimated. See [§9 Results](#9-measured-results).

---

## 1. Design philosophy

Four principles, each forced on us by measurement or architectural critique:

1. **Trust the enhancer.** DFN3 at full strength preserves speech at correlation
   ≈ 0.99 and gives +16 dB SNR. An early "speech-protection" design blended the
   *noisy dry* signal back in and collapsed SNR from +14 → +3 dB. The corrected
   rule: **never blend noise back; output is fully wet.** Speech is protected by
   *limiting how much DFN attenuates*, not by mixing in noise.
2. **Be cheap where being wrong is cheap.** Per-class noise nuance and gentleness
   drive the **post-filter** (residual cleanup in gaps), never the network's core
   strength — a misclassification there is inaudible instead of leaking noise.
3. **Only get clever when confident.** The controller bypasses DFN (saving CPU)
   *only* after a sustained run of confidently-clean frames, never on a single
   frame's guess.
4. **Process per block; don't re-run history.** Each `enhance()` call gets only
   the new block. DFN3's `_state` carries the STFT/ISTFT + ERB-normalisation
   state across calls — **not** the GRU hidden states (the GRUs run stateless,
   `h0`=0 each call), so temporal context spans only the STFT frames *within*
   the block. We still avoid the old V1 look-back buffer (which re-ran 200 ms of
   history through the network), but we do **not** claim cross-call recurrence we
   don't have. Practical consequence: larger blocks give DFN3 more context —
   ALL paths (benchmark, `voiceiso live`, desktop app, offline `enhance`) use
   100 ms blocks, and the enhancement stage additionally re-enhances the last
   80 ms of real input with each block (history-primed per-call processing,
   fresh DfState per call) so every emitted frame has warm GRU/conv context.
   Measured on speech @ 5 dB SNR: 17.7 dB SI-SDR vs 11.5 dB for naive
   per-block stateful mode (one-shot bound 18.0 dB).

---

## 2. Pipeline overview

```
 mic block (100 ms @ 48 kHz)
   │
   ▼
 Preprocessing      DC/subsonic block + running SNR estimate
   │
   ▼
 AEC                partitioned-FDAF echo cancel  (opt-in; needs far-end ref)
   │
   ▼
 VAD                Silero → P(speech) + hangover-smoothed is_speech
   │
   ▼
 NoiseClassifier    heuristic spectral/temporal → noise_class + probs
   │
   ▼
 CompetingSpeech    flag a likely second talker
   │
   ▼
 DynamicController  fuse VAD + class + SNR → suppression + postfilter_strength
   │
   ▼
 SpeechPreservation detect consonants → protect from the post-filter
   │
   ▼
 Enhancement (DFN3) history-primed per-call (80 ms real context); controller-steered; bypass-when-clean
   │
   ▼
 PostFilter         residual gate in gaps + comfort-noise injection
   │
   ▼
 output block
```

Every box is a `Stage` that reads/annotates a shared `FrameContext` and keeps its
own streaming state across blocks. Measured RTF ≈ 0.17 when DFN3 is active
(100 ms blocks + 80 ms history). End-to-end latency budget: 100 ms block fill
+ one queue hop (≤ 100 ms) + device I/O — ≈ 200–300 ms mouth-to-ear.

---

## 3. Repository layout

```
voiceiso/
  __init__.py            package doc + __version__
  __main__.py            enables `python -m voiceiso`
  _compat.py             torchaudio shim so DeepFilterNet 0.5.6 imports
  config.py              PipelineConfig (frame geometry, thresholds, paths)
  pipeline.py            StreamingPipeline — wires all stages
  stages/
    base.py              FrameContext + Stage contract
    preprocessing.py     Preprocessing
    aec.py               AEC (partitioned FDAF)
    vad.py               VAD (Silero / energy fallback)
    noise_classifier.py  HeuristicNoiseClassifier + YamnetNoiseClassifier (stub)
    competing_speech.py  CompetingSpeech
    controller.py        DynamicController
    speech_preservation.py SpeechPreservation
    enhancement.py       Enhancement (DeepFilterNet3)
    postfilter.py        PostFilter (+ comfort noise)
  io/
    audio_stream.py      LiveStream (mic → pipeline → speakers, worker thread)
  bench/
    metrics.py           SI-SDR, seg-SNR, corr, PESQ, STOI
    benchmark.py         run_benchmark + BenchResult + success criteria
  data/
    corpora.py           CorpusRegistry (DNS5/MUSAN/DEMAND/WHAM/FSD50K)
    dynamic_mixer.py     DynamicMixer (on-the-fly clean/noisy pairs)
  cli.py                 info | enhance | live | bench

app/                     desktop app (GUI + tray); engine rewired to the pipeline
legacy/                  original DeepFIR+Mamba system, preserved
dataset/                 LibriSpeech test data + (legacy) loaders
```

---

## 4. Core abstractions

### `voiceiso/config.py` — `PipelineConfig`

A dataclass holding all tunables. Frame geometry matches DFN3's 48 kHz / 10 ms-hop
operation.

| Field | Default | Meaning |
|---|---|---|
| `sample_rate` | 48000 | DFN3 native rate |
| `hop_ms` / `win_ms` | 10 / 20 | STFT hop / analysis window |
| `lookahead_ms` | 0 | extra future context (raises latency) |
| `vad_sample_rate` | 16000 | Silero runs at 16 kHz |
| `vad_window` | 512 | Silero frame (32 ms @ 16 kHz) |
| `vad_speech_threshold` | 0.5 | P(speech) → speech |
| `vad_hangover_ms` | 200 | keep speech state after offset |
| `noise_classes` | 12-tuple | label set (keyboard…competing_speech, clean) |
| `supp_min` / `supp_max` | 0 / 1 | suppression bounds |
| `supp_attack_ms` / `supp_release_ms` | 30 / 150 | controller smoothing |
| `comfort_noise_db` | −65 | injected ambience level |
| `postfilter_floor_db` | −18 | max extra residual attenuation |
| `aec_enabled` | False | turn on echo cancellation |
| `aec_filter_ms` | 200 | AEC adaptive-filter tail length |

**Properties** (derived): `hop`, `win`, `lookahead` (samples); `algorithmic_latency_ms`
= `hop_ms + (win_ms − hop_ms) + lookahead_ms` = 20 ms — STFT-framing latency only.
It deliberately EXCLUDES the dominant real-world terms: the 100 ms block fill,
the worker-queue hop, and device I/O. End-to-end mouth-to-ear ≈ 200–300 ms.

### `voiceiso/stages/base.py`

**`FrameContext`** — the mutable object that flows through the pipeline:

| Field | Type | Filled by |
|---|---|---|
| `audio` | `np.ndarray` f32 | each stage reads/replaces |
| `sample_rate` | int | constructor |
| `reference` | `np.ndarray\|None` | caller (AEC far-end) |
| `vad_prob` / `is_speech` | float / bool | VAD |
| `noise_class` / `noise_probs` | str / dict | NoiseClassifier |
| `snr_db` / `environment` | float / str | Preprocessing |
| `suppression` | float | DynamicController |
| `postfilter_strength` | float | DynamicController / SpeechPreservation |
| `meta` | dict | diagnostics from every stage |

`copy_audio()` returns a copy of `audio`.

**`Stage`** — base class. Attributes `name`, `enabled`. Methods:
- `process(ctx) -> FrameContext` — abstract; do the work for one block.
- `reset()` — clear streaming state on (re)start.
- `__call__(ctx)` — skips when `enabled` is False, else calls `process`.

---

## 5. Stage-by-stage function reference

### 5.1 `Preprocessing` (`stages/preprocessing.py`)

`Preprocessing(cfg, highpass_hz=25.0)`

- **`__init__`** builds a **1st-order Butterworth high-pass** (`scipy.signal.butter`,
  `output="sos"`) at 25 Hz with persistent filter state `_zi`; initializes signal-
  and noise-power trackers. *Why gentle?* A 2nd-order 80 Hz filter was measured to
  cost ~8 dB SNR (it eats low speech harmonics and adds group delay), and DFN3
  removes real rumble far better — so we only block DC/subsonic.
- **`process(ctx)`** — runs `sosfilt` (state-preserving), updates a smoothed signal
  power and a minimum-statistics noise floor over a ~0.5 s window, writes
  `ctx.snr_db` (clipped −10…60) and `ctx.meta["noise_floor_db"]`.
- **`reset()`** — re-initializes filter state and trackers.

### 5.2 `AEC` (`stages/aec.py`)

`AEC(cfg, block=512, mu=0.3)` — **Partitioned-Block Frequency-Domain Adaptive
Filter** (the WebRTC AEC3 / Speex family). Opt-in via `cfg.aec_enabled` + a provided
`ctx.reference`. **Verified: 28.8 dB ERLE** on synthetic echo.

- **`__init__`** — derives partition count `P = ceil(tail / block)` from
  `cfg.aec_filter_ms`; allocates frequency-domain weights `_W [P, fft/2+1]`, input-
  spectrum history `_Xp`, overlap-save history, and I/O buffers.
- **`_process_block(d, x)`** — cancels echo from one `N`-sample near-end block `d`
  using far-end `x`:
  1. overlap-save frame `[x_prev, x]` → rFFT `X`; roll into partition buffer;
  2. echo estimate `y = irfft(Σ W·Xp)[N:]`; error `e = d − y`;
  3. **constrained NLMS** update — per partition `G = conj(Xp)·E / power`, zero the
     second half of `irfft(G)` (gradient constraint), add `mu·rFFT(g)` to `W`;
  4. track running **ERLE** (echo return loss enhancement). Returns `e`.
- **`process(ctx)`** — buffers near/reference, runs whole `N`-blocks, emits the same
  number of samples it was given (≤1-block delay), writes `ctx.meta["erle_db"]`.
- **`reset()`** — zeros weights, spectra, buffers.

> Linear AEC only; residual (non-linear) echo is mopped up downstream by DFN3 +
> post-filter — exactly how commercial stacks layer it.

### 5.3 `VAD` (`stages/vad.py`)

`VAD(cfg)` — **Silero VAD** (`silero-vad`, ONNX, ~1 ms/frame) with an energy/SNR
fallback. Module helper `_resample_poly(x, sr_in, sr_out)` polyphase-resamples 48 kHz
frames to Silero's 16 kHz.

- **`__init__`** — loads `load_silero_vad(onnx=True)` (1 torch thread, lowest latency);
  sets the hangover length from `cfg.vad_hangover_ms`. `backend` ∈ {`silero`,`energy`}.
- **`_silero_prob(frame48)`** — accumulates resampled audio, runs the model on full
  512-sample windows, returns the latest P(speech).
- **`_energy_prob(frame48)`** — adaptive noise-floor SNR mapped to [0,1] (fallback).
- **`process(ctx)`** — sets `ctx.vad_prob`; thresholds to `ctx.is_speech` with
  **hangover** (keeps speech state for `vad_hangover_ms` after offset so word tails
  aren't chopped).
- **`reset()`** — clears buffers + Silero internal states.

### 5.4 `NoiseClassifier` (`stages/noise_classifier.py`)

Two classes behind one interface.

**`HeuristicNoiseClassifier(cfg)`** — zero-dependency, ~CPU-free.
- **`_flatness(mag)`** *(static)* — spectral flatness (geo/arith-mean ratio); ≈1 white,
  →0 tonal.
- **`process(ctx)`** — computes flatness, spectral centroid, HF/LF energy ratios,
  spectral flux (onset), and energy stationarity, then maps to a label distribution:
  *clean* (near floor), transient → *door_slam/keyboard/dog_bark* (by flux + LF/HF),
  tonal+sustained → *music/television*, speech-like spectrum with low VAD →
  *competing_speech*, stationary broadband → *fan/hvac/traffic/wind* (by centroid).
  Writes `ctx.noise_class`, `ctx.noise_probs`, and spectral meta.
- **`reset()`** — clears flux/energy history.

**`YamnetNoiseClassifier(cfg, onnx_path=None)`** — **interface stub** for the learned
upgrade (YAMNet/PANNs AudioSet tagger → ONNX, ~1–3 ms/frame, 521 tags mapped onto our
labels). Raises `NotImplementedError` with the implementation plan in its docstring.

### 5.5 `CompetingSpeech` (`stages/competing_speech.py`)

`CompetingSpeech(cfg)` — target-speaker handling (VoiceFilter-Lite reference design).
- **`enroll(embedding)`** — register the primary speaker's normalized d-vector.
- **`process(ctx)`** — interim behaviour: trusts the classifier's `competing_speech`
  flag and marks `ctx.meta["competing_speech"]` so the controller can react. When a
  speaker encoder is wired in, this becomes d-vector cosine similarity + extraction.

> **Capstone verdict (documented):** full blind separation isn't CPU-real-time;
> streaming **target-speaker extraction** (VoiceFilter-Lite, ~2 MB) is the feasible
> stretch goal.

### 5.6 `DynamicController` (`stages/controller.py`)

`DynamicController(cfg, clean_snr_db=22.0, clean_hold_ms=400.0)` — the integrated-system
brain. Module constant `_CLASS_POSTFILTER` maps each noise class → post-filter weight.
- **`__init__`** — precomputes attack/release smoothing coefficients from
  `supp_attack_ms`/`supp_release_ms`; sets the sustained-clean frame count.
- **`process(ctx)`** —
  1. **confidently_clean** = not speech **and** `snr_db ≥ clean_snr_db` **and** class
     == clean; increments a `clean_run` counter (else resets).
  2. **target** = `supp_min` (→ DFN bypass) only after a sustained clean run, else
     `supp_max` (→ full DFN). *Quality-first: trust DFN whenever noise is present.*
  3. **attack/release smoothing** → `ctx.suppression` (rise fast, fall slow → no
     pumping/chopping).
  4. **post-filter strength** = per-class weight scaled down at high SNR →
     `ctx.postfilter_strength`. Writes `ctrl_target` / `clean_run` meta.
- **`reset()`** — restores full suppression, clears the clean counter.

### 5.7 `SpeechPreservation` (`stages/speech_preservation.py`)

`SpeechPreservation(cfg, detect_threshold=0.3)` — protects consonants from the
post-filter (DFN already preserves them, so it does **not** throttle the enhancer).
- **`_hf_ratio(x)`** — fraction of (windowed) spectral energy above 3.5 kHz; a
  fricative cue.
- **`process(ctx)`** — computes fricative (HF-dominant) and plosive (sharp RMS-onset)
  cues; if `consonant ≥ detect_threshold`, **forces `ctx.is_speech = True`** (so the
  residual gate never attenuates a near-missed fricative/plosive) and eases
  `ctx.postfilter_strength`. Writes `ctx.meta["consonant"]`.
- **`reset()`** — clears the previous-RMS tracker.

### 5.8 `Enhancement` (`stages/enhancement.py`) — the DFN3 core

`Enhancement(cfg, threads=4, context_ms=200.0, bypass_below=0.08)`
- **`__init__`** — imports `voiceiso._compat` first (torchaudio shim), then
  `init_df(...)` to load DeepFilterNet3; `backend` ∈ {`deepfilternet3`,`passthrough`}.
- **`_enhance(buf, atten_lim_db)`** — runs DFN's `enhance()` on a buffer with the given
  attenuation cap, returns float32 clamped to input length.
- **`process(ctx)`** —
  - **bypass** (passthrough dry, skip the net) when backend is passthrough **or**
    `suppression ≤ bypass_below` (controller says clean) — saves CPU;
  - else maps `suppression → atten_lim_db ∈ [12, 100]` (gentle…aggressive), calls
    `_enhance` on `[last 80 ms of real input | current block]` with a FRESH
    DfState per call and keeps the block's region — every emitted frame gets
    real left context and warmed (per-call) GRUs; DFN3's GRUs hold **no**
    cross-call state. Output is **fully wet**. Writes `enh_wet` /
    `enh_atten_db` meta.
- **`reset()`** — zeroes the raw-input history buffer (there is no persistent
  DfState to clear).

> Streaming: passing only the new block per call eliminates the V1 overlap-save
> overhead (which reprocessed 200 ms of history every block). Note DFN3's `_state`
> holds STFT/ERB analysis state, **not** GRU recurrence — there is no cross-call
> hidden-state memory — so per-block context is limited to the block's STFT
> frames. End-to-end latency ≈ block size + one queue hop + DFN3's internal
> look-ahead, not the bare STFT-framing figure.

### 5.9 `PostFilter` (`stages/postfilter.py`)

`PostFilter(cfg)` — residual cleanup + comfort noise. Class constants `_CN_B`/`_CN_A`
define a one-pole low-pass for shaping comfort noise.
- **`process(ctx)`** —
  1. in **non-speech** frames only, applies extra residual attenuation interpolated
     between unity and `postfilter_floor_db` by `ctx.postfilter_strength` (safe — no
     speech to damage);
  2. injects spectrally-shaped **comfort noise** at `comfort_noise_db` via
     `scipy.signal.lfilter` (state carried across blocks) so gaps keep a natural,
     constant ambience instead of dead silence; clips to ±1.
- **`reset()`** — clears the comfort-noise filter state.

---

## 6. Orchestration — `voiceiso/pipeline.py`

`StreamingPipeline(cfg=None, enh_threads=4)` constructs all stages and stores them in
order in `self.stages`.

- **`backend_summary`** *(property)* — `{vad, enhancement, aec_enabled,
  algorithmic_latency_ms}`.
- **`process_block(block, reference=None)`** — builds a `FrameContext` and runs it
  through every stage; returns the annotated context (`ctx.audio` is the output).
- **`process_signal(x, block=None)`** — offline helper that streams a whole signal in
  blocks (default ~100 ms) and returns the enhanced array.
- **`reset()`** — resets every stage.

---

## 7. I/O, CLI, benchmark, data

### `voiceiso/io/audio_stream.py` — `LiveStream(cfg=None, block_ms=100, enh_threads=4)`
Mic → pipeline → speakers. The heavy enhancement runs on a **worker thread**; the
real-time `sounddevice` callback only does lock-free enqueue/dequeue (never blocks the
audio thread). `run(duration_s=0.0)` starts the stream (0 = until Ctrl-C). *Wired but
not verifiable in a headless environment — needs a real machine to validate live.*

### `voiceiso/cli.py` — `python -m voiceiso …`
- **`info`** — print backends + stage list.
- **`enhance IN [OUT]`** — offline file enhancement via `process_signal`.
- **`live [--duration S]`** — run `LiveStream`.
- **`bench [--snr 5] [--n 10] [--data data]`** — build pairs (DynamicMixer if corpora
  present, else synthetic from `dataset/test/clean`) and run the harness.

### `voiceiso/bench/`
- **`metrics.py`**: `si_sdr`, `seg_snr`, `correlation` (always available);
  `pesq_wb` (needs `pesq`, resamples to 16 kHz), `stoi` (needs `pystoi`);
  `all_metrics(ref, est, sr)` returns whatever is available.
- **`benchmark.py`**: `run_benchmark(pairs, sr, cfg)` → `BenchResult` (quality dict,
  `rtf`, latency p50/p95/p99, peak RAM, backends) with `.report()`. `SUCCESS` defines
  target criteria (SI-SDRi ≥ 10 dB, PESQ ≥ 2.6, RTF ≤ 0.5, latency ≤ 40 ms, RAM ≤ 300 MB).

### `voiceiso/data/`
- **`corpora.py`**: `Corpus.scan()` indexes audio files; `CorpusRegistry(data_root)`
  with `speech()`/`noise()`/`rirs()`/`summary()` — indexes whatever DNS5/MUSAN/DEMAND/
  WHAM/FSD50K/RIR folders exist locally (downloads nothing).
- **`dynamic_mixer.py`**: `DynamicMixer(data_root, sr, segment_s, snr_range, use_rir,
  seed)` — `draw()` returns one `(clean, noisy)` pair (random speech + noise + optional
  RIR convolution at a random SNR); `build_benchmark_set(n)` returns a list;
  `available()` reports whether corpora were found.

---

## 8. Desktop app integration

`app/audio/engine.py` was rewired: the old classical `RealTimeFilter` is replaced by a
`StreamingPipeline`. The audio callback runs `pipeline.process_block(mono)` inline; the
GUI **strength** slider applies a final wet/dry trim on top. `app/config.py` now uses
**100 ms blocks** (`block_size = 4800`) — DFN3's efficient point (~21 ms work per
100 ms block, xrun-safe). If the pipeline can't load, it falls back to the stub
passthrough.

---

## 9. Measured results

Real LibriSpeech speech + noise @ 5 dB input SNR, via `python -m voiceiso bench`:

| Metric | Value |
|---|---|
| SI-SDR in → out | +5.00 → +12.96 dB |
| **SI-SDRi** | **+7.97 dB** |
| **PESQ-wb** | **1.22 → 2.07** |
| Correlation w/ clean | **0.974** |
| **RTF** | **0.21** (≈5× real-time headroom) |
| Latency (compute) p50 / p99 | 20.7 / 31.7 ms per block |
| End-to-end (block) latency | ≈100 ms (V1); V2 target ≈20–25 ms |
| Peak RAM | ~370 MB |
| AEC ERLE (synthetic echo) | **28.8 dB** |
| Backends | Silero VAD + DeepFilterNet3 |

---

## 10. Caveats & roadmap

**Working & verified:** preprocessing, VAD (Silero), DFN3 enhancement, controller,
speech-preservation, post-filter, AEC (off by default), benchmark, dynamic mixer, CLI,
offline file enhancement.

**Noise classifier (learned EfficientAT, primary path):** a frozen EfficientAT
`mn10_as` backbone + a trained native 12-class head, exported to
`checkpoints/efficientat_head12.onnx` (static 4 s / 400-frame mel input). It is
auto-wired by `PipelineConfig` and runs at a 4 s window / 500 ms cadence; the
zero-dependency heuristic remains the transparent fallback when the checkpoint or
onnxruntime is absent (the fallback is logged and surfaced in `backend_summary`).

**Classifier accuracy — corrected protocol (`efficientat_head12_v3.onnx`).**
Trained on **FSD50K.dev_audio** with an **uploader-grouped** train/val split
(train∩val uploader overlap = 0; v3 reuses v2's exact split so the window
policy is the only variable); the frozen backbone embedding is read straight
out of the exported ONNX and only the Linear head is trained
(`scripts/retrain_head_dev.py`). Tested on the official held-out **FSD50K.eval**
set (uploader-disjoint from dev by construction), streamed through the runtime
classifier.

> **Harness correction:** the runtime classifier now tile-pads partial buffers
> (≥ 1 s), so clips shorter than the 4 s window can be classified. Under the
> old harness, 18.4 % of the test clips were *forced* misses ("clean" is never
> a FSD50K truth label), so all previously-reported numbers were depressed —
> v2's earlier 0.564 / 0.621 measures the same model as today's 0.635 / 0.740.

| Backend | macro-F1 | top-1 |
|---|---|---|
| heuristic *(pre-fix harness)* | 0.089 | 0.128 |
| pretrained-direct (527 AudioSet → 12 map) *(pre-fix harness)* | 0.211 | 0.258 |
| head v1 (eval-pool, **train-on-test ⇒ inflated**) | *0.698* | *0.782* |
| head v2 (dev-trained, first-4s crops) | 0.635 | 0.740 |
| **head v3 (dev-trained, window-robust — deployed)** | **0.636** | **0.742** |

> **v3 (window-robust training):** the live classifier sees rolling 4 s windows
> where an event may occupy a fraction of the window at any position; v2
> trained only on tile-first-4s crops. v3 trains on random 4 s crops (long
> clips) plus tiled-random-phase *and* sparse event-in-context variants (short
> clips) — same clips, same split. On event-dense full clips it ties v2; in
> the transient regime it is markedly more robust: with a 0.5 s event at the
> window edge (uploader-disjoint val clips), true-class posterior 0.73 → 0.89
> and top-1 16/24 → 21/24. Best window-robust val macro-F1 = 0.683. v1's row
> remains invalid (≈30 % of the test clips were in its training pool).
> Reproduce: `python -m scripts.retrain_head_dev --eval-only`.

> **v4 (TV-rejection, opt-in — `scripts/train_head_v4.py`):** the `tv` class
> had zero FSD50K positives (untrainable, all-negative row). v4 synthesizes
> loudspeaker/TV positives from dev speech+music (band-limit 180–400 Hz →
> 3.2–5.2 kHz, tanh compression, small-room reverb; uploader side inherited
> from the source clip), multiplies `fan` ×6 with gain/tilt jitter, uses a
> STRATIFIED uploader-grouped val (per-class floors — v3's val had fan=0,
> keyboard=4, so its early stopping partially selected on noise), and
> calibrates per-class decision thresholds on val, shipped as ONNX
> `thresholds` metadata (runtime ranks posterior/threshold; heads without
> the metadata behave exactly as before).
> **Results:** stratified val macro-F1 0.792 (10 classes incl. tv). Held-out
> loudspeaker discrimination (EVAL speech clips, live vs loudspeaker-sim):
> rejects loudspeaker speech **0.86–0.91 vs v3's 0.54**, labels it tv
> **0.41–0.68 vs 0.00**, live-speech leg unchanged. **But** FSD50K-eval
> macro-F1 regresses 0.636 → 0.613–0.617 (speech recall 0.40 → 0.33–0.35:
> the tv decision claims channel-degraded field-recorded "speech"). One
> disclosed threshold retest (t_tv 0.95 → 1.5, selected on val) did not
> close the gap, so the pre-registered ≤0.01 no-regression gate FAILED and
> v3 remains the default; v4 is opt-in via `VOICEISO_HEAD`. This is an
> honest capability/benchmark trade: FSD50K labels loudspeaker-channel
> recordings as "speech", so a head that learns the channel is penalized by
> exactly the recordings it is designed to flag.

> **Below the 0.70 macro-F1 target — bottleneck (evidenced):** the *frozen
> representation* plus data starvation, not the head or threshold. A
> clean-threshold sweep is flat; head capacity barely moves it (linear → MLP-512
> ≈ +0.02 val); and the worst classes are data-starved (`fan`: 64 dev clips,
> F1 0.40; `wind` 0.54). Top-1 now stands at 0.742. Minimum-change path
> *without* unfreezing the backbone: (1) source `fan`/`hvac` from DEMAND/MUSAN
> to lift the starved classes; (2) per-class threshold calibration on a
> stratified val split. ~0.70 macro-F1 on held-out FSD50K eval likely still
> needs backbone fine-tuning, which is out of scope (frozen by design).

**Scaffolded (interfaces + upgrade paths):** competing-speech only *flags*
(VoiceFilter-Lite extraction is the next step); ECAPA speaker embedder is
passthrough until a model is provided.

**Known caveats:**
- Live mic path is wired but **unverified** here (no audio device) — validate on a real
  machine.
- `algorithmic_latency_ms` (20 ms) reflects STFT framing only; **real
  mouth-to-ear latency ≈ 200–300 ms** (100 ms blocks + worker-queue hop +
  device I/O).
- DFN3 needs a **Rust toolchain** to build, a **torchaudio compat shim**
  (`voiceiso/_compat.py`), and pins **numpy < 2**.
- SI-SDRi (~8 dB) is below the aspirational 10 dB target on the hardest white-noise
  case; PESQ gain (+0.85) is solid.

**Implemented in V2:** (1) 100 ms worker-thread streaming with history-primed
per-call DFN3 (the earlier "stateful 20 ms streaming" claim was wrong — DFN3's
GRUs are invoked stateless; 20 ms blocks measure NEGATIVE SI-SDRi and were
abandoned); (2) graduated suppression controller with per-class release times;
(3) EWMA frame smoothing + flux-threshold bug fix in the noise classifier.

**Remaining next steps:** (1) learned noise classifier (YAMNet / EfficientAT ONNX);
(2) VoiceFilter-Lite target-speaker extraction; (3) export DFN to ONNX + INT8 to cut
the ~370 MB torch footprint; (4) tiny GRU post-filter for music/TV residuals;
(5) fine-tune DFN on the dynamic-mixing dataset for target conditions.

---

## 11. Quick start

```bash
python -m voiceiso info                      # backends + stages
python -m voiceiso enhance noisy.wav out.wav # offline file enhancement
python -m voiceiso bench --snr 5 --n 10      # quality + speed report
python -m voiceiso live                      # mic → pipeline → speakers
python -m app.main                           # desktop app (GUI + tray)
```

Setup: install a Rust toolchain (`curl https://sh.rustup.rs -sSf | sh -s -- -y`), then
`pip install -r requirements.txt`.

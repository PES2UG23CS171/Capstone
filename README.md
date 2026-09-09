# voiceiso — CPU-only real-time voice isolation

[![CI](https://github.com/PES2UG23CS171/Capstone/actions/workflows/ci.yml/badge.svg)](https://github.com/PES2UG23CS171/Capstone/actions/workflows/ci.yml)

A commercial-style voice-isolation pipeline (in the spirit of Krisp / Apple Voice
Isolation) built around **DeepFilterNet3** on CPU, wrapped with preprocessing,
AEC, Silero VAD, a **learned EfficientAT 12-class noise classifier**, a dynamic
suppression controller, speech preservation, and a residual post-filter.

See [Docs/ARCHITECTURE.md](Docs/ARCHITECTURE.md) for the full design and stage reference.

## Quick start (clean clone)

```bash
# 1. Rust toolchain — required to build DeepFilterNet's native extension
curl https://sh.rustup.rs -sSf | sh -s -- -y

# 2. Python deps (Python 3.10–3.12; numpy is pinned <2 for deepfilternet)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # includes onnxruntime (needed by the classifier)

# 3. Sanity-check the backends actually loaded (NOT heuristic/passthrough)
python -m voiceiso info
```

`info` prints `backend_summary`. For a correct setup you should see
`'enhancement': 'deepfilternet3'` and `'noise_classifier': 'efficientat'`. If you
see `'heuristic'` / `'passthrough'`, the model checkpoints or `onnxruntime` are
missing — the pipeline logs a **WARNING** and reports the reason in
`noise_classifier_fallback` (it never fails silently).

### Checkpoints

The learned classifier needs `checkpoints/efficientat_head12.onnx`
(+ `.onnx.data`). `PipelineConfig` auto-resolves it relative to the repo root, so
no path needs to be set. DeepFilterNet3's own weights are downloaded by `init_df`
on first run. DNSMOS is optional: `python -m scripts.fetch_dnsmos` pulls
`sig_bak_ovr.onnx` into `checkpoints/` (auto-wired thereafter).

## Usage

```bash
python -m voiceiso info                         # backends + stage list
python -m voiceiso enhance noisy.wav out.wav    # offline file enhancement
python -m voiceiso bench --n 10                 # quality + speed report
python -m voiceiso live                         # mic → pipeline → speakers
python -m app.main                              # desktop app (GUI + tray)
```

The live paths run DFN3 at **100 ms blocks on a worker thread** (DFN3's efficient
design point; it has no cross-call GRU memory, so 20 ms blocks measurably degrade
quality — see Docs/ARCHITECTURE.md §1.4). The audio callback never runs the network,
so it is xrun-safe. All models are warmed up before the stream starts, and the
callback drains to the freshest processed block, so mouth-to-ear latency stays at
the ~200–300 ms design point instead of ratcheting after load spikes.

### Call mode — clean audio INTO Google Meet / Zoom / Teams

VoiceISO sits between the physical microphone and the conferencing app, so
the person on the other side hears the denoised stream (the Krisp topology).
macOS needs a free virtual audio driver as the hand-off point — one-time
setup:

```bash
brew install blackhole-2ch
```

Then (either path):

- **CLI:** `python -m voiceiso live --output BlackHole`
- **App:** `./run.sh`, set *Output device* → `BlackHole 2ch`

…and in **Google Meet → Settings → Audio → Microphone**, select
**BlackHole 2ch** (Speakers must stay on real headphones/speakers, never
BlackHole). The far side now receives the processed audio. Your real mic
stays selected inside VoiceISO; Meet's own **Noise cancellation** and
**Studio sound** (its AI mic enhancer) must both be turned OFF to demo ours —
with either on, the far side can't tell whose cleaning it is hearing. `python -m voiceiso devices` lists devices and flags
usable call-mode sinks.

> **Use a Chromium browser for the call (Chrome / Brave / Edge) — not
> Safari.** Safari routes call audio through Apple's voice-processing unit,
> which fails silently with virtual devices: with BlackHole selected as the
> microphone, Safari's call audio dies in *both* directions, while the same
> call works with the built-in mic. Chromium browsers use their own audio
> stack and work with BlackHole as expected. (Diagnosed live 2026-09-08:
> VoiceISO's output was verified at normal speech level inside BlackHole and
> Safari had BlackHole open as its input — the break was inside Safari.)
> Also quit other apps holding the microphone (e.g. GarageBand) before a demo.

Notes: you will not hear yourself locally in this mode (normal for calls —
the far end hears the clean stream); mouth-to-ear adds ~200–300 ms, fine for
a demo call. To also monitor locally, create a Multi-Output Device in
*Audio MIDI Setup* combining BlackHole + your headphones and use that as
VoiceISO's output.

### Live-demo checklist

- Launch with `./run.sh` (pins the venv — running the system Python silently
  drops DFN3 to passthrough; the startup log must say
  `'enhancement': 'deepfilternet3'`).
- **Use headphones** (or a headset). AEC is off by default and mic→speaker
  loopback echoes the presenter's voice back — DFN3 *preserves* speech by design.
- **Use a wired 48 kHz-capable microphone.** Bluetooth HFP mics run at 8/16 kHz
  and can fail to open (the app streams fixed 48 kHz).
- Connect all audio devices **before** launching the app (the device list is a
  startup snapshot). macOS mic mode: **Standard** (don't stack Apple's own
  Voice Isolation in front).
- **Strength slider at 100 %** — the wet/dry mix caps total suppression at
  −20 dB when the slider sits at 90 %.
- Give it ~2 s after start before judging (first classifier label lands ~1 s).

### Demo choreography (plays to measured strengths)

1. **Fan / steady noise while talking** — headline: −40 dB fan kill, speech
   preserved.
2. **Claps & snaps in speech pauses** — the transient kick crushes every
   impulse (−48…−53 dB, no decay across repeats). Clap *between* phrases —
   a clap overlapping your own words shares the speech bands DFN3 must
   protect and only attenuates 2–4 dB (known architectural limit; say so).
3. **Keyboard while talking** — strongest classifier class (F1 0.84).
4. **Phone playing a voice next to you** *(optional segment)* — relaunch with
   `VOICEISO_HEAD=checkpoints/efficientat_head12_v4.onnx ./run.sh` to enable
   the TV-rejection head (0.86 loudspeaker rejection vs 0.54 default; the
   trade-off is documented in Docs/ARCHITECTURE.md §10).
5. To let the audience hear the processed stream: record with
   `python -m scripts.live_probe` and play `diag_capture/probe_out.wav`, or
   route the app's output into a call. The room itself always carries the
   raw sound — suppression exists only in the stream.

## Benchmarks (reproducible)

```bash
# Enhancement quality/speed at the live 100 ms operating point:
python -m voiceiso bench --n 20 --snr 5           # add --block-ms 20 for the low-latency point

# Classifier: retrain on dev (uploader-grouped) + evaluate on held-out eval:
python -m scripts.retrain_head_dev               # train v2 + full eval report
python -m scripts.retrain_head_dev --eval-only   # just the eval table

# Real-noise condition matrix (FSD50K eval clips) → Docs/REALWORLD_RESULTS.md:
python -m scripts.realworld_matrix

# Ground truth on YOUR OWN clean / noisy recording pairs (any format afconvert reads):
python -m scripts.ground_truth_eval prep --noisy "Personal Test/audio demo.m4a" \
    --clean "Personal Test/pes jargon.m4a" --tag jargon
python -m scripts.ground_truth_eval run --tag jargon --name defaults
python -m scripts.ground_truth_eval run --tag jargon --name previous \
    --cfg dfn_history_ms=80 --cfg dfn_tail_ms=0 --cfg artifact_rollback_enabled=True
```

> Without real noise corpora under `--data`, `bench` falls back to a **synthetic
> white-noise** condition and says so loudly — those numbers are not comparable to
> real-corpora results.

> **How intrusive metrics are scored.** The pipeline's output is delayed by a
> constant, known 975 samples (20 ms deferred DFN3 tail + 15-sample FIR group
> delay; `StreamingPipeline.output_delay_samples`). SI-SDR, seg-SNR,
> correlation, PESQ and STOI are computed **only on the valid overlap** of
> reference and output at that delay — never against zero padding — and the
> cross-correlation estimate cross-checks it, **rejecting** a pair whose
> alignment is impossible or ambiguous instead of scoring it. DNSMOS is
> non-intrusive and uses the full output. The report states three criteria
> separately: the **Phase-2 project objective** (SI-SDRi ≥ 4.0 dB @ 5 dB SNR —
> the release-gating one), the **aspirational Krisp-parity targets** from the
> June redesign (SI-SDRi ≥ 10 dB etc. — informational), and the **real-time
> budget** (RTF ≤ 0.5). It never prints "PASS".

### Headline results

Synthetic +5 dB white-noise condition (`python -m voiceiso bench --n 20 --snr 5`),
n=20, 100 ms live operating point, intrusive metrics scored on the valid
reference/output overlap at the known 975-sample output delay (3.98 s of each
4 s pair; no padding scored) — measured 2026-09-08 with the current defaults
(160 ms priming, 20 ms deferred tail, rollback off, speech cap 0.20,
speech-energy floor 9 dB): **SI-SDRi +4.13 dB**, **PESQ 1.24 → 1.51**,
**STOI 0.82**, DNSMOS **SIG 3.15 / BAK 3.70 / OVRL 2.77** (non-intrusive, full
output); RTF 0.22 with the DNSMOS/PESQ harness in-process, per-block p50 / p95
/ p99 = 22.0 / 35.8 / 39.0 ms against the 100 ms budget, algorithmic latency
40 ms (20 ms STFT framing + the 20 ms deferred tail), peak RSS ~625 MB
(harness-inclusive). Repeated runs produce bit-identical output (sha256 prefix
`e5aa0ddcbfd26aca`). (Cap alone, floor off: SI-SDRi 4.12, PESQ 1.55, STOI
0.83, BAK 3.72 — the floor costs 0.03 PESQ / 0.015 STOI on white noise.)

Like-for-like "before" (previous defaults — 80 ms priming, no tail, rollback
on — identical clips and identical scoring): SI-SDRi +3.84 dB, PESQ 1.34,
STOI 0.79, DNSMOS SIG 2.92 / BAK 2.64 / OVRL 2.08. The new defaults improved
all measured metrics on the evaluated sets, for ≈ +15 % compute (RTF 0.19 →
0.23 under identical conditions) and +20 ms of latency. The four latency
figures in this project are different things: the **20 ms deferred tail**
(the only lookahead), the **40 ms algorithmic latency** (tail + STFT framing),
the **100 ms frame accumulation** of the live paths, and the **≈ 200–300 ms
mouth-to-ear on a call** (block fill + queue hop + device I/O).

Criteria, stated separately: the **Phase-2 project objective** (SI-SDRi
≥ 4.0 dB @ 5 dB SNR) is met on this synthetic condition (+4.05; the previous
defaults were below it at +3.84) and on five of six real-noise conditions
(below; not TV); the **aspirational Krisp-parity target** (SI-SDRi ≥ 10 dB,
from the June redesign) is not met and never has been; the **real-time
budget** (RTF ≤ 0.5) is met. Synthetic white-noise and real-noise numbers are
different conditions and are never quoted against each other.

History of the headline, because it matters: the July sprint reported
**SI-SDRi +5.0 dB** (PESQ 1.36, OVRL 2.24, BAK 3.00), and that run reproduces
exactly on today's clips. The next DSP commit caged the transient kick after
real listening revealed onset distortion; the pre-cage kick had been firing on
every speech onset over white noise — forcing DFN3 to full strength on exactly
this benchmark — and the cage cost 1.15 dB of SI-SDRi that was never
re-measured. Re-tested on ground-truth recordings: an eager kick recovers
+5.5 dB here but costs clean-speech PESQ 3.96 → 3.84, so the cage stays.
Two personal recordings (a male and a female voice under plates/claps and car
noise) served as **regression evidence** for these changes — both improved on
all measured metrics except one flat SI-SDRi (0.13 → 0.12 on the second
voice); they are not a representative evaluation set.

**Female voices (fixed 2026-09-08).** Live testing with women speaking
reported words clipped mid-way and a thin, distorted timbre while a male
voice sounded good. Reproduced offline on a clean female recording at the
same level: PESQ 3.00 vs 3.99 for the male voice, with 100–300 ms holes of
30+ dB while the VAD read 1.0. Cause: per-block DFN3 with 160 ms of context
sometimes judges creaky / low-centroid phrase endings (frequent in female
speech) as low-frequency noise — one-shot DFN3 keeps every one of them — and
the controller's "speech cap" was a constant 0.92 (a 93 dB attenuation cap)
that let each misjudgement fall 30+ dB. Fix: the cap is now the config field
meant for it, `supp_speech_floor` = 0.20 (≈ 30 dB while the VAD is
confident). Measured on identical inputs: female clean PESQ 3.00 → 3.10
(worst hole −34 → −28 dB), male 3.99 → 4.00; standard benchmark SI-SDRi
4.05 → 4.12, PESQ 1.45 → 1.55, STOI 0.830 → 0.832; the two noisy personal
recordings −0.045 / −0.02 PESQ with DNSMOS SIG/OVRL up; no compute cost. The
value was set on the *noisy* recordings: a tighter cap helps a clean female
voice more (0.10 → 3.24) but re-admits noise under speech when the noise is
loud (0.10 → −0.25 PESQ on the plate recording). A speech-energy floor
(`speech_floor_db` = 9, **on by default since 2026-09-09**) repairs the
residual holes: on the clean female voice it is purely positive (holes
−18 dB, PESQ 3.10 → 3.17), and on the noisy car recording it was the
listener-preferred version. Its trade: when speech and a loud burst overlap,
the only thing it can blend back is the dry block with the burst, so on the
plate recording speech gets clearer (DNSMOS SIG +0.06) while the noise under
it is more audible (BAK −0.10, PESQ −0.20). Honest status: the female voice is
improved, not equalised (3.17 vs 4.00 on clean speech); the rest of the gap is
DFN3's own behaviour on this voice (one-shot 3.85 vs 4.26) plus the streaming
context (see ARCHITECTURE.md).

Settable without touching code (`VOICEISO_CFG="key=value,..."`):

```bash
VOICEISO_CFG="dfn_history_ms=240" python -m app.main   # more context: female 3.29, +9 % CPU
VOICEISO_CFG="speech_floor_db=0" python -m app.main    # disable the repair in rooms with speech-level crashes
```

Real-condition behaviour on held-out FSD50K eval clips
([Docs/REALWORLD_RESULTS.md](Docs/REALWORLD_RESULTS.md), regenerated 2026-09-08
with the current defaults): silence-regime attenuation **fan +42 dB, clap
+43 dB, traffic +42 dB, competing speech +31 dB, keyboard +28 dB, TV +14 dB**;
during speech at 0 dB SNR, SI-SDRi from **+5.2 (TV) to +10.9 (keyboard)**.
(Versus the sprint matrix: speech-regime numbers rose 0.2–2.1 dB except
traffic, and fan/traffic/competing-speech silence attenuation rose 3–8 dB; TV
silence attenuation fell 19 → 14 dB — DFN3 with fuller context treats
loudspeaker speech more like speech — and clap silence 45 → 43 dB. The
speech-energy floor's known trade shows here too: traffic during speech
6.9 → 6.0 dB at 0 dB SNR and 4.9 → 4.6 at 5 dB, the burst-under-speech case.)

### Classifier (`efficientat_head12_v3.onnx`, corrected protocol + window-robust)

Trained on **FSD50K.dev_audio** (uploader-grouped train/val, same split as v2),
tested on the held-out **FSD50K.eval** set, streamed through the runtime
classifier (which now tile-pads partial buffers — previously 18.4 % of test
clips were shorter than the 4 s window and physically could not be classified,
depressing every backend's numbers):

| Backend | macro-F1 | top-1 |
|---|---|---|
| heuristic *(pre-fix harness)* | 0.089 | 0.128 |
| pretrained-direct 527→12 *(pre-fix harness)* | 0.211 | 0.258 |
| head v1 (train-on-test, inflated) | *0.698* | *0.782* |
| head v2 (dev-trained, first-4s crops) | 0.635 | 0.740 |
| **head v3 (dev-trained, window-robust — deployed)** | **0.636** | **0.742** |

v3 trains on rolling-window-consistent crops (random 4 s slices; short clips
also placed sparsely in low-level context). On event-dense full clips it ties
v2, but in the live-mic transient regime it is markedly more robust: with a
0.5 s event at the window edge, true-class posterior 0.73 → **0.89** and top-1
16/24 → **21/24** (uploader-disjoint val clips). v1's row is invalid (30 % of
the test clips were in its training pool). The remaining gap to the 0.70
macro-F1 target is the **frozen mn10_as embedding ceiling** plus data-starved
classes (`fan`: 64 dev clips → F1 0.40); see Docs/ARCHITECTURE.md §10.

**Opt-in v4 "TV-rejection" head** (`VOICEISO_HEAD=checkpoints/efficientat_head12_v4.onnx`):
trained with synthesized loudspeaker/TV speech (the `tv` class had ZERO
FSD50K positives and could never fire), 6× fan augmentation, a stratified
validation split, and per-class calibrated thresholds shipped in ONNX
metadata. On held-out eval speech it rejects loudspeaker-played speech
**0.86–0.91 vs v3's 0.54** (labels it `tv` 0.41–0.68 vs **0.00**) with no live-speech
regression — but costs ~0.02 FSD50K macro-F1 (the new tv decision claims
channel-degraded field-recorded "speech" clips), so it **failed the
pre-registered no-regression deploy gate** and ships opt-in rather than
default. Use it for demos featuring a phone/TV playing voice next to the
speaker.

## Tests

```bash
python -m tests.test_phase_a        # classifier / config wiring (5)
python -m tests.test_robustness     # live containment, NaN immunity, reset independence (5)
python -m tests.test_deferred_tail  # deferred-tail timeline + metric alignment (13; fake DFN3
                                    # backend — no weights needed; 1 optional real-DFN3 test)
python -m tests.test_speech_guard   # speech cap wiring + speech-energy floor (6; fake backend)
```

Each runner is plain Python (prints PASS/SKIP/FAIL) and is also pytest-compatible.
`test_deferred_tail` pins: 20 ms tail = exactly 960 samples; tail 0 reproduces the
previous behaviour; bypass and DFN3-active blocks share one timeline; transitions
never splice; reset clears history and tail state; partial blocks; offline flush
recovers the tail; scoring never compares padding; known delays ±960 / +15 are
estimated exactly and impossible alignments are rejected; SI-SDR identity and
scale invariance; bit-identical repeated runs.

## Known limitations

- ECAPA speaker embedder, VoiceFilter-Lite mask, and tiny-GRU post-filter are
  scaffolded (passthrough) until models are provided.
- AEC is implemented but off by default (needs a far-end reference signal).
- No INT8 model ships; the ONNX export is FP32.
- **Extremely loud plate/crockery crashes and continuous clatter that overlap
  speech at ~0 dB local SNR are not removed** (measured 0.2–0.3 dB cut on the
  team's own recording): DFN3 treats the ringing, tonal impulses as
  speech-like. Hand-claps overlapping speech are cut ~12 dB; impulses in
  silence 30–50 dB. A sub-frame limiter was built for this case, measured
  inert, and removed — it is a limitation of the frozen architecture.
- Loudspeaker/TV speech is the weakest real-noise condition (silence-regime
  attenuation 14 dB; speech-regime SI-SDRi +3.8 dB at 5 dB SNR, below the
  4 dB objective) — see the opt-in v4 head above.
- Offline `process_signal()` / `voiceiso enhance` flush the 20 ms deferred DFN3
  tail at end-of-file against fabricated (zero) right context, so the final
  20 ms of an offline file have the pre-tail design's quality; live streams
  never do this.

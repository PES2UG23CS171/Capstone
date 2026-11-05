# voiceiso — working context

CPU-only real-time voice isolation for live calls (capstone, Team 190, PES University).
Mic → 13-stage streaming pipeline → headphones **or** a virtual microphone that
Google Meet / Zoom / Teams see as an input device. No GPU, no network, no cloud.

**Read this file instead of re-deriving the project.** For depth:
`Docs/ARCHITECTURE.md` (design + per-stage reference) and `README.md` (results,
runbooks). `Docs/ARCHITECTURE_DEEP_DIVE.md` is a local-only (gitignored)
long-form explainer written for panel preparation.

---

## Always use the venv interpreter

```bash
cd /Users/dhrus/Desktop/Capstone
PYTHONPATH=. .venv_poc/bin/python -m app.main          # desktop app
PYTHONPATH=. .venv_poc/bin/python -m voiceiso live     # CLI live path
PYTHONPATH=. .venv_poc/bin/python -m voiceiso devices  # list audio devices
```

System Python lacks `deepfilternet`; running with it makes the app start and do
**nothing audible** — this cost a full debugging session once. Always
`.venv_poc/bin/python` with `PYTHONPATH=.`.

Config can be overridden without editing code:
`VOICEISO_CFG="speech_floor_db=0,dfn_history_ms=240" PYTHONPATH=. .venv_poc/bin/python -m app.main`

---

## Architecture (frozen)

100 ms blocks @ 48 kHz. Stage order:

```
Preprocessing → AEC → VAD → SpeakerEmbedder → EfficientATNoiseClassifier →
CompetingSpeech → DynamicController → SpeechPreservation → Enhancement(DFN3) →
TargetSpeakerMask → MultiBandGainModulator → TinyGRUPostFilter → PostFilter
```

Every stage implements `process(ctx) -> ctx` / `reset()` over a shared
`FrameContext`. `SpeakerEmbedder`, `TargetSpeakerMask`, `TinyGRUPostFilter` are
**wired identity pass-throughs** awaiting checkpoints. `AEC` is implemented but
**off** (needs a far-end loopback reference that the app does not capture).

**Two facts about DeepFilterNet3 explain most of the design:**

1. **No cross-call memory.** Its GRUs reset every `enhance()` call; only STFT
   buffers persist. Small blocks starve it — 20 ms blocks measure *negative*
   SI-SDRi. Hence 100 ms blocks everywhere, and per-call priming with real
   history.
2. **It must be forced into a hard real-time budget.** All DSP runs on a worker
   thread; the audio callback only moves data across bounded drop-oldest queues,
   so a slow block costs bounded latency instead of an audible gap (xrun).

`StreamingPipeline.output_delay_samples` = **975** (960 deferred DFN3 tail +
15-sample FIR group delay). `process_signal()` flushes it so offline output is
sample-aligned and full length; live paths just carry it.

---

## Tuned defaults — every value was measured, do not "clean them up"

| Setting | Value | Why |
|---|---|---|
| `dfn_history_ms` | 160 | GRU state unconverged at 80 ms; clean-speech PESQ 3.72 → 3.85. 240 → +0.02 only. |
| `dfn_tail_ms` | 20 | DFN3 looks ~20 ms ahead; against pad zeros the block's last frames degrade. Costs exactly 20 ms latency, no compute. |
| `supp_speech_floor` | 0.20 | Cap on suppression while VAD confident (≈30 dB). **Was a dead constant 0.92 (93 dB) that protected nothing** → female voices lost whole word endings. Tighter (0.10) helps clean speech but re-admits noise under loud noise. |
| `speech_floor_db` | 9.0 | Speech-energy floor: repairs confident-speech dropouts. ON by **listener preference**; costs PESQ on plate-crash material (blends the removed burst back). `0` disables. |
| `artifact_rollback_enabled` | False | The 4-cue rollback fired on the *noise* and re-injected it. Off improved every metric (DNSMOS BAK 2.61 → 3.69). |
| `kick_crest_db` | 18.0 | Transient detector uses **intra-block crest** (max−min over 10 ms sub-frames) — level-independent. Earlier absolute-level designs worked in synthetic silence and failed live. |
| `kick_vad_max` | 0.75 | Confident speech vetoes the clap kick (fail-safe: a missed clap beats a clipped word). |
| `comfort_noise_db` | −85 | −65 and −78 were audible as static on sealed headphones once real noise was gone. |
| classifier head | `efficientat_head12_v3.onnx` | v1 was train-on-test (invalid). v4 rejects TV/loudspeaker far better but **failed a pre-registered no-regression gate** → opt-in via `VOICEISO_HEAD`. |

Also non-obvious: `_TARGET_TO_INTERNAL["speech"] = "other"` — mapping it to
`"clean"` disabled DFN3 exactly when the user spoke in a noisy room
(SI-SDRi −0.06 → +4.12 dB when fixed).

---

## Measuring — and the one trap

```bash
PYTHONPATH=. .venv_poc/bin/python -m voiceiso bench --n 20 --snr 5   # synthetic white noise
PYTHONPATH=. .venv_poc/bin/python -m scripts.realworld_matrix        # real FSD50K eval clips
PYTHONPATH=. .venv_poc/bin/python -m scripts.ground_truth_eval run --tag jargon --name x
```

**Intrusive metrics must be scored on the valid overlap** at the known 975-sample
delay — never against zero padding (that cost 0.24 dB SI-SDRi). Use
`estimate_lag` / `resolve_lag` / `valid_overlap` in `voiceiso/bench/benchmark.py`;
they reject impossible or ambiguous alignments instead of scoring garbage.
DNSMOS is non-intrusive → full unaligned output.

**Validate speech quality with PESQ on clean speech, not only SI-SDR on noisy
speech.** SI-SDR is dominated by residual noise energy and hid a real
block-boundary artifact for weeks.

Current headline (synthetic +5 dB, n=20, valid-overlap scoring): SI-SDRi +4.13,
PESQ 1.24 → 1.51, STOI 0.82, DNSMOS SIG 3.15 / BAK 3.70 / OVRL 2.77, RTF ~0.22,
p99 ≈ 39 ms, algorithmic latency 40 ms, mouth-to-ear ≈ 200–300 ms. Runs are
bit-identical (hash printed by the benchmark).

Criteria are **three separate things** and must not be conflated: Phase-2
objective SI-SDRi ≥ 4.0 dB (the release gate), aspirational Krisp-parity ≥ 10 dB
(never met, informational), real-time budget RTF ≤ 0.5.

---

## Tests

```bash
PYTHONPATH=. .venv_poc/bin/python -m tests.test_phase_a        # 5
PYTHONPATH=. .venv_poc/bin/python -m tests.test_robustness     # 5
PYTHONPATH=. .venv_poc/bin/python -m tests.test_deferred_tail  # 13
PYTHONPATH=. .venv_poc/bin/python -m tests.test_speech_guard   # 8
PYTHONPATH=. .venv_poc/bin/python -m tests.test_devices        # 12
```

43 total. `test_devices` runs the real engine loop against a fake `sounddevice`.
`test_deferred_tail` / `test_speech_guard` drive the real stages with a
**deterministic fake DFN3 backend**, so they need no model weights — that is why
CI can run them. GitHub Actions runs all four on every push (`.github/workflows/ci.yml`).

---

## Environment gotchas

- **Call mode fails in Safari.** With BlackHole selected as the mic, Safari's call
  audio dies in both directions. Demo in **Brave/Chrome**. Quit GarageBand first
  (it holds the mic).
- For an honest demo, turn OFF Meet's **Noise cancellation** *and* **Studio sound**.
- PyQt6 pinned to **6.7.1** — Qt 6.11's cocoa plugin fails on macOS 26.
  `app/main.py` also scrubs empty `QT_*` env vars before re-exec.
- **macOS 27 kills the process on any AppKit/UI call off the main thread**
  (SIGTRAP, `assertBarrierOnQueue`). The old pystray tray did exactly that and
  crashed the app ~3 s after launch. The tray is now Qt's `QSystemTrayIcon` on
  the main thread — never reintroduce pystray, PyObjC, or threaded UI code.
- After **any macOS update**, run `PYTHONPATH=. .venv_poc/bin/python -m scripts.gui_smoke`
  first: real window + tray, stub engine, no audio. CI cannot catch these.
- The engine child stops itself if the GUI dies (`parent_process().is_alive()`
  in `run_engine`'s loop). Before that, every GUI crash left an orphan engine
  holding the mic and speakers at ~38 % CPU.
- Speakers can't feed back on launch any more: the engine mutes loudspeaker
  outputs until the user clicks Unmute. Because BlackHole is installed on this Mac,
  the app now starts in call mode (no sound in headphones until you switch output).
- `~/Downloads` is TCC-blocked for the shell: `cp`/`unzip`/`textutil` get
  "Operation not permitted". Ask the user to move files into the repo.
- Not installed on this machine: `pandoc`, `poppler` (`pdftoppm`), LibreOffice
  (`soffice`). Word/Pages AppleScript automation is **not authorized**, so
  rendering `.docx`/`.pptx` to image for visual QA is not possible here — validate
  structurally and have the user eyeball it.
- The project must stay **out of iCloud Drive**; iCloud sync previously corrupted
  venv wheels and caused phantom Qt failures.

---

## How to work on this repo

Architecture is **frozen**: do not replace DFN3, EfficientAT, ONNX Runtime, the
controller or the streaming design; no new models, no speculative DSP.

Every change needs: observed bug → root cause → objective evidence → before/after
on identical inputs → benchmark comparison → rollback justification. If no
improvement is measurable, reject the change. Several plausible ideas were built,
measured, and **removed** on this rule (a sub-frame transient limiter; the artifact
rollback; a tighter speech cap) — that history is in `Docs/ARCHITECTURE.md`.

Do not quote synthetic white-noise numbers against real-noise numbers.

---

## Known limitations (state these honestly)

- Loud **plate/crockery crashes overlapping speech** at ~0 dB local SNR are not
  removed (~0.2 dB cut). DFN3 treats the ringing impulse as speech-like. A
  sub-frame limiter was built for it, measured inert, and deleted.
- **Female voices are improved, not equalised** (clean-speech PESQ ~3.17 vs ~4.00
  male). The residual gap is DFN3's own behaviour on that voice — it scores lower
  even offline one-shot (3.85 vs 4.26).
- **TV / loudspeaker speech** is the weakest real-noise class (−14 dB silence
  attenuation; +3.8 dB SI-SDRi at 5 dB SNR, below the 4 dB objective).
- Classifier ceiling is the **frozen backbone** plus data starvation, not the head
  (threshold sweep flat; MLP head +0.02).

## Status

Feature-complete as of 2026-11, by the team's decision after the panel review.
Female-voice tuning, calibration, personalization, and AEC validation are **not
planned**; don't start them unasked. The limitations above still stand and should
still be stated honestly.

## Packaging

`packaging/` builds a Mac `.dmg` and a Windows installer with PyInstaller. Each
platform must build its own, so Windows builds run only on GitHub Actions
(`.github/workflows/package.yml`). Release builds use Python **3.11**, because
`deepfilterlib` has no 3.12 wheels for Windows or Apple Silicon; pins are in
`packaging/requirements-build.txt`. Mac is Apple Silicon only (torch has no Intel
builds).

- Check any packaged build with `VoiceISO --self-test --self-test-out r.json`.
  `output_sha256` must match a source run with the same libraries **on the same
  machine**. It can differ between machines: two GitHub Mac runners with identical
  libraries and image gave different hashes and the same 48.6 dB noise reduction.
- Windows builds must ship one Microsoft C++ runtime. PyInstaller collected 14.26
  (PyQt6), 14.38 (Python) and 14.40 (a Java install on the runner's PATH); with those
  the app installed fine but no model loaded ("DLL initialization routine failed",
  0 dB reduction). `voiceiso.spec` replaces them all with the build machine's
  System32 copies. Which old copy was at fault was not isolated.
- Things that only break on a machine that isn't this one: `df` calls `git` at
  model load (no git → silent passthrough, patched in `voiceiso/_compat.py`); `df`
  writes a log into the model dir (`log_file=None`); windowed builds have
  `sys.stdout = None`, which crashes `df`'s logger (`packaging/launcher.py` redirects
  it to a log file); Windows lists each device once per host API (`devices.py`
  filters them).
- **Windows audio I/O has never run on real hardware**: CI runners have no sound
  devices. Test on a real PC after changing `app/audio/`.
- The app is not code-signed, so first launch shows OS warnings. Call mode needs
  a third-party driver (BlackHole / VB-CABLE) that the user installs once.
- On launch the engine picks a virtual cable if one is installed; otherwise it
  uses the system default, and loudspeaker outputs stay muted until the user
  clicks Unmute.

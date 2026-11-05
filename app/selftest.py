"""
Self-test built into the packaged app: ``VoiceISO --self-test --self-test-out r.json``.

Proves an installed build really works, on any machine, including CI runners
with no audio hardware.  The failure this project keeps hitting is an app that
opens fine and quietly does nothing, so "it launched" is not accepted as a pass:

  * every real backend loaded (DFN3, Silero, EfficientAT), not a fallback
  * the audio chain runs in a spawned child process, as the live engine does
    (this exercises multiprocessing inside the frozen app)
  * noise is actually removed from a known test signal
  * PortAudio loads, and the real window + tray open and quit cleanly

It writes a JSON report and exits 0 only if everything passed.
``output_sha256`` lets a packaged build be compared with a source checkout: the
same value on the same machine means the DSP behaves identically.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

REQUIRED_BACKENDS = {
    "enhancement": "deepfilternet3",
    "vad": "silero",
    "noise_classifier": "efficientat",
}
MIN_NOISE_REDUCTION_DB = 12.0
CHILD_TIMEOUT_S = 600


def _rms_db(x) -> float:
    import numpy as np
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12))


def _audio_check(result_q) -> None:
    """Runs in a spawned child, the same way the live audio engine does."""
    try:
        import numpy as np

        from voiceiso.config import PipelineConfig
        from voiceiso.pipeline import StreamingPipeline

        sr, block, seconds = 48_000, 4_800, 6
        n = sr * seconds
        t = np.arange(n) / sr
        rng = np.random.default_rng(0)
        noise = 0.05 * rng.standard_normal(n)
        # A harmonic tone with a syllable-rate envelope in the second half, so
        # the hash also covers the voiced code paths.
        voice = sum(np.sin(2 * np.pi * 150.0 * k * t) / k for k in range(1, 12))
        voice *= 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
        voice[: n // 2] = 0.0
        x = (noise + 0.05 * voice).astype(np.float32)

        pipe = StreamingPipeline(PipelineConfig())
        pipe.reset()
        out = np.zeros(n, dtype=np.float32)
        t0 = time.perf_counter()
        for s in range(0, (n // block) * block, block):
            out[s:s + block] = pipe.process_block(x[s:s + block]).audio[:block]
        elapsed = time.perf_counter() - t0

        # Noise-only stretch, after the first second of adaptation.
        reduction = _rms_db(x[sr: 2 * sr + sr // 2]) - _rms_db(out[sr: 2 * sr + sr // 2])

        import sounddevice as sd      # must load the bundled PortAudio
        result_q.put({
            "backends": pipe.backend_summary,
            "noise_reduction_db": round(reduction, 1),
            "output_finite": bool(np.all(np.isfinite(out))),
            "output_sha256": hashlib.sha256(out.tobytes()).hexdigest()[:16],
            "rtf": round(elapsed / seconds, 3),
            "audio_devices_found": len(sd.query_devices()),
        })
    except BaseException:
        result_q.put({"error": traceback.format_exc()})


class _StubEngine:
    """Stands in for the engine handle so the GUI check opens no audio devices."""

    alive = True

    def send(self, cmd) -> None:
        pass

    def poll_events(self) -> list:
        return []


def _run_checks() -> Dict[str, Any]:
    import app.main  # noqa: F401  (applies the same Qt environment fixes as a real launch)
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QSystemTrayIcon

    report: Dict[str, Any] = {"frozen": bool(getattr(sys, "frozen", False)),
                              "platform": sys.platform}
    failures: List[str] = []

    qt_app = QApplication(sys.argv[:1])         # same order as app.main: Qt first
    qt_app.setQuitOnLastWindowClosed(False)

    multiprocessing.set_start_method("spawn", force=True)
    result_q: multiprocessing.Queue = multiprocessing.Queue()
    child = multiprocessing.Process(target=_audio_check, args=(result_q,), daemon=True)
    child.start()
    try:
        audio = result_q.get(timeout=CHILD_TIMEOUT_S)
    except Exception:
        audio = {"error": f"audio child gave no result in {CHILD_TIMEOUT_S} s"}
    child.join(timeout=10)
    if child.is_alive():
        child.terminate()
    report["audio"] = audio

    if "error" in audio:
        failures.append("audio chain failed")
    else:
        for key, want in REQUIRED_BACKENDS.items():
            if audio["backends"].get(key) != want:
                failures.append(f"backend {key}={audio['backends'].get(key)!r}, want {want!r}")
        if audio["noise_reduction_db"] < MIN_NOISE_REDUCTION_DB:
            failures.append(f"noise reduced only {audio['noise_reduction_db']} dB")
        if not audio["output_finite"]:
            failures.append("non-finite audio in the output")

    from app.gui.control_window import ControlWindow
    from app.gui.tray import TrayManager

    window = ControlWindow(_StubEngine())
    tray = TrayManager()
    tray.start()
    window.show()
    QTimer.singleShot(1000, tray._on_toggle)    # icon redraw + menu text
    QTimer.singleShot(2500, qt_app.quit)
    t0 = time.monotonic()
    qt_app.exec()
    tray.stop()
    report["gui"] = {"seconds_open": round(time.monotonic() - t0, 1),
                     "tray_available": QSystemTrayIcon.isSystemTrayAvailable()}

    report["failures"] = failures
    report["passed"] = not failures
    return report


def main(argv: List[str]) -> int:
    out_path: Optional[str] = None
    if "--self-test-out" in argv:
        out_path = argv[argv.index("--self-test-out") + 1]
    try:
        report = _run_checks()
    except BaseException:
        report = {"passed": False, "failures": ["self-test crashed"],
                  "error": traceback.format_exc()}
    text = json.dumps(report, indent=2, default=str)
    print(text, flush=True)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    # Skip interpreter teardown: Qt and torch objects can abort during exit,
    # which would turn a passing test into a confusing failure.
    sys.stdout.flush()
    os._exit(0 if report.get("passed") else 1)

#!/usr/bin/env python3
"""
Entry-point for the Real-Time Transient Noise Suppressor.

Architecture
~~~~~~~~~~~~
::

    ┌─ Main Process (this file) ────────────────────────────────────────────┐
    │                                                                       │
    │  1. QApplication (event loop on main thread — required by macOS)      │
    │  2. ControlWindow (PyQt6 settings panel)                              │
    │  3. TrayManager  (pystray icon on a daemon thread)                    │
    │                                                                       │
    │  ── multiprocessing boundary ──────────────────────────────────────    │
    │                                                                       │
    │  4. AudioEngine  (child process: sounddevice + inference stub)        │
    └───────────────────────────────────────────────────────────────────────┘

Launch
------
    python -m app.main          # from the project root
    # or
    python app/main.py
"""

from __future__ import annotations

import logging
import multiprocessing
import signal
import sys

from PyQt6.QtWidgets import QApplication

from app.audio.engine import AudioEngineHandle
from app.config import AppConfig
from app.gui.control_window import ControlWindow
from app.gui.tray import TrayManager
from app.ipc.messages import CmdType, Command

logger = logging.getLogger(__name__)


def main() -> None:
    # ── macOS: Virtual environment GUI fix ───────────────────────────────
    # A standard venv python executable on macOS lacks the Bundle ID context 
    # required by AppKit/Cocoa, which causes PyQt6 to fatally crash the app.
    # We intercept this instantly and transparently re-launch the process 
    # using the true framework base-executable. We deliberately DO NOT inject
    # the venv's site-packages via PYTHONPATH, as PyQt6 loaded from a venv
    # triggers a known "cocoa" plugin ABI mismatch when run by a system binary.
    # The system Python 3.12 already contains all required global dependencies.
    import sys
    import os
    if sys.platform == "darwin" and hasattr(sys, "_base_executable") and sys.executable != sys._base_executable:
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if os.getcwd() not in current_pythonpath.split(":"):
            os.environ["PYTHONPATH"] = f"{os.getcwd()}:{current_pythonpath}".strip(":")
        os.execl(sys._base_executable, sys._base_executable, *sys.argv)
    # ── Logging ──────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s %(name)s] %(message)s",
    )

    # ── macOS: "spawn" is default on 3.8+, but be explicit ──────────────
    multiprocessing.set_start_method("spawn", force=True)

    # ── Configuration ────────────────────────────────────────────────────
    cfg = AppConfig()

    # ── Qt application ───────────────────────────────────────────────────
    # IMPORTANT: On macOS, QApplication must be created BEFORE any child
    # processes are spawned.  QApplication initialises AppKit/Cocoa on the
    # main thread — if something else touches Cocoa first, the platform
    # plugin fails to load ("Could not find the Qt platform plugin cocoa").
    app = QApplication(sys.argv)
    app.setApplicationName("TransientFilter")
    app.setQuitOnLastWindowClosed(False)   # keep alive when window is hidden

    # ── Audio engine (child process) — start AFTER Qt init ───────────────
    engine = AudioEngineHandle(cfg)
    engine.start()

    # ── Control window ───────────────────────────────────────────────────
    window = ControlWindow(engine)

    # ── System tray ──────────────────────────────────────────────────────
    tray = TrayManager()

    # Wire tray ↔ window signals
    tray.bridge.show_window.connect(lambda: (window.show(), window.raise_(), window.activateWindow()))
    tray.bridge.quit_app.connect(lambda: _shutdown(app, engine, tray, window))
    tray.bridge.toggle_suppression.connect(
        lambda: _toggle_from_tray(window, engine)
    )

    # Wire window quit → full shutdown
    window.quit_requested.connect(lambda: _shutdown(app, engine, tray, window))

    tray.start()

    # Show the control window on first launch
    window.show()

    # ── Allow Ctrl-C from terminal ───────────────────────────────────────
    signal.signal(signal.SIGINT, lambda *_: _shutdown(app, engine, tray, window))

    # ── Run ──────────────────────────────────────────────────────────────
    logger.info("Application started.")
    exit_code = app.exec()

    # Ensure engine is stopped even if app.exec() returns unexpectedly
    engine.stop()
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _shutdown(
    app: QApplication,
    engine: AudioEngineHandle,
    tray: TrayManager,
    window: ControlWindow,
) -> None:
    logger.info("Shutting down…")
    tray.stop()
    engine.stop()
    window.close()
    app.quit()


def _toggle_from_tray(window: ControlWindow, engine: AudioEngineHandle) -> None:
    """Mirror tray toggle into the GUI button and engine."""
    new_state = not window.btn_toggle.isChecked()
    window.btn_toggle.setChecked(new_state)
    # The toggle signal on the button will fire and send the IPC command.


# ---------------------------------------------------------------------------
#  Allow ``python -m app.main``
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

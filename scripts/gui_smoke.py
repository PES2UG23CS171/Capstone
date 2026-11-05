"""GUI smoke test — run this FIRST after any macOS update.

Opens the real control window and the menu-bar tray on the real display for
8 seconds with a stub engine: no microphone, no speakers, no models.  CI cannot
catch this class of failure (it has no window server), and macOS upgrades have
broken the GUI twice already:

  * macOS 26: Qt 6.11's cocoa plugin failed to load  → PyQt6 pinned to 6.7.1
  * macOS 27: the pystray tray touched AppKit from a background thread → the
    OS killed the process with SIGTRAP ~3 s after launch → QSystemTrayIcon

    PYTHONPATH=. .venv_poc/bin/python -m scripts.gui_smoke

Exit 0 and "GUI SMOKE: OK" = the GUI survives on this OS.  A crash shows up as
a non-zero exit (133 = SIGTRAP) plus a Python-*.ips report in
~/Library/Logs/DiagnosticReports/.
"""
import os
import sys
import sysconfig
from pathlib import Path

# Same re-exec as app/main.py: AppKit needs the framework interpreter's bundle
# context, and that interpreter must still see the venv's site-packages.
if sys.platform == "darwin" and hasattr(sys, "_base_executable") and sys.executable != sys._base_executable:
    repo = str(Path(__file__).resolve().parent.parent)
    site = [p for p in (sysconfig.get_path("purelib"), sysconfig.get_path("platlib")) if p]
    rest = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p not in site and p != repo]
    os.environ["PYTHONPATH"] = os.pathsep.join([repo] + site + rest)
    os.execv(sys._base_executable, [sys._base_executable, "-m", "scripts.gui_smoke"] + sys.argv[1:])

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


class _StubEngine:
    """The three methods ControlWindow calls on the engine handle."""

    def send(self, cmd):
        pass

    def poll_events(self):
        return []

    def alive(self):
        return True


def main() -> int:
    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)

    from app.gui.control_window import ControlWindow  # after QApplication (as in app.main)
    from app.gui.tray import TrayManager

    window = ControlWindow(_StubEngine())
    tray = TrayManager()
    tray.bridge.quit_app.connect(app.quit)
    tray.start()
    window.show()

    QTimer.singleShot(3000, tray._on_toggle)   # exercise icon redraw + menu text
    QTimer.singleShot(8000, app.quit)
    rc = app.exec()
    tray.stop()
    print(f"GUI SMOKE: {'OK' if rc == 0 else 'FAILED'} — window + tray survived 8 s, exit {rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())

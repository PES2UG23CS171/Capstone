"""
System-tray icon powered by **pystray**.

Runs in a background daemon thread so the PyQt6 event loop keeps the main
thread (required on macOS).  Menu actions communicate back to the GUI via
thread-safe Qt signals on an intermediary ``QObject`` bridge.
"""

from __future__ import annotations

import sys
import threading
from typing import Optional

from PIL import Image, ImageDraw

from PyQt6.QtCore import QObject, pyqtSignal

import pystray


# ---------------------------------------------------------------------------
#  Signal bridge (lives in the main/Qt thread, called from the tray thread)
# ---------------------------------------------------------------------------


class _TrayBridge(QObject):
    """Thread-safe signal emitter used by the pystray menu callbacks."""

    show_window = pyqtSignal()
    toggle_suppression = pyqtSignal()
    quit_app = pyqtSignal()


# ---------------------------------------------------------------------------
#  Icon generation (no external image file required)
# ---------------------------------------------------------------------------


def _make_icon(size: int = 64, active: bool = True) -> Image.Image:
    """Draw a simple mic icon as a PIL Image.

    Green when *active*, red when bypassed.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    colour = (34, 197, 94) if active else (239, 68, 68)  # green / red
    pad = size // 6

    # Microphone body (rounded rectangle approximation)
    draw.ellipse(
        [pad, pad // 2, size - pad, size // 2 + pad],
        fill=colour,
    )
    draw.rectangle(
        [pad, size // 4, size - pad, size // 2 + pad],
        fill=colour,
    )

    # Stand
    mid_x = size // 2
    stem_top = size // 2 + pad
    stem_bottom = size - pad
    draw.line(
        [(mid_x, stem_top), (mid_x, stem_bottom)],
        fill=colour,
        width=max(2, size // 12),
    )
    # Base
    draw.line(
        [(mid_x - pad, stem_bottom), (mid_x + pad, stem_bottom)],
        fill=colour,
        width=max(2, size // 12),
    )

    return img


# ---------------------------------------------------------------------------
#  TrayManager
# ---------------------------------------------------------------------------


class TrayManager:
    """Manages the pystray ``Icon`` on a background thread.

    Usage::

        tray = TrayManager()
        tray.bridge.show_window.connect(control_window.show)
        tray.bridge.toggle_suppression.connect(on_toggle)
        tray.bridge.quit_app.connect(app.quit)
        tray.start()
    """

    def __init__(self) -> None:
        self.bridge = _TrayBridge()

        self._suppression_on = True

        menu = pystray.Menu(
            pystray.MenuItem(
                "Show / Hide",
                self._on_show,
                default=True,          # double-click action
            ),
            pystray.MenuItem(
                lambda _: "Suppression: ON" if self._suppression_on else "Suppression: OFF",
                self._on_toggle,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._on_quit),
        )

        self._icon = pystray.Icon(
            name="TransientFilter",
            icon=_make_icon(active=True),
            title="Transient Noise Suppressor",
            menu=menu,
        )

        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the tray icon on a daemon thread."""
        self._thread = threading.Thread(
            target=self._icon.run,
            daemon=True,
            name="SysTray",
        )
        self._thread.start()

    def stop(self) -> None:
        """Programmatically remove the tray icon."""
        try:
            self._icon.stop()
        except Exception:
            pass

    # ── menu callbacks (called on the tray thread) ───────────────────────

    def _on_show(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self.bridge.show_window.emit()

    def _on_toggle(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._suppression_on = not self._suppression_on
        self._icon.icon = _make_icon(active=self._suppression_on)
        self.bridge.toggle_suppression.emit()

    def _on_quit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self.bridge.quit_app.emit()

"""
System-tray (menu-bar) icon built on Qt's ``QSystemTrayIcon``.

Everything here runs on the Qt main thread.

Why not pystray (the previous implementation): it ran ``pystray.Icon.run``
on a background thread, and on macOS pystray creates and updates the
``NSStatusItem`` from that thread.  AppKit never supported UI calls off the
main thread; macOS 27 now enforces it — ``-[NSStatusItem setMenu:]`` reaches
``-[BSServiceMainRunLoopQueue assertBarrierOnQueue]`` and the GUI process
dies with SIGTRAP about 3 s after launch (reproduced with the tray alone on
the real menu bar, exit 133).  ``QSystemTrayIcon`` does the same AppKit work
on the main thread, needs no extra thread, and drops the pystray / PyObjC /
Pillow dependency chain.

The public interface is unchanged::

    tray = TrayManager()                      # after QApplication exists
    tray.bridge.show_window.connect(control_window.show)
    tray.bridge.toggle_suppression.connect(on_toggle)
    tray.bridge.quit_app.connect(app.quit)
    tray.start()
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon


class _TrayBridge(QObject):
    """Signals the tray emits towards the GUI."""

    show_window = pyqtSignal()
    toggle_suppression = pyqtSignal()
    quit_app = pyqtSignal()


def _make_icon(size: int = 64, active: bool = True) -> QIcon:
    """Draw the microphone glyph: green when *active*, red when bypassed.

    Same geometry as the former PIL icon, drawn with QPainter (requires a
    QGuiApplication, which ``TrayManager`` callers already have).
    """
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    colour = QColor(34, 197, 94) if active else QColor(239, 68, 68)
    pad = size // 6

    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    # Microphone head (ellipse + rectangle)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(colour)
    p.drawEllipse(QRectF(pad, pad // 2, size - 2 * pad, size // 2 + pad - pad // 2))
    p.drawRect(QRectF(pad, size // 4, size - 2 * pad, size // 2 + pad - size // 4))
    # Stand and base
    p.setPen(QPen(colour, max(2, size // 12)))
    mid_x = size / 2
    stem_top = size // 2 + pad
    stem_bottom = size - pad
    p.drawLine(QPointF(mid_x, stem_top), QPointF(mid_x, stem_bottom))
    p.drawLine(QPointF(mid_x - pad, stem_bottom), QPointF(mid_x + pad, stem_bottom))
    p.end()
    return QIcon(pix)


class TrayManager:
    """Owns the ``QSystemTrayIcon``.  Main thread only."""

    def __init__(self) -> None:
        self.bridge = _TrayBridge()
        self._suppression_on = True
        self._icon: Optional[QSystemTrayIcon] = None

        if not QSystemTrayIcon.isSystemTrayAvailable():
            return          # no tray on this desktop: the window is the whole UI

        self._menu = QMenu()
        act_show = QAction("Show / Hide", self._menu)
        act_show.triggered.connect(self.bridge.show_window)
        self._act_toggle = QAction("Suppression: ON", self._menu)
        self._act_toggle.triggered.connect(self._on_toggle)
        act_quit = QAction("Quit", self._menu)
        act_quit.triggered.connect(self.bridge.quit_app)
        self._menu.addAction(act_show)
        self._menu.addAction(self._act_toggle)
        self._menu.addSeparator()
        self._menu.addAction(act_quit)

        self._icon = QSystemTrayIcon(_make_icon(active=True))
        self._icon.setToolTip("Transient Noise Suppressor")
        self._icon.setContextMenu(self._menu)
        # Double-click opens the window (Windows/Linux; on macOS a click opens
        # the menu, whose first item does the same).
        self._icon.activated.connect(self._on_activated)

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._icon is not None:
            self._icon.show()

    def stop(self) -> None:
        if self._icon is not None:
            self._icon.hide()

    # ── callbacks (main thread) ──────────────────────────────────────────

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.bridge.show_window.emit()

    def _on_toggle(self, *_: object) -> None:
        self._suppression_on = not self._suppression_on
        self._act_toggle.setText("Suppression: ON" if self._suppression_on else "Suppression: OFF")
        if self._icon is not None:
            self._icon.setIcon(_make_icon(active=self._suppression_on))
        self.bridge.toggle_suppression.emit()

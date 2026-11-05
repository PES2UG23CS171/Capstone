"""PyQt6 control window — the settings panel for the audio filter.

Layout (top to bottom):
    header   : app name, live status dot (starting / running / died), RTF
    toggle   : hero Suppression ON/OFF pill
    devices  : mic + output combos, with an automatic "Call mode" badge when
               the output is a virtual sink (BlackHole/Loopback/VB-Cable) —
               i.e. the far side of a Meet/Zoom call hears the clean stream
               and a link to the free driver when no virtual cable is installed
    guard    : banner while a loudspeaker output is muted to prevent feedback
    sliders  : strength (0–100 %), output gain (−12 … +6 dB; capped for
               open-speaker feedback safety)
    meters   : input / output level bars (dBFS)
    A/B      : momentary-style raw-mic passthrough for live comparisons

The window hides (not quits) on close so the tray icon stays active.  All
engine communication goes through the AudioEngineHandle command/event queues;
this class never touches audio directly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.audio.devices import DRIVER_URLS, call_app_mic_name, is_virtual_device
from app.audio.engine import AudioEngineHandle
from app.ipc.messages import (
    CmdType,
    Command,
    DeviceInfo,
    Event,
    EvtType,
    StatusPayload,
)

_MIC_PERMISSION_HINT = {
    "darwin": "System Settings → Privacy & Security → Microphone",
    "win32": "Settings → Privacy & security → Microphone",
}

_QSS = """
QMainWindow, QWidget { background: #14161a; color: #e5e7eb;
                       font-size: 13px; }
QLabel[cls="title"]   { font-size: 17px; font-weight: 700; color: #f3f4f6; }
QLabel[cls="caption"] { font-size: 11px; color: #9ca3af; }
QLabel[cls="section"] { font-size: 11px; font-weight: 700; color: #6b7280;
                        letter-spacing: 1px; }
QLabel[cls="value"]   { color: #d1d5db; }
QLabel[cls="badge"]   { background: #123c2b; color: #34d399;
                        border: 1px solid #1d5c42; border-radius: 9px;
                        padding: 3px 10px; font-size: 11px; font-weight: 600; }
QLabel[cls="warn"]    { color: #fbbf24; font-size: 12px; }

QPushButton#toggle { background: #16a34a; color: white; border: none;
                     border-radius: 10px; font-size: 15px; font-weight: 700;
                     min-height: 46px; }
QPushButton#toggle:!checked { background: #b91c1c; }
QPushButton#toggle:hover { background: #15803d; }
QPushButton#toggle:!checked:hover { background: #dc2626; }

QPushButton#ab { background: #1f2430; color: #9ca3af;
                 border: 1px solid #2d3442; border-radius: 8px;
                 min-height: 32px; font-weight: 600; }
QPushButton#ab:checked { background: #1d4ed8; color: white;
                         border-color: #1d4ed8; }
QPushButton#ab:hover { border-color: #4b5563; }

QComboBox { background: #1f2430; border: 1px solid #2d3442; border-radius: 7px;
            padding: 5px 10px; min-height: 22px; }
QComboBox:hover { border-color: #4b5563; }
QComboBox QAbstractItemView { background: #1f2430; color: #e5e7eb;
                              selection-background-color: #1d4ed8; }

QSlider::groove:horizontal { height: 5px; background: #2d3442;
                             border-radius: 2px; }
QSlider::sub-page:horizontal { background: #3b82f6; border-radius: 2px; }
QSlider::handle:horizontal { width: 16px; height: 16px; margin: -6px 0;
                             border-radius: 8px; background: #e5e7eb; }
QSlider::handle:horizontal:hover { background: #ffffff; }
QSlider:disabled::sub-page:horizontal { background: #374151; }

QProgressBar { border: none; border-radius: 3px; background: #1f2430;
               height: 10px; }
QProgressBar::chunk { border-radius: 3px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #22c55e, stop:0.72 #facc15, stop:1.0 #ef4444); }

QStatusBar { background: #101215; color: #9ca3af; font-size: 11px; }
"""


class LevelMeter(QProgressBar):
    """Compact horizontal level-meter (dBFS)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setRange(-60, 0)
        self.setValue(-60)
        self.setTextVisible(False)
        self.setFixedHeight(10)

    def set_level(self, db: float) -> None:
        self.setValue(max(-60, min(0, int(db))))


def _section(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setProperty("cls", "section")
    return lbl


class ControlWindow(QMainWindow):
    """Main settings window shown from the system tray."""

    # Emitted when the user explicitly quits (tray Quit).
    quit_requested = pyqtSignal()

    # Seconds an ERROR message stays pinned before STATUS may overwrite it.
    _ERROR_HOLD_S = 8.0

    def __init__(self, engine: AudioEngineHandle,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._error_until = 0.0
        self._engine_seen_alive = False
        self._saw_status = False
        self._mic_silent_since: Optional[float] = None

        self.setWindowTitle("VoiceISO")
        # Title bar and taskbar on Windows; the Dock when running from source on Mac.
        QApplication.setWindowIcon(QIcon(str(Path(__file__).with_name("logo.png"))))
        self.setMinimumWidth(440)
        self.setStyleSheet(_QSS)

        self._build_ui()
        self._connect_signals()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)          # ~20 fps
        self._poll_timer.timeout.connect(self._poll_engine)
        self._poll_timer.start()

        self._engine.send(Command(CmdType.GET_DEVICES))

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(14)
        root.setContentsMargins(18, 16, 18, 12)

        # Header: title + status dot + RTF caption
        head = QHBoxLayout()
        title = QLabel("VoiceISO")
        title.setProperty("cls", "title")
        head.addWidget(title)
        head.addStretch(1)
        self.lbl_state = QLabel("●  starting…")
        self.lbl_state.setStyleSheet("color: #f59e0b; font-weight: 600;")
        head.addWidget(self.lbl_state)
        root.addLayout(head)

        self.lbl_rtf = QLabel("real-time voice isolation — warming up")
        self.lbl_rtf.setProperty("cls", "caption")
        root.addWidget(self.lbl_rtf)

        # Hero toggle
        self.btn_toggle = QPushButton("Suppression  ON")
        self.btn_toggle.setObjectName("toggle")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(True)
        root.addWidget(self.btn_toggle)

        # Devices
        dev_head = QHBoxLayout()
        dev_head.addWidget(_section("Devices"))
        dev_head.addStretch(1)
        self.btn_rescan = QPushButton("↻  Rescan")
        self.btn_rescan.setObjectName("ab")
        self.btn_rescan.setToolTip("Plugged in headphones or installed a driver? Click to find it.")
        dev_head.addWidget(self.btn_rescan)
        root.addLayout(dev_head)
        dev = QGridLayout()
        dev.setHorizontalSpacing(10)
        dev.setVerticalSpacing(8)
        dev.addWidget(QLabel("Mic"), 0, 0)
        self.combo_input = QComboBox()
        dev.addWidget(self.combo_input, 0, 1)
        dev.addWidget(QLabel("Output"), 1, 0)
        self.combo_output = QComboBox()
        dev.addWidget(self.combo_output, 1, 1)
        dev.setColumnStretch(1, 1)
        root.addLayout(dev)

        self.lbl_callmode = QLabel()
        self.lbl_callmode.setProperty("cls", "badge")
        self.lbl_callmode.setVisible(False)
        root.addWidget(self.lbl_callmode, alignment=Qt.AlignmentFlag.AlignLeft)

        # Shown only when no virtual cable is installed.
        self.lbl_driver = QLabel()
        self.lbl_driver.setProperty("cls", "caption")
        self.lbl_driver.setOpenExternalLinks(True)
        self.lbl_driver.setWordWrap(True)
        self.lbl_driver.setVisible(False)
        driver = DRIVER_URLS.get(sys.platform)
        if driver is not None:
            self.lbl_driver.setText(
                "Using Meet or Zoom? Install the free "
                f'<a style="color:#60a5fa" href="{driver[1]}">{driver[0]}</a> '
                "driver once, then click Rescan.")
        root.addWidget(self.lbl_driver)

        # Feedback guard
        self.row_guard = QWidget()
        guard = QHBoxLayout(self.row_guard)
        guard.setContentsMargins(0, 0, 0, 0)
        lbl_guard = QLabel("🔇  Output muted: these speakers could feed back into the "
                           "mic. Use headphones and click Rescan, or:")
        lbl_guard.setProperty("cls", "warn")
        lbl_guard.setWordWrap(True)
        guard.addWidget(lbl_guard, 1)
        self.btn_unmute = QPushButton("Unmute")
        self.btn_unmute.setObjectName("ab")
        guard.addWidget(self.btn_unmute)
        self.row_guard.setVisible(False)
        root.addWidget(self.row_guard)

        # Sliders
        root.addWidget(_section("Processing"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        grid.addWidget(QLabel("Strength"), 0, 0)
        self.slider_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_strength.setRange(0, 100)
        self.slider_strength.setValue(100)
        grid.addWidget(self.slider_strength, 0, 1)
        self.lbl_strength = QLabel("100 %")
        self.lbl_strength.setProperty("cls", "value")
        self.lbl_strength.setFixedWidth(58)
        self.lbl_strength.setAlignment(Qt.AlignmentFlag.AlignRight
                                       | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.lbl_strength, 0, 2)

        grid.addWidget(QLabel("Gain"), 1, 0)
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(-120, 60)       # tenths of dB; +6 dB cap
        self.slider_gain.setValue(0)
        grid.addWidget(self.slider_gain, 1, 1)
        self.lbl_gain = QLabel("+0.0 dB")
        self.lbl_gain.setProperty("cls", "value")
        self.lbl_gain.setFixedWidth(58)
        self.lbl_gain.setAlignment(Qt.AlignmentFlag.AlignRight
                                   | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.lbl_gain, 1, 2)
        grid.setColumnStretch(1, 1)
        root.addLayout(grid)

        # Meters
        root.addWidget(_section("Levels"))
        meters = QGridLayout()
        meters.setHorizontalSpacing(10)
        meters.setVerticalSpacing(8)
        meters.addWidget(QLabel("In"), 0, 0)
        self.meter_in = LevelMeter()
        meters.addWidget(self.meter_in, 0, 1)
        self.lbl_in_db = QLabel("−∞")
        self.lbl_in_db.setProperty("cls", "value")
        self.lbl_in_db.setFixedWidth(58)
        self.lbl_in_db.setAlignment(Qt.AlignmentFlag.AlignRight
                                    | Qt.AlignmentFlag.AlignVCenter)
        meters.addWidget(self.lbl_in_db, 0, 2)
        meters.addWidget(QLabel("Out"), 1, 0)
        self.meter_out = LevelMeter()
        meters.addWidget(self.meter_out, 1, 1)
        self.lbl_out_db = QLabel("−∞")
        self.lbl_out_db.setProperty("cls", "value")
        self.lbl_out_db.setFixedWidth(58)
        self.lbl_out_db.setAlignment(Qt.AlignmentFlag.AlignRight
                                     | Qt.AlignmentFlag.AlignVCenter)
        meters.addWidget(self.lbl_out_db, 1, 2)
        meters.setColumnStretch(1, 1)
        root.addLayout(meters)

        # A/B raw-mic comparison (passthrough)
        self.btn_passthrough = QPushButton("A / B  —  hear the RAW mic")
        self.btn_passthrough.setObjectName("ab")
        self.btn_passthrough.setCheckable(True)
        root.addWidget(self.btn_passthrough)

        self.statusBar().showMessage("Engine starting…")

    # ── Signal wiring ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.btn_toggle.toggled.connect(self._on_toggle)
        self.btn_passthrough.toggled.connect(self._on_passthrough)
        self.slider_strength.valueChanged.connect(self._on_strength)
        self.slider_gain.valueChanged.connect(self._on_gain)
        self.combo_input.currentIndexChanged.connect(self._on_input_device)
        self.combo_output.currentIndexChanged.connect(self._on_output_device)
        self.btn_rescan.clicked.connect(
            lambda: self._engine.send(Command(CmdType.REFRESH_DEVICES)))
        self.btn_unmute.clicked.connect(
            lambda: self._engine.send(Command(CmdType.ALLOW_SPEAKERS)))

    # ── Slots ────────────────────────────────────────────────────────────

    def _update_callmode_badge(self) -> None:
        name = self.combo_output.currentText()
        is_call_mode = is_virtual_device(name)
        if is_call_mode:
            self.lbl_callmode.setText(
                f"☎  Call mode — in Meet/Zoom, choose “{call_app_mic_name(name)}” "
                "as the mic. You won't hear yourself here.")
        self.lbl_callmode.setVisible(is_call_mode)

    def _on_toggle(self, checked: bool) -> None:
        self.btn_toggle.setText(f"Suppression  {'ON' if checked else 'OFF'}")
        self._engine.send(Command(CmdType.SET_ENABLED, checked))

    def _on_passthrough(self, checked: bool) -> None:
        self._engine.send(Command(CmdType.SET_PASSTHROUGH, checked))
        self.btn_passthrough.setText(
            "A / B  —  RAW MIC (unprocessed!)" if checked
            else "A / B  —  hear the RAW mic")
        # Suppression controls are inert while raw audio bypasses the chain.
        self.slider_strength.setEnabled(not checked)
        self.btn_toggle.setEnabled(not checked)
        self.slider_gain.setEnabled(not checked)

    def _on_strength(self, value: int) -> None:
        self.lbl_strength.setText(f"{value} %")
        self._engine.send(Command(CmdType.SET_STRENGTH, value / 100.0))

    def _on_gain(self, value: int) -> None:
        db = value / 10.0
        self.lbl_gain.setText(f"{db:+.1f} dB")
        self._engine.send(Command(CmdType.SET_GAIN, db))

    def _on_input_device(self, index: int) -> None:
        dev_idx = self.combo_input.itemData(index)
        if dev_idx is not None:
            self._engine.send(Command(CmdType.SET_INPUT_DEVICE, dev_idx))

    def _on_output_device(self, index: int) -> None:
        dev_idx = self.combo_output.itemData(index)
        if dev_idx is not None:
            self._engine.send(Command(CmdType.SET_OUTPUT_DEVICE, dev_idx))
        self._update_callmode_badge()

    # ── Engine event polling ─────────────────────────────────────────────

    def _poll_engine(self) -> None:
        for evt in self._engine.poll_events():
            if evt.kind == EvtType.STATUS:
                self._saw_status = True
                self._handle_status(evt.payload)
            elif evt.kind == EvtType.DEVICE_LIST:
                self._handle_device_list(evt.payload)
            elif evt.kind == EvtType.ERROR:
                self._error_until = time.monotonic() + self._ERROR_HOLD_S
                self.statusBar().showMessage(f"⚠  {evt.payload}")
            elif evt.kind == EvtType.ENGINE_STOPPED:
                self.statusBar().showMessage("Engine stopped.")
        # Death check covers both the running phase and STARTUP (a crash
        # during model load / warm-up, before the first STATUS).
        alive = self._engine.alive
        if self._engine_seen_alive and not alive:
            self.lbl_state.setText("●  engine died")
            self.lbl_state.setStyleSheet("color: #ef4444; font-weight: 700;")
            self.statusBar().showMessage(
                "🔴 ENGINE PROCESS DIED — audio has stopped. Restart the app.")
            self.meter_in.set_level(-120.0)
            self.meter_out.set_level(-120.0)
            return
        if alive:
            self._engine_seen_alive = True

    def _handle_status(self, s: StatusPayload) -> None:
        self.meter_in.set_level(s.input_level_db)
        self.meter_out.set_level(s.output_level_db)

        def _fmt(db: float) -> str:
            return "−∞" if db <= -120 else f"{db:.0f} dB"

        self.lbl_in_db.setText(_fmt(s.input_level_db))
        self.lbl_out_db.setText(_fmt(s.output_level_db))
        self.row_guard.setVisible(s.output_muted)

        self.lbl_state.setText("●  running")
        self.lbl_state.setStyleSheet("color: #22c55e; font-weight: 600;")
        if s.rtf > 0:
            self.lbl_rtf.setText(
                f"RTF {s.rtf:.2f}  ·  {1.0 / s.rtf:.1f}× real-time headroom")
        else:
            self.lbl_rtf.setText("real-time voice isolation")

        if time.monotonic() < self._error_until:
            return                     # keep the pinned error visible

        # A real mic always has some noise.  Exact digital silence for seconds
        # means the OS is withholding it, usually a denied permission.
        now = time.monotonic()
        if s.input_level_db > -119.0:
            self._mic_silent_since = None
        elif self._mic_silent_since is None:
            self._mic_silent_since = now
        elif now - self._mic_silent_since > 5.0:
            where = _MIC_PERMISSION_HINT.get(sys.platform, "your system settings")
            self.statusBar().showMessage(f"⚠  No mic signal. Allow access in {where}.")
            return

        xr = f"  |  x-runs: {s.xruns}" if s.xruns else ""
        self.statusBar().showMessage(f"Engine running{xr}")

    def _handle_device_list(self, devices: List[DeviceInfo]) -> None:
        self.combo_input.blockSignals(True)
        self.combo_output.blockSignals(True)
        self.combo_input.clear()
        self.combo_output.clear()
        self.combo_input.addItem("System Default", None)
        self.combo_output.addItem("System Default", None)

        # Show what the engine is actually using (signals are blocked, so this
        # does not send a switch command back).
        current_in = 0
        current_out = 0
        has_cable = False
        for d in devices:
            if d.max_input_channels > 0:
                self.combo_input.addItem(d.name, d.index)
                if d.is_selected_input:
                    current_in = self.combo_input.count() - 1
            if d.max_output_channels > 0:
                self.combo_output.addItem(d.name, d.index)
                has_cable = has_cable or is_virtual_device(d.name)
                if d.is_selected_output:
                    current_out = self.combo_output.count() - 1

        self.combo_input.setCurrentIndex(current_in)
        self.combo_output.setCurrentIndex(current_out)
        self.combo_input.blockSignals(False)
        self.combo_output.blockSignals(False)
        self._update_callmode_badge()
        self.lbl_driver.setVisible(not has_cable and bool(self.lbl_driver.text()))

    # ── Window behaviour ─────────────────────────────────────────────────

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Hide instead of quitting so the tray icon stays active."""
        event.ignore()
        self.hide()

    def request_quit(self) -> None:
        """Called by the tray Quit action — actually exits."""
        self._poll_timer.stop()
        self.quit_requested.emit()

"""PyQt6 control window — the settings panel for the audio filter.

Layout (top to bottom):
    header   : app name, live status dot (starting / running / died), RTF
    toggle   : hero Suppression ON/OFF pill
    devices  : mic + output combos, with an automatic "Call mode" badge when
               the output is a virtual sink (BlackHole/Loopback/VB-Cable) —
               i.e. the far side of a Meet/Zoom call hears the clean stream
    sliders  : strength (0–100 %), output gain (−12 … +6 dB; capped for
               open-speaker feedback safety)
    meters   : input / output level bars (dBFS)
    A/B      : momentary-style raw-mic passthrough for live comparisons

The window hides (not quits) on close so the tray icon stays active.  All
engine communication goes through the AudioEngineHandle command/event queues;
this class never touches audio directly.
"""

from __future__ import annotations

import time
from typing import List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
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

from app.audio.engine import AudioEngineHandle
from app.ipc.messages import (
    CmdType,
    Command,
    DeviceInfo,
    Event,
    EvtType,
    StatusPayload,
)

_VIRTUAL_SINKS = ("blackhole", "loopback", "vb-cable", "soundflower")

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

        self.setWindowTitle("VoiceISO")
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
        root.addWidget(_section("Devices"))
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

        self.lbl_callmode = QLabel("☎  Call mode — the far side hears the clean stream")
        self.lbl_callmode.setProperty("cls", "badge")
        self.lbl_callmode.setVisible(False)
        root.addWidget(self.lbl_callmode, alignment=Qt.AlignmentFlag.AlignLeft)

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

    # ── Slots ────────────────────────────────────────────────────────────

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
        name = self.combo_output.itemText(index).lower()
        self.lbl_callmode.setVisible(any(m in name for m in _VIRTUAL_SINKS))

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

        self.lbl_state.setText("●  running")
        self.lbl_state.setStyleSheet("color: #22c55e; font-weight: 600;")
        if s.rtf > 0:
            self.lbl_rtf.setText(
                f"RTF {s.rtf:.2f}  ·  {1.0 / s.rtf:.1f}× real-time headroom")
        else:
            self.lbl_rtf.setText("real-time voice isolation")

        if time.monotonic() < self._error_until:
            return                     # keep the pinned error visible
        xr = f"  |  x-runs: {s.xruns}" if s.xruns else ""
        self.statusBar().showMessage(f"Engine running{xr}")

    def _handle_device_list(self, devices: List[DeviceInfo]) -> None:
        self.combo_input.blockSignals(True)
        self.combo_output.blockSignals(True)
        self.combo_input.clear()
        self.combo_output.clear()
        self.combo_input.addItem("System Default", None)
        self.combo_output.addItem("System Default", None)

        default_in_idx = 0
        default_out_idx = 0
        for d in devices:
            if d.max_input_channels > 0:
                self.combo_input.addItem(d.name, d.index)
                if d.is_default_input:
                    default_in_idx = self.combo_input.count() - 1
            if d.max_output_channels > 0:
                self.combo_output.addItem(d.name, d.index)
                if d.is_default_output:
                    default_out_idx = self.combo_output.count() - 1

        self.combo_input.setCurrentIndex(default_in_idx)
        self.combo_output.setCurrentIndex(default_out_idx)
        self.combo_input.blockSignals(False)
        self.combo_output.blockSignals(False)
        # Reflect the (restored) output selection in the call-mode badge.
        name = self.combo_output.currentText().lower()
        self.lbl_callmode.setVisible(any(m in name for m in _VIRTUAL_SINKS))

    # ── Window behaviour ─────────────────────────────────────────────────

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Hide instead of quitting so the tray icon stays active."""
        event.ignore()
        self.hide()

    def request_quit(self) -> None:
        """Called by the tray Quit action — actually exits."""
        self._poll_timer.stop()
        self.quit_requested.emit()

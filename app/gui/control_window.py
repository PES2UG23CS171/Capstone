"""
PyQt6 control window — the settings panel for the audio filter.

Provides:
    • On / Off toggle for noise suppression
    • Strength slider (0 – 100 %)
    • Output gain slider (-12 … +12 dB)
    • Input / output device combo-boxes
    • Real-time input & output level meters
    • Status bar with x-run counter

The window is designed to be shown / hidden from the system tray and does
**not** terminate the application when closed — it simply hides itself.
"""

from __future__ import annotations

import queue
from typing import List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QFont, QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QStatusBar,
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


class LevelMeter(QProgressBar):
    """Compact horizontal level-meter (dBFS)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setRange(-60, 0)
        self.setValue(-60)
        self.setTextVisible(False)
        self.setFixedHeight(14)
        self.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                background: #1e1e1e;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #22c55e, stop:0.7 #facc15, stop:1.0 #ef4444
                );
                border-radius: 2px;
            }
            """
        )

    def set_level(self, db: float) -> None:
        self.setValue(max(-60, min(0, int(db))))


class ControlWindow(QMainWindow):
    """Main settings window shown from the system tray."""

    # Emitted when the user explicitly quits (File → Quit or tray Quit).
    quit_requested = pyqtSignal()

    def __init__(
        self,
        engine: AudioEngineHandle,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine

        self.setWindowTitle("Transient Noise Suppressor")
        self.setMinimumWidth(420)
        self.setMinimumHeight(380)

        self._build_ui()
        self._connect_signals()

        # Poll engine events at ~20 fps
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_engine)
        self._poll_timer.start()

        # Request initial device list
        self._engine.send(Command(CmdType.GET_DEVICES))

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # ── On/Off toggle + RTF label ─────────────────────────────────────
        toggle_row = QHBoxLayout()
        self.btn_toggle = QPushButton("Suppression: ON")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(True)
        self.btn_toggle.setMinimumHeight(40)
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        self.btn_toggle.setFont(font)
        self._style_toggle(True)
        toggle_row.addWidget(self.btn_toggle)

        self.lbl_rtf = QLabel("RTF: —")
        rtf_font = QFont()
        rtf_font.setPointSize(9)
        self.lbl_rtf.setFont(rtf_font)
        self.lbl_rtf.setStyleSheet("color: #22c55e; padding-left: 8px;")
        self.lbl_rtf.setFixedWidth(180)
        self.lbl_rtf.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        toggle_row.addWidget(self.lbl_rtf)
        root.addLayout(toggle_row)

        # ── Strength slider ──────────────────────────────────────────────
        grp_strength = QGroupBox("Suppression Strength")
        lay_strength = QHBoxLayout(grp_strength)
        self.slider_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_strength.setRange(0, 100)
        self.slider_strength.setValue(100)
        self.slider_strength.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_strength.setTickInterval(10)
        self.lbl_strength = QLabel("100 %")
        self.lbl_strength.setFixedWidth(48)
        self.lbl_strength.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay_strength.addWidget(self.slider_strength)
        lay_strength.addWidget(self.lbl_strength)
        root.addWidget(grp_strength)

        # ── Gain slider ─────────────────────────────────────────────────
        grp_gain = QGroupBox("Output Gain")
        lay_gain = QHBoxLayout(grp_gain)
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(-120, 120)   # tenths of dB
        self.slider_gain.setValue(0)
        self.slider_gain.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_gain.setTickInterval(30)
        self.lbl_gain = QLabel("0.0 dB")
        self.lbl_gain.setFixedWidth(60)
        self.lbl_gain.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay_gain.addWidget(self.slider_gain)
        lay_gain.addWidget(self.lbl_gain)
        root.addWidget(grp_gain)

        # ── Device selectors ─────────────────────────────────────────────
        grp_dev = QGroupBox("Audio Devices")
        lay_dev = QVBoxLayout(grp_dev)

        lay_in = QHBoxLayout()
        lay_in.addWidget(QLabel("Input:"))
        self.combo_input = QComboBox()
        self.combo_input.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        lay_in.addWidget(self.combo_input, 1)
        lay_dev.addLayout(lay_in)

        lay_out = QHBoxLayout()
        lay_out.addWidget(QLabel("Output:"))
        self.combo_output = QComboBox()
        self.combo_output.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        lay_out.addWidget(self.combo_output, 1)
        lay_dev.addLayout(lay_out)

        root.addWidget(grp_dev)

        # ── Level meters ────────────────────────────────────────────────
        grp_meters = QGroupBox("Levels (dBFS)")
        lay_meters = QVBoxLayout(grp_meters)

        lay_m_in = QHBoxLayout()
        lay_m_in.addWidget(QLabel("In "))
        self.meter_in = LevelMeter()
        lay_m_in.addWidget(self.meter_in, 1)
        self.lbl_in_db = QLabel("-∞")
        self.lbl_in_db.setFixedWidth(48)
        self.lbl_in_db.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay_m_in.addWidget(self.lbl_in_db)
        lay_meters.addLayout(lay_m_in)

        lay_m_out = QHBoxLayout()
        lay_m_out.addWidget(QLabel("Out"))
        self.meter_out = LevelMeter()
        lay_m_out.addWidget(self.meter_out, 1)
        self.lbl_out_db = QLabel("-∞")
        self.lbl_out_db.setFixedWidth(48)
        self.lbl_out_db.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay_m_out.addWidget(self.lbl_out_db)
        lay_meters.addLayout(lay_m_out)

        root.addWidget(grp_meters)

        # ── Proof of Concept button ──────────────────────────────────────
        self.btn_poc = QPushButton("⚡  Proof of Concept")
        self.btn_poc.setMinimumHeight(36)
        poc_font = QFont()
        poc_font.setPointSize(11)
        poc_font.setBold(True)
        self.btn_poc.setFont(poc_font)
        self.btn_poc.setStyleSheet(
            "QPushButton { background-color: #2563eb; color: white; border-radius: 6px; }"
            "QPushButton:hover { background-color: #3b82f6; }"
        )
        self.btn_poc.clicked.connect(self._open_waveform_viewer)
        root.addWidget(self.btn_poc)

        # ── Status bar ──────────────────────────────────────────────────
        self.statusBar().showMessage("Engine starting…")

        # ── Waveform viewer reference ────────────────────────────────────
        self._waveform_viewer = None

    # ── Styling ──────────────────────────────────────────────────────────

    def _style_toggle(self, on: bool) -> None:
        if on:
            self.btn_toggle.setStyleSheet(
                "QPushButton { background-color: #22c55e; color: white; border-radius: 6px; }"
            )
        else:
            self.btn_toggle.setStyleSheet(
                "QPushButton { background-color: #ef4444; color: white; border-radius: 6px; }"
            )

    # ── Signal wiring ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.btn_toggle.toggled.connect(self._on_toggle)
        self.slider_strength.valueChanged.connect(self._on_strength)
        self.slider_gain.valueChanged.connect(self._on_gain)
        self.combo_input.currentIndexChanged.connect(self._on_input_device)
        self.combo_output.currentIndexChanged.connect(self._on_output_device)

    # ── Slots ────────────────────────────────────────────────────────────

    def _on_toggle(self, checked: bool) -> None:
        self.btn_toggle.setText(f"Suppression: {'ON' if checked else 'OFF'}")
        self._style_toggle(checked)
        self._engine.send(Command(CmdType.SET_ENABLED, checked))

    def _on_strength(self, value: int) -> None:
        pct = value / 100.0
        self.lbl_strength.setText(f"{value} %")
        self._engine.send(Command(CmdType.SET_STRENGTH, pct))

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

    # ── Engine event polling ─────────────────────────────────────────────

    def _poll_engine(self) -> None:
        for evt in self._engine.poll_events():
            if evt.kind == EvtType.STATUS:
                self._handle_status(evt.payload)
            elif evt.kind == EvtType.DEVICE_LIST:
                self._handle_device_list(evt.payload)
            elif evt.kind == EvtType.ERROR:
                self.statusBar().showMessage(f"⚠  {evt.payload}")
            elif evt.kind == EvtType.ENGINE_STOPPED:
                self.statusBar().showMessage("Engine stopped.")

    def _handle_status(self, s: StatusPayload) -> None:
        self.meter_in.set_level(s.input_level_db)
        self.meter_out.set_level(s.output_level_db)

        def _fmt(db: float) -> str:
            return "-∞" if db <= -120 else f"{db:.0f}"

        self.lbl_in_db.setText(_fmt(s.input_level_db))
        self.lbl_out_db.setText(_fmt(s.output_level_db))

        # Update RTF label
        if s.rtf > 0:
            headroom = 1.0 / s.rtf if s.rtf > 0 else 9999
            self.lbl_rtf.setText(f"RTF: {s.rtf:.4f}  ({headroom:.0f}× headroom)")
        else:
            self.lbl_rtf.setText("RTF: —")

        xr = f"  |  x-runs: {s.xruns}" if s.xruns else ""
        self.statusBar().showMessage(f"Engine running{xr}")

    def _handle_device_list(self, devices: List[DeviceInfo]) -> None:
        # Block signals while repopulating
        self.combo_input.blockSignals(True)
        self.combo_output.blockSignals(True)

        self.combo_input.clear()
        self.combo_output.clear()

        self.combo_input.addItem("(System Default)", None)
        self.combo_output.addItem("(System Default)", None)

        default_in_idx = 0
        default_out_idx = 0

        for d in devices:
            if d.max_input_channels > 0:
                label = f"{d.name}  ({d.max_input_channels}ch)"
                self.combo_input.addItem(label, d.index)
                if d.is_default_input:
                    default_in_idx = self.combo_input.count() - 1

            if d.max_output_channels > 0:
                label = f"{d.name}  ({d.max_output_channels}ch)"
                self.combo_output.addItem(label, d.index)
                if d.is_default_output:
                    default_out_idx = self.combo_output.count() - 1

        self.combo_input.setCurrentIndex(default_in_idx)
        self.combo_output.setCurrentIndex(default_out_idx)

        self.combo_input.blockSignals(False)
        self.combo_output.blockSignals(False)

    # ── Waveform viewer ───────────────────────────────────────────────────

    def _open_waveform_viewer(self) -> None:
        """Open the Proof of Concept waveform viewer window."""
        from app.gui.waveform_viewer import WaveformViewer

        if self._waveform_viewer is not None and self._waveform_viewer.isVisible():
            self._waveform_viewer.raise_()
            self._waveform_viewer.activateWindow()
            return

        self._waveform_viewer = WaveformViewer(parent=None)
        self._waveform_viewer.show()

    # ── Window behaviour ─────────────────────────────────────────────────

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Hide instead of quitting so the tray icon stays active."""
        event.ignore()
        self.hide()

    def request_quit(self) -> None:
        """Called by the tray Quit action — actually exits."""
        self._poll_timer.stop()
        self.quit_requested.emit()

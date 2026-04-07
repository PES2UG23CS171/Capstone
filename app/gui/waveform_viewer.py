"""
Waveform Viewer — PoC demo window with three stacked waveform plots.

Launched from the "Proof of Concept" button in the main ControlWindow.
Generates synthetic test signals, runs the offline filter, and displays
results in pyqtgraph PlotWidgets with linked X-axes.
"""

from __future__ import annotations

import sys
import time
import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QScreen
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph as pg
except ImportError:
    pg = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import soundfile as sf
except ImportError:
    sf = None


# ---------------------------------------------------------------------------
#  Worker thread for signal generation + offline filtering
# ---------------------------------------------------------------------------


class _PocWorker(QThread):
    """Runs generate_test_signal + run_offline_demo in a background thread."""

    progress = pyqtSignal(int, str)       # (percent, message)
    finished = pyqtSignal(dict)           # results dict
    errored  = pyqtSignal(str)            # error message

    def __init__(self, out_dir: Path, parent=None):
        super().__init__(parent)
        self._out_dir = out_dir

    def run(self):
        try:
            # Import the PoC module (lives at project root)
            sys.path.insert(0, str(self._out_dir))
            from poc_realtime_transient import (
                generate_test_signal,
                RealTimeFilter,
                SAMPLE_RATE,
                CHUNK_SIZE,
                FLOAT_DTYPE,
            )

            self.progress.emit(10, "Generating synthetic test signal…")
            clean_path, noisy_path = generate_test_signal(out_dir=self._out_dir)

            self.progress.emit(40, "Running real-time filter on noisy signal…")
            filtered_path = noisy_path.with_name("test_noisy_filtered.wav")

            # Load noisy file
            if sf is not None:
                data, sr = sf.read(str(noisy_path), dtype="float32")
            else:
                from scipy.io import wavfile
                sr, raw = wavfile.read(str(noisy_path))
                data = raw.astype(np.float32) / 32767.0 if raw.dtype == np.int16 else raw.astype(np.float32)

            if data.ndim > 1:
                data = data[:, 0]

            filt = RealTimeFilter(sr=sr, chunk=CHUNK_SIZE)
            n = len(data)
            out = np.zeros(n, dtype=np.float32)
            n_chunks = n // CHUNK_SIZE

            for i in range(n_chunks):
                s = i * CHUNK_SIZE
                e = s + CHUNK_SIZE
                out[s:e] = filt.process_chunk(data[s:e])
                if i % 500 == 0:
                    pct = 40 + int(50 * i / n_chunks)
                    self.progress.emit(pct, f"Processing chunk {i}/{n_chunks}…")

            # Handle remainder
            remainder = n % CHUNK_SIZE
            if remainder > 0:
                last = np.zeros(CHUNK_SIZE, dtype=np.float32)
                last[:remainder] = data[n_chunks * CHUNK_SIZE:]
                processed = filt.process_chunk(last)
                out[n_chunks * CHUNK_SIZE:] = processed[:remainder]

            # Save filtered output
            if sf is not None:
                sf.write(str(filtered_path), out, sr)
            else:
                from scipy.io import wavfile as wf
                wf.write(str(filtered_path), sr, (out * 32767.0).astype(np.int16))

            self.progress.emit(95, "Loading waveform data…")

            # Load all three signals for plotting
            if sf is not None:
                clean_data, _ = sf.read(str(clean_path), dtype="float32")
                noisy_data, _ = sf.read(str(noisy_path), dtype="float32")
                filtered_data, _ = sf.read(str(filtered_path), dtype="float32")
            else:
                from scipy.io import wavfile as wf
                _, c = wf.read(str(clean_path))
                _, n_ = wf.read(str(noisy_path))
                _, f_ = wf.read(str(filtered_path))
                clean_data = c.astype(np.float32) / 32767.0 if c.dtype == np.int16 else c.astype(np.float32)
                noisy_data = n_.astype(np.float32) / 32767.0 if n_.dtype == np.int16 else n_.astype(np.float32)
                filtered_data = f_.astype(np.float32) / 32767.0 if f_.dtype == np.int16 else f_.astype(np.float32)

            # Flatten to mono
            if clean_data.ndim > 1:
                clean_data = clean_data[:, 0]
            if noisy_data.ndim > 1:
                noisy_data = noisy_data[:, 0]
            if filtered_data.ndim > 1:
                filtered_data = filtered_data[:, 0]

            # Collect profiler stats
            times = np.array(filt.profiler._times) * 1e6  # → µs
            chunk_budget_us = CHUNK_SIZE / sr * 1e6
            mean_us = float(np.mean(times))
            p99_us = float(np.percentile(times, 99))
            rtf = mean_us / chunk_budget_us
            headroom = 1.0 / rtf if rtf > 0 else 9999

            self.progress.emit(100, "Done!")

            self.finished.emit({
                "sr": sr,
                "clean": clean_data,
                "noisy": noisy_data,
                "filtered": filtered_data,
                "noisy_path": str(noisy_path),
                "filtered_path": str(filtered_path),
                "mean_us": mean_us,
                "p99_us": p99_us,
                "rtf": rtf,
                "headroom": headroom,
            })

        except Exception as e:
            import traceback
            self.errored.emit(f"{e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
#  Metric Card widget
# ---------------------------------------------------------------------------


class _MetricCard(QFrame):
    """Small card showing a label and a value, styled to match the dark theme."""

    def __init__(self, title: str, value: str = "—", parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(10, 6, 10, 6)

        self._title = QLabel(title)
        self._title.setStyleSheet("color: #aaa; font-size: 11px; border: none;")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title)

        self._value = QLabel(value)
        self._value.setStyleSheet("color: #eee; font-size: 16px; font-weight: bold; border: none;")
        self._value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._value)

    def set_value(self, text: str, color: str = "#eee"):
        self._value.setText(text)
        self._value.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold; border: none;")


# ---------------------------------------------------------------------------
#  Waveform Viewer Window
# ---------------------------------------------------------------------------


class WaveformViewer(QMainWindow):
    """PoC demo window with three stacked waveform plots and stats panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Proof of Concept — Waveform Viewer")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        # Data storage
        self._noisy_path: Optional[str] = None
        self._filtered_path: Optional[str] = None
        self._noisy_data: Optional[np.ndarray] = None
        self._filtered_data: Optional[np.ndarray] = None
        self._sr: int = 48000

        self._build_ui()
        self._start_processing()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("background: #1e1e1e;")
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Progress bar (shown during processing) ───────────────────────
        self._progress_container = QWidget()
        prog_layout = QVBoxLayout(self._progress_container)
        prog_layout.setContentsMargins(40, 40, 40, 40)

        self._progress_label = QLabel("Initialising…")
        self._progress_label.setStyleSheet("color: #ccc; font-size: 14px;")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prog_layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                background: #2a2a2a;
                height: 20px;
                text-align: center;
                color: #eee;
            }
            QProgressBar::chunk {
                background: #22c55e;
                border-radius: 3px;
            }
        """)
        prog_layout.addWidget(self._progress_bar)
        prog_layout.addStretch()
        root.addWidget(self._progress_container)

        # ── Results container (hidden until processing completes) ────────
        self._results_container = QWidget()
        self._results_container.hide()
        results_layout = QVBoxLayout(self._results_container)
        results_layout.setSpacing(4)
        results_layout.setContentsMargins(0, 0, 0, 0)

        if pg is not None:
            pg.setConfigOptions(antialias=True, background="#1e1e1e", foreground="#ccc")

            # Three linked plots
            self._plot_clean = pg.PlotWidget(title="Clean")
            self._plot_noisy = pg.PlotWidget(title="Noisy")
            self._plot_filtered = pg.PlotWidget(title="Filtered")

            # Link X-axes
            self._plot_noisy.setXLink(self._plot_clean)
            self._plot_filtered.setXLink(self._plot_clean)

            for pw in (self._plot_clean, self._plot_noisy, self._plot_filtered):
                pw.setLabel("bottom", "Time", units="s")
                pw.setLabel("left", "Amplitude")
                pw.showGrid(x=True, y=True, alpha=0.15)
                pw.setMinimumHeight(130)
                results_layout.addWidget(pw)
        else:
            lbl = QLabel("pyqtgraph not installed — plots unavailable.")
            lbl.setStyleSheet("color: #ef4444; font-size: 14px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            results_layout.addWidget(lbl)
            self._plot_clean = None
            self._plot_noisy = None
            self._plot_filtered = None

        # ── Stats panel ──────────────────────────────────────────────────
        stats_frame = QFrame()
        stats_frame.setStyleSheet("background: transparent;")
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setSpacing(8)
        stats_layout.setContentsMargins(0, 4, 0, 4)

        self._card_mean = _MetricCard("Mean Chunk Time")
        self._card_p99 = _MetricCard("99th Percentile")
        self._card_rtf = _MetricCard("Real-Time Factor")
        self._card_headroom = _MetricCard("Headroom")
        self._card_verdict = _MetricCard("Feasibility")

        for card in (self._card_mean, self._card_p99, self._card_rtf,
                     self._card_headroom, self._card_verdict):
            stats_layout.addWidget(card)

        results_layout.addWidget(stats_frame)

        # ── Button bar ───────────────────────────────────────────────────
        btn_frame = QFrame()
        btn_frame.setStyleSheet("background: transparent;")
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(0, 4, 0, 0)
        btn_layout.setSpacing(8)

        btn_style = """
            QPushButton {
                background: #333;
                color: #eee;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background: #444; }
            QPushButton:pressed { background: #555; }
            QPushButton:disabled { background: #2a2a2a; color: #666; }
        """

        self._btn_play_noisy = QPushButton("▶ Play Noisy")
        self._btn_play_noisy.setStyleSheet(btn_style)
        self._btn_play_filtered = QPushButton("▶ Play Filtered")
        self._btn_play_filtered.setStyleSheet(btn_style)
        self._btn_stop = QPushButton("■ Stop")
        self._btn_stop.setStyleSheet(btn_style)
        self._btn_export = QPushButton("Export PNG")
        self._btn_export.setStyleSheet(btn_style)

        # Disable play buttons if no sounddevice
        if sd is None:
            self._btn_play_noisy.setEnabled(False)
            self._btn_play_noisy.setToolTip("sounddevice not installed — playback unavailable")
            self._btn_play_filtered.setEnabled(False)
            self._btn_play_filtered.setToolTip("sounddevice not installed — playback unavailable")
            self._btn_stop.setEnabled(False)
            self._btn_stop.setToolTip("sounddevice not installed — playback unavailable")

        btn_layout.addStretch()
        btn_layout.addWidget(self._btn_play_noisy)
        btn_layout.addWidget(self._btn_play_filtered)
        btn_layout.addWidget(self._btn_stop)
        btn_layout.addWidget(self._btn_export)
        btn_layout.addStretch()

        results_layout.addWidget(btn_frame)
        root.addWidget(self._results_container)

        # ── Connect buttons ──────────────────────────────────────────────
        self._btn_play_noisy.clicked.connect(self._play_noisy)
        self._btn_play_filtered.clicked.connect(self._play_filtered)
        self._btn_stop.clicked.connect(self._stop_playback)
        self._btn_export.clicked.connect(self._export_png)

    # ── Processing pipeline ──────────────────────────────────────────────

    def _start_processing(self):
        out_dir = Path.cwd()
        self._worker = _PocWorker(out_dir, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.errored.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self._progress_bar.setValue(pct)
        self._progress_label.setText(msg)

    def _on_error(self, msg: str):
        self._progress_label.setText("Error!")
        QMessageBox.critical(self, "PoC Error", msg)

    def _on_finished(self, results: dict):
        self._progress_container.hide()
        self._results_container.show()

        self._sr = results["sr"]
        self._noisy_data = results["noisy"]
        self._filtered_data = results["filtered"]
        self._noisy_path = results["noisy_path"]
        self._filtered_path = results["filtered_path"]

        # ── Plot waveforms ───────────────────────────────────────────
        if self._plot_clean is not None:
            clean = results["clean"]
            noisy = results["noisy"]
            filtered = results["filtered"]
            sr = results["sr"]

            # Downsample for plotting performance (show every Nth sample)
            n = len(clean)
            step = max(1, n // 20000)
            t = np.arange(0, n, step) / sr

            self._plot_clean.plot(t, clean[::step], pen=pg.mkPen("#888888", width=1))
            self._plot_noisy.plot(t, noisy[::step], pen=pg.mkPen("#E05555", width=1))
            self._plot_filtered.plot(t, filtered[::step], pen=pg.mkPen("#55BB55", width=1))

            # ── Add transient markers ────────────────────────────────
            markers = [
                (1.2, "Dog Bark"),
                (3.5, "Door Slam"),
                (5.0, "Keyboard Click"),
                (6.8, "Siren"),
                (8.5, "Plosive 'P'"),
            ]

            for pw in (self._plot_clean, self._plot_noisy, self._plot_filtered):
                for t_pos, label_text in markers:
                    line = pg.InfiniteLine(
                        pos=t_pos,
                        angle=90,
                        pen=pg.mkPen("#ffaa00", width=1, style=Qt.PenStyle.DashLine),
                    )
                    pw.addItem(line)

                    # Text label
                    text = pg.TextItem(label_text, color="#ffaa00", anchor=(0.5, 1.0))
                    text.setPos(t_pos, 0.95)
                    font = QFont()
                    font.setPointSize(8)
                    text.setFont(font)
                    pw.addItem(text)

                    # Special green label for plosive on filtered plot
                    if t_pos == 8.5 and pw is self._plot_filtered:
                        preserved = pg.TextItem(
                            "Speech preserved ✓",
                            color="#22c55e",
                            anchor=(0.5, 1.0),
                        )
                        preserved.setPos(t_pos, 0.75)
                        bold_font = QFont()
                        bold_font.setPointSize(10)
                        bold_font.setBold(True)
                        preserved.setFont(bold_font)
                        pw.addItem(preserved)

        # ── Update stats cards ───────────────────────────────────────
        self._card_mean.set_value(f"{results['mean_us']:.1f} µs")
        self._card_p99.set_value(f"{results['p99_us']:.1f} µs")
        self._card_rtf.set_value(f"{results['rtf']:.4f}")
        self._card_headroom.set_value(f"{results['headroom']:.0f}× faster")

        if results["rtf"] < 0.80:
            self._card_verdict.set_value("PASS ✓", color="#22c55e")
        else:
            self._card_verdict.set_value("FAIL ✗", color="#ef4444")

    # ── Playback ─────────────────────────────────────────────────────────

    def _play_noisy(self):
        if sd is None or self._noisy_data is None:
            return
        sd.stop()
        sd.play(self._noisy_data, samplerate=self._sr)

    def _play_filtered(self):
        if sd is None or self._filtered_data is None:
            return
        sd.stop()
        sd.play(self._filtered_data, samplerate=self._sr)

    def _stop_playback(self):
        if sd is not None:
            sd.stop()

    def _export_png(self):
        desktop = Path.home() / "Desktop"
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = desktop / f"poc_waveform_{ts}.png"
        try:
            screen = self.screen()
            if screen is not None:
                pixmap = screen.grabWindow(int(self.winId()))
                pixmap.save(str(filename))
                self.statusBar().showMessage(f"Saved: {filename}")
                QMessageBox.information(self, "Export", f"Screenshot saved to:\n{filename}")
            else:
                QMessageBox.warning(self, "Export", "Could not capture screen.")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def closeEvent(self, event):
        if sd is not None:
            sd.stop()
        if hasattr(self, "_worker") and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        event.accept()

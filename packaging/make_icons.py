"""Regenerate packaging/icon.icns and icon.ico (macOS only, needs iconutil).

    PYTHONPATH=. QT_QPA_PLATFORM=offscreen .venv_poc/bin/python packaging/make_icons.py
"""
import struct
import subprocess
import tempfile
from pathlib import Path

from PyQt6.QtCore import QBuffer, QIODevice, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen

HERE = Path(__file__).resolve().parent


def render(size: int) -> QImage:
    """The tray's microphone glyph on the app's dark rounded square."""
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    u = size / 100.0

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#14161a"))
    p.drawRoundedRect(QRectF(4 * u, 4 * u, 92 * u, 92 * u), 22 * u, 22 * u)

    green = QColor(34, 197, 94)
    p.setBrush(green)
    p.drawRoundedRect(QRectF(38 * u, 18 * u, 24 * u, 42 * u), 12 * u, 12 * u)
    pen = QPen(green, 6 * u)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRectF(28 * u, 30 * u, 44 * u, 40 * u), 180 * 16, 180 * 16)
    p.drawLine(QPointF(50 * u, 70 * u), QPointF(50 * u, 82 * u))
    p.drawLine(QPointF(36 * u, 82 * u), QPointF(64 * u, 82 * u))
    p.end()
    return img


def png_bytes(img: QImage) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def dib_bytes(img: QImage) -> bytes:
    """Uncompressed ICO entry, which every Windows tool accepts."""
    s = img.width()
    bits = img.constBits()
    bits.setsize(img.sizeInBytes())
    raw = bytes(bits)
    stride = img.bytesPerLine()
    pixels = b"".join(raw[y * stride: y * stride + s * 4] for y in range(s - 1, -1, -1))
    mask = bytes(((s + 31) // 32) * 4 * s)      # all opaque: alpha channel decides
    header = struct.pack("<IiiHHIIiiII", 40, s, s * 2, 1, 32, 0, len(pixels) + len(mask),
                         0, 0, 0, 0)
    return header + pixels + mask


def write_ico(path: Path) -> None:
    entries = [(s, dib_bytes(render(s))) for s in (16, 24, 32, 48, 64)]
    entries.append((256, png_bytes(render(256))))       # 256 px entries are PNG by convention
    out = struct.pack("<HHH", 0, 1, len(entries))
    offset = 6 + 16 * len(entries)
    for s, data in entries:
        dim = 0 if s == 256 else s
        out += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    path.write_bytes(out + b"".join(d for _, d in entries))


def write_icns(path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for s in (16, 32, 128, 256, 512):
            render(s).save(str(iconset / f"icon_{s}x{s}.png"))
            render(s * 2).save(str(iconset / f"icon_{s}x{s}@2x.png"))
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(path)], check=True)


if __name__ == "__main__":
    app = QGuiApplication([])
    write_ico(HERE / "icon.ico")
    write_icns(HERE / "icon.icns")
    render(512).save(str(Path(tempfile.gettempdir()) / "voiceiso_icon_preview.png"))
    print("wrote", HERE / "icon.ico", "and", HERE / "icon.icns")

# PyInstaller recipe for VoiceISO.  Build with packaging/build_macos.sh or
# packaging/build_windows.ps1, on the platform being built for: PyInstaller
# cannot cross-compile.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).parent
VERSION = (ROOT / "packaging" / "VERSION").read_text().strip()

# The deployed classifier only.  Other heads in checkpoints/ are research
# artifacts (v4 stays available to source checkouts via VOICEISO_HEAD).
HEAD = "efficientat_head12_v3.onnx"
datas = [
    (str(ROOT / "checkpoints" / HEAD), "checkpoints"),
    (str(ROOT / "checkpoints" / f"{HEAD}.data"), "checkpoints"),
]

# DFN3 weights, so that first launch needs no internet.  Only these two paths:
# the cache also holds a log file.  voiceiso/config.py finds them at models/.
sys.path.insert(0, str(ROOT))
import voiceiso._compat  # noqa: E402,F401  (must precede df)
from df.enhance import maybe_download_model  # noqa: E402

model = Path(maybe_download_model("DeepFilterNet3"))
datas.append((str(model / "config.ini"), "models/DeepFilterNet3"))
datas.append((str(model / "checkpoints"), "models/DeepFilterNet3/checkpoints"))

# Silero VAD loads its ONNX model as package data.
datas += collect_data_files("silero_vad")

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    datas=datas,
    # df loads the model code by name at runtime: import_module("df." + model).
    hiddenimports=["df.deepfilternet3"],
    excludes=[
        # Research, plotting, and dev tools that are never imported at runtime.
        "IPython", "jupyter", "matplotlib", "notebook", "pandas", "pytest",
        "tensorboard", "tkinter", "torchvision", "librosa", "numba", "sklearn",
        "pesq", "pystoi", "onnx", "pyqtgraph", "PIL",
        # DeepFilterNet declares sympy, but enhancement never uses it.
        "sympy", "mpmath",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

is_mac = sys.platform == "darwin"
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VoiceISO",
    console=False,
    upx=False,                                  # UPX breaks torch and Qt libraries
    target_arch="arm64" if is_mac else None,    # torch has no Intel Mac builds
    icon=str(ROOT / "packaging" / ("icon.icns" if is_mac else "icon.ico")),
)
coll = COLLECT(exe, a.binaries, a.datas, upx=False, name="VoiceISO")

if is_mac:
    app = BUNDLE(
        coll,
        name="VoiceISO.app",
        icon=str(ROOT / "packaging" / "icon.icns"),
        bundle_identifier="edu.pes.team190.voiceiso",
        version=VERSION,
        info_plist={
            "CFBundleDisplayName": "VoiceISO",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            # Without this string macOS refuses mic access with no prompt at all.
            "NSMicrophoneUsageDescription":
                "VoiceISO removes background noise from your microphone. "
                "Audio is processed on this Mac and never leaves it.",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "13.0",
        },
    )

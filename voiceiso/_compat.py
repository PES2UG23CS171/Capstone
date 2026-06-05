"""
Compatibility shims so DeepFilterNet 0.5.6 imports under modern torch stacks.

DFN 0.5.6's ``df/io.py`` does ``from torchaudio.backend.common import
AudioMetaData``, a path removed in torchaudio ≥ 2.1.  We register a stand-in
module *before* ``df`` is imported anywhere.  Importing this module has the
side effect of installing the shim; import it first.
"""

from __future__ import annotations

import sys
import types


def install_torchaudio_shim() -> None:
    try:
        import torchaudio
    except Exception:
        return
    AudioMetaData = getattr(torchaudio, "AudioMetaData", None)
    if AudioMetaData is None:
        class AudioMetaData:  # minimal placeholder — DFN only needs the symbol
            def __init__(self, *a, **k) -> None:  # noqa: D401
                pass
    for modname in ("torchaudio.backend", "torchaudio.backend.common"):
        if modname not in sys.modules:
            sys.modules[modname] = types.ModuleType(modname)
    sys.modules["torchaudio.backend.common"].AudioMetaData = AudioMetaData


install_torchaudio_shim()

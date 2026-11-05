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


def disable_df_git_calls() -> None:
    """Stop ``df.init_df`` from shelling out to ``git``.  Call after importing df.

    Its logger runs ``git rev-parse`` just to print a commit hash, and only
    catches ``CalledProcessError``.  On a machine without git that raises
    ``FileNotFoundError``, ``init_df`` fails, and the whole product silently
    drops to no-denoising passthrough (reproduced with git hidden from PATH).
    On a Mac without developer tools, ``/usr/bin/git`` can pop up an Xcode
    install dialog instead.
    """
    import df.logger

    df.logger.get_commit_hash = lambda: None
    df.logger.get_branch_name = lambda: None


install_torchaudio_shim()

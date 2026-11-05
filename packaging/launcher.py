"""Entry point of the packaged app (see packaging/voiceiso.spec)."""

import multiprocessing
import os
import sys
from pathlib import Path


def _log_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "VoiceISO" / "voiceiso.log"
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "VoiceISO" / "voiceiso.log"


def _redirect_output_to_log() -> None:
    """A windowed app has no console: stdout and stderr can be None.

    DeepFilterNet's logger raises when handed a None stream, which would disable
    denoising, so send everything to a file (it also gives users something to
    send when asking for help).  Spawned helper processes append; only the main
    process starts a fresh log.
    """
    is_helper = "--multiprocessing-fork" in sys.argv or "-c" in sys.argv
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = open(path, "a" if is_helper else "w", buffering=1,
                      encoding="utf-8", errors="replace")
    except OSError:
        stream = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = sys.stderr = stream


if __name__ == "__main__":
    # A self-test run from a terminal prints its report there.  Windowed Windows
    # builds have no stdout at all and rely on --self-test-out for the report.
    if "--self-test" not in sys.argv or sys.stdout is None or sys.stderr is None:
        _redirect_output_to_log()
    multiprocessing.freeze_support()    # spawned children stop here and run their target

    if "--self-test" in sys.argv:
        from app.selftest import main as self_test
        self_test(sys.argv)
    else:
        from app.main import main
        main()

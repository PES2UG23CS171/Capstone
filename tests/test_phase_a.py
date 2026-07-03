"""
Phase-A regression tests (audit fixes C1–C4).

Dependency-free: runs under plain ``python -m tests.test_phase_a`` (prints
PASS/SKIP per check) and is also collected by pytest if installed.  Tests that
need onnxruntime + the shipped head ONNX skip cleanly when those are absent, so
the suite never fails on a machine without the checkpoint.
"""
from __future__ import annotations

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages.base import FrameContext


class _Skip(Exception):
    pass


def _has_efficientat(cfg: PipelineConfig) -> bool:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return bool(cfg.noise_classifier_model_path)


# ── C1: config auto-wires the head and the window matches the model ──────────
def test_config_autowires_head_and_window():
    cfg = PipelineConfig()
    # window default must be the 4 s the shipped head was exported with.
    assert cfg.noise_classifier_window_s == 4.0
    # If a head checkpoint exists, the path must be auto-resolved (absolute) to
    # one of the known head ONNX files (v2 preferred over v1).
    from pathlib import Path
    ckdir = Path(__file__).resolve().parent.parent / "checkpoints"
    if any((ckdir / f).exists() for f in ("efficientat_head12_v2.onnx", "efficientat_head12.onnx")):
        p = cfg.noise_classifier_model_path
        assert p is not None and Path(p).is_absolute()
        assert Path(p).name in ("efficientat_head12_v2.onnx", "efficientat_head12.onnx")
        # v2 must win when present
        if (ckdir / "efficientat_head12_v2.onnx").exists():
            assert Path(p).name == "efficientat_head12_v2.onnx"


# ── C1: classifier reads required frames from ONNX and emits a real label ────
def test_classifier_reads_frames_and_emits_label():
    cfg = PipelineConfig()
    if not _has_efficientat(cfg):
        raise _Skip("onnxruntime / head ONNX unavailable")
    from voiceiso.stages.noise_classifier import EfficientATNoiseClassifier
    clf = EfficientATNoiseClassifier(cfg)
    assert clf._req_frames == 400, clf._req_frames
    # Feed > window seconds of a 220 Hz tone (music-ish) and confirm it neither
    # crashes nor stays stuck at all-zero / 'clean'.
    sr = cfg.sample_rate
    t = np.arange(int(sr * 5.0)) / sr
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    clf.reset()
    block = 960
    last = None
    for s in range(0, (len(tone) // block) * block, block):
        ctx = FrameContext(audio=tone[s:s + block].copy(), sample_rate=sr)
        ctx.meta = {}
        ctx = clf.process(ctx)
        last = ctx.meta.get("noise_label")
    assert last is not None
    # at least one non-clean target accumulated probability
    assert max(v for k, v in clf._smoothed.items() if k != "clean") > 0.0


# ── C1: robust to a window/frame mismatch (no more silent InvalidArgument) ───
def test_classifier_robust_to_short_window():
    cfg = PipelineConfig()
    if not _has_efficientat(cfg):
        raise _Skip("onnxruntime / head ONNX unavailable")
    from voiceiso.stages.noise_classifier import EfficientATNoiseClassifier
    # Deliberately misconfigure to the OLD broken 0.20 s window.
    cfg.noise_classifier_window_s = 0.20
    clf = EfficientATNoiseClassifier(cfg)
    sr = cfg.sample_rate
    noise = (0.1 * np.random.default_rng(0).standard_normal(int(sr * 1.0))).astype(np.float32)
    clf.reset()
    block = 960
    for s in range(0, (len(noise) // block) * block, block):
        ctx = FrameContext(audio=noise[s:s + block].copy(), sample_rate=sr)
        ctx.meta = {}
        clf.process(ctx)        # must NOT raise; pad/crop handles 20→400 frames
    assert clf._warned_infer is False, "pad/crop should prevent inference errors"


# ── C4: bad model path falls back to heuristic VISIBLY ───────────────────────
def test_bad_path_fallback_is_visible():
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        raise _Skip("onnxruntime unavailable")
    from voiceiso.stages.noise_classifier import EfficientATNoiseClassifier
    cfg = PipelineConfig(noise_classifier_model_path="/nonexistent/model.onnx")
    raised = False
    try:
        EfficientATNoiseClassifier(cfg)
    except RuntimeError as exc:
        raised = True
        assert "not set or missing" in str(exc)
    assert raised, "missing model must raise (so the pipeline can fall back loudly)"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    npass = nskip = nfail = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            npass += 1
        except _Skip as s:
            print(f"SKIP  {fn.__name__}  ({s})")
            nskip += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}  -> {type(e).__name__}: {e}")
            nfail += 1
    print(f"\n{npass} passed, {nskip} skipped, {nfail} failed")
    return nfail


if __name__ == "__main__":
    import sys
    sys.exit(1 if _run_all() else 0)

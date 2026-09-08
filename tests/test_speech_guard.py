"""
Speech-protection regression tests (2026-09-08 female-voice fix).

Dependency-free runner (``python -m tests.test_speech_guard``), pytest-compatible.
Uses the deterministic fake DeepFilterNet3 backend from ``tests.test_deferred_tail``
(no model weights).  Pins:

  1. the controller's speech cap is wired to ``cfg.supp_speech_floor`` and only
     binds while the VAD is confident (0.92 reproduces the previous behaviour);
  2. the enhancement stage's speech-energy floor bounds how far a confident-
     speech block may be dropped, is inert when the network keeps the speech,
     is fully off at ``speech_floor_db=0``, never touches non-speech blocks,
     ramps its mix per sample, and resets with the stage.
"""
from __future__ import annotations

import numpy as np

from voiceiso.config import PipelineConfig
from voiceiso.stages import enhancement as E
from voiceiso.stages.base import FrameContext
from voiceiso.stages.controller import DynamicController

from tests.test_deferred_tail import _FakeDF, _Skip, SR, B


def _band_energy(x):
    return E.Enhancement._speech_band_energy(np.asarray(x, np.float32), SR)


def _speech_block(seed=0):
    """Harmonic 'vowel' with energy in the speech band."""
    t = np.arange(B) / SR
    x = sum(np.sin(2 * np.pi * 220.0 * k * t + seed) / k for k in range(1, 12))
    return (0.1 * x / np.sqrt(np.mean(x ** 2))).astype(np.float32)


def _mk(gain, **cfg):
    """Enhancement stage with a fake DFN3 that scales the input by ``gain``."""
    def fake_enhance(model, state, x, atten_lim_db=None):
        return x * gain
    with _FakeDF():
        E.enhance = fake_enhance
        enh = E.Enhancement(PipelineConfig(dfn_tail_ms=0.0, **cfg), threads=1)
    assert enh.backend == "deepfilternet3"
    enh._fake_gain = gain
    return enh


def _process(enh, x, vad, supp=1.0):
    def fake_enhance(model, state, x_, atten_lim_db=None):
        return x_ * enh._fake_gain
    with _FakeDF():
        E.enhance = fake_enhance
        ctx = FrameContext(audio=x.copy(), sample_rate=SR)
        ctx.vad_prob = vad
        ctx.suppression = supp
        return enh.process(ctx)


# ── 1. controller speech cap ──────────────────────────────────────────────────
def test_controller_speech_cap_is_wired_and_vad_gated():
    ctrl = DynamicController(PipelineConfig(supp_speech_floor=0.10))
    ctx = FrameContext(audio=np.zeros(B, np.float32), sample_rate=SR)
    ctx.snr_db = 0.0                       # worst SNR → base suppression 1.0
    ctx.noise_class = "fan"; ctx.noise_conf = 0.95
    ctx.vad_prob = 1.0
    assert ctrl._compute_target(ctx) <= 0.10 + 1e-9, "confident speech must be capped at supp_speech_floor"
    ctx.vad_prob = 0.0
    assert ctrl._compute_target(ctx) > 0.5, "no VAD → the cap must not bind"
    ctx.vad_prob = 0.65                     # halfway between 0.4 and 0.9 → cap halfway between 1.0 and 0.10
    assert abs(ctrl._compute_target(ctx) - 0.55) < 1e-6
    legacy = DynamicController(PipelineConfig(supp_speech_floor=0.92))
    ctx.vad_prob = 1.0
    assert abs(legacy._compute_target(ctx) - 0.92) < 1e-9, "0.92 must reproduce the previous behaviour"


# ── 2. speech-energy floor ────────────────────────────────────────────────────
def test_floor_bounds_confident_speech_dropout():
    x = _speech_block()
    enh = _mk(gain=0.01, speech_floor_db=9.0)           # fake DFN3 drops the block by 40 dB
    out = _process(enh, x, vad=1.0).audio
    drop = 10 * np.log10(_band_energy(x) / _band_energy(out))
    assert drop < 9.0 + 1.5, f"speech-band drop {drop:.1f} dB must be bounded at the 9 dB floor (+10 ms ramp)"
    assert 0.0 < enh._floor_mix_prev <= 1.0
    # the repair must be a ramp, not a step: first sample ≈ wet (mix starts at 0),
    # and after the 10 ms ramp the mix holds at its target value
    assert abs(out[0] - 0.01 * x[0]) < 1e-3
    m = enh._floor_mix_prev
    k = 2000
    assert abs(out[k] - ((1 - m) * 0.01 * x[k] + m * x[k])) < 1e-5


def test_floor_is_inert_when_speech_is_kept():
    x = _speech_block()
    enh = _mk(gain=0.7, speech_floor_db=9.0)            # only −3 dB → below the floor
    out = _process(enh, x, vad=1.0).audio
    assert np.allclose(out, 0.7 * x, atol=1e-6), "no repair when the network keeps the speech"
    assert enh._floor_mix_prev == 0.0


def test_floor_ignores_legitimate_noise_removal():
    """A loud noise removed from under the voice: the dry block is far above the
    wet block, but the wet stays at the speaker's recent output level → this is
    NOT a dropout and must not blend noise back."""
    x = _speech_block()
    enh = _mk(gain=1.0, speech_floor_db=9.0)             # fake DFN3 = identity on speech
    for _ in range(12):                                   # establish the wet speech envelope
        _process(enh, x, vad=1.0)
    assert enh._wet_env_db is not None
    # Now a block where the DRY contains speech + a 20 dB louder noise burst, and the
    # network removes the noise: emulate by feeding dry = speech + noise while the
    # fake returns only the speech part (gain applied to a speech-only copy).
    rng = np.random.default_rng(4)
    noise = (1.0 * rng.standard_normal(B)).astype(np.float32)          # ~+20 dB over the speech
    dry = (x + noise).astype(np.float32)
    def fake_enhance(model, state, t, atten_lim_db=None):
        import torch
        y = t.clone()                                                    # full [history | block] segment
        y[:, -B:] = torch.from_numpy(x.reshape(1, -1).copy())            # block region → clean speech
        return y
    with _FakeDF():
        E.enhance = fake_enhance
        ctx = FrameContext(audio=dry.copy(), sample_rate=SR); ctx.vad_prob = 1.0; ctx.suppression = 1.0
        out = enh.process(ctx).audio
    assert ctx.meta["enh_floor_mix"] == 0.0, "legitimate noise removal must not trigger the floor"
    assert np.allclose(out, x, atol=1e-6)


def test_floor_off_and_non_speech_untouched():
    x = _speech_block()
    enh = _mk(gain=0.01, speech_floor_db=0.0)
    out = _process(enh, x, vad=1.0).audio
    assert np.allclose(out, 0.01 * x, atol=1e-7), "speech_floor_db=0 must be fully off"
    enh = _mk(gain=0.01, speech_floor_db=9.0)
    out = _process(enh, x, vad=0.3).audio               # not confident speech
    assert np.allclose(out, 0.01 * x, atol=1e-7), "the floor must never touch non-speech blocks"


def test_floor_mix_ramps_and_resets():
    x = _speech_block()
    enh = _mk(gain=0.01, speech_floor_db=9.0)
    _process(enh, x, vad=1.0)
    m1 = enh._floor_mix_prev
    assert m1 > 0.0
    out2 = _process(enh, x, vad=1.0).audio               # second block: ramp starts at m1
    expect_first = (1 - m1) * 0.01 * x[0] + m1 * x[0]
    assert abs(out2[0] - expect_first) < 1e-4, "the mix must ramp from the previous block's value"
    enh.reset()
    assert enh._floor_mix_prev == 0.0, "reset must clear the guard state"


def test_env_cfg_override():
    import os
    old = os.environ.get("VOICEISO_CFG")
    try:
        os.environ["VOICEISO_CFG"] = "speech_floor_db=9, dfn_history_ms=240,kick_enabled=false,no_such_field=1"
        cfg = PipelineConfig()
        assert cfg.speech_floor_db == 9.0 and cfg.dfn_history_ms == 240.0 and cfg.kick_enabled is False
        assert cfg.supp_speech_floor == 0.20, "untouched fields keep their defaults"
        os.environ["VOICEISO_CFG"] = ""
        assert PipelineConfig().speech_floor_db == 0.0, "empty override → defaults"
    finally:
        if old is None:
            os.environ.pop("VOICEISO_CFG", None)
        else:
            os.environ["VOICEISO_CFG"] = old


def test_guard_is_deterministic():
    x = _speech_block(seed=1)
    outs = []
    for _ in range(2):
        enh = _mk(gain=0.02, speech_floor_db=9.0)
        outs.append(np.concatenate([_process(enh, x, vad=1.0).audio for _ in range(3)]))
    assert np.array_equal(outs[0], outs[1])


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

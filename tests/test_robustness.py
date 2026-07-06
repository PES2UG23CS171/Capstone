"""
Streaming-robustness regression tests (audit fixes C2/C3).

Same conventions as test_phase_a: dependency-free runner (``python -m
tests.test_robustness``), pytest-compatible, skips cleanly when heavy
backends are unavailable.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from voiceiso.config import PipelineConfig


class _Skip(Exception):
    pass


def _mk_pipeline():
    try:
        from voiceiso.pipeline import StreamingPipeline
        return StreamingPipeline(PipelineConfig())
    except Exception as exc:  # pragma: no cover
        raise _Skip(f"pipeline unavailable ({exc})")


# ── C3: a stage exception must NOT kill the live worker (dry fallback) ───────
def test_live_worker_contains_stage_exception():
    try:
        from voiceiso.io.audio_stream import LiveStream
    except Exception as exc:
        raise _Skip(f"sounddevice unavailable ({exc})")
    ls = LiveStream.__new__(LiveStream)          # skip __init__ (no device I/O)
    ls._in_q = queue.Queue(maxsize=4)
    ls._out_q = queue.Queue(maxsize=4)
    ls._run = True
    ls.drops_in = ls.drops_out = ls.worker_errors = 0

    class _Boom:
        def process_block(self, block):
            raise RuntimeError("injected stage failure")

    ls.pipe = _Boom()
    t = threading.Thread(target=ls._worker_loop, daemon=True)
    t.start()
    block = np.full(4800, 0.25, dtype=np.float32)
    ls._in_q.put(block)
    out = ls._out_q.get(timeout=2.0)             # worker must still emit…
    ls._run = False
    t.join(timeout=1.0)
    assert not t.is_alive(), "worker thread must survive and exit cleanly"
    assert np.array_equal(out, block), "fallback must be the DRY block, not silence"
    assert ls.worker_errors == 1


# ── C3: NaN injected mid-chain must not reach the pipeline output ────────────
def test_pipeline_sanitises_nan():
    pipe = _mk_pipeline()

    class _NaNStage:
        name = "nan_injector"
        def __call__(self, ctx):
            ctx.audio = ctx.audio.copy()
            ctx.audio[10] = np.nan
            ctx.audio[20] = np.inf
            return ctx
        def reset(self):
            pass

    pipe.stages.insert(len(pipe.stages) - 1, _NaNStage())
    out = pipe.process_block(np.zeros(4800, dtype=np.float32)).audio
    assert np.all(np.isfinite(out)), "NaN/Inf must be sanitised at the output"
    assert pipe._nan_warned is True


# ── C2: warm-up leaves the pipeline fast and does not reset itself ───────────
def test_warmup_primes_pipeline():
    pipe = _mk_pipeline()
    pipe.warmup(seconds=1.0)                     # short warm-up for test speed
    t0 = time.perf_counter()
    pipe.process_block(np.zeros(4800, dtype=np.float32))
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 100.0, f"post-warmup block took {dt_ms:.1f} ms (budget 100)"


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

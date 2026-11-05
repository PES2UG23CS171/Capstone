"""
Device choice and feedback-guard tests (packaging work, 2026-11).

Runs anywhere, including CI with no audio hardware: a fake ``sounddevice`` stands
in for PortAudio, so Windows device lists can be tested from any platform.  The
engine tests drive the real ``run_engine`` loop and call its real audio callback.

    python -m tests.test_devices
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import types

import numpy as np


class _Skip(Exception):
    pass


class _FakeStream:
    last: "_FakeStream | None" = None

    def __init__(self, samplerate, blocksize, device, channels, dtype, callback, latency):
        self.device = device
        self.channels = channels
        self.callback = callback
        _FakeStream.last = self

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


def _fake_sd(devices, default=(0, 1), hostapi=0):
    """A stand-in for the sounddevice module.  devices: (name, in, out, hostapi)."""
    rows = [dict(name=n, max_input_channels=i, max_output_channels=o,
                 default_samplerate=48000.0, hostapi=h) for n, i, o, h in devices]
    sd = types.ModuleType("sounddevice")
    sd.rows = rows
    sd.rescans = 0
    sd.default = types.SimpleNamespace(hostapi=hostapi, device=default)

    def query_devices(index=None, kind=None):
        if index is None and kind is None:
            return list(sd.rows)
        if index is None:
            index = sd.default.device[0 if kind == "input" else 1]
        return sd.rows[index]

    def _initialize():
        sd.rescans += 1

    sd.query_devices = query_devices
    sd.Stream = _FakeStream
    sd.CallbackFlags = object
    sd._terminate = lambda: None
    sd._initialize = _initialize
    return sd


def _load(sd):
    """Import app.audio.devices and engine against a fake sounddevice."""
    sys.modules["sounddevice"] = sd
    for name in ("app.audio.devices", "app.audio.engine"):
        sys.modules.pop(name, None)
    import app.audio.devices as devices
    import app.audio.engine as engine
    return devices, engine


MAC = [("MacBook Pro Microphone", 1, 0, 0), ("MacBook Pro Speakers", 0, 2, 0)]
MAC_WITH_CABLE = MAC + [("BlackHole 2ch", 2, 2, 0)]
# The shape Windows really reports: each device once per host API, with MME
# truncating names to 31 characters.
WINDOWS = [
    ("Microsoft Sound Mapper - Input", 2, 0, 0),
    ("Microphone (Realtek(R) Audio)", 2, 0, 0),
    ("Microsoft Sound Mapper - Output", 0, 2, 0),
    ("Speakers (Realtek(R) Audio)", 0, 2, 0),
    ("CABLE Input (VB-Audio Virtual C", 0, 2, 0),
    ("CABLE Output (VB-Audio Virtual ", 2, 0, 0),
    ("Microphone (Realtek(R) Audio)", 2, 0, 1),       # DirectSound
    ("Speakers (Realtek(R) Audio)", 0, 2, 1),
    ("Microphone (Realtek(R) Audio)", 2, 0, 2),       # WASAPI
    ("Speakers (Realtek(R) Audio)", 0, 2, 2),
]


# ── device list and first-launch choice ──────────────────────────────────────
def test_windows_lists_only_one_host_api():
    d, _ = _load(_fake_sd(WINDOWS, default=(0, 2)))
    listed = d.query_devices()
    assert len(listed) == 6, [x.name for x in listed]
    assert len({x.name for x in listed}) == 6, "a device is listed twice"
    assert {x.index for x in listed} == {0, 1, 2, 3, 4, 5}


def test_first_launch_prefers_installed_cable():
    d, _ = _load(_fake_sd(MAC_WITH_CABLE))
    assert d.pick_devices(d.query_devices()) == (0, 2)
    d, _ = _load(_fake_sd(WINDOWS, default=(0, 2)))
    mic, out = d.pick_devices(d.query_devices())
    assert out == 4, "must play into CABLE Input"
    assert mic == 0, "must not record from CABLE Output"


def test_first_launch_without_cable_uses_system_default():
    d, _ = _load(_fake_sd(MAC))
    assert d.pick_devices(d.query_devices()) == (0, 1)


def test_system_mic_set_to_cable_is_not_captured():
    d, _ = _load(_fake_sd(MAC_WITH_CABLE, default=(2, 1)))
    mic, _ = d.pick_devices(d.query_devices())
    assert mic == 0, "capturing our own output would produce silence"


def test_explicit_choice_survives_rescan_and_changed_indices():
    d, _ = _load(_fake_sd([("Headset", 1, 2, 0)] + MAC_WITH_CABLE, default=(1, 2)))
    assert d.pick_devices(d.query_devices(), "Headset", "Headset") == (0, 0)
    assert d.pick_devices(d.query_devices(), "Unplugged", "Unplugged") == (1, 3)


def test_no_audio_hardware_does_not_raise():
    d, _ = _load(_fake_sd([], default=(-1, -1)))
    assert d.pick_devices(d.query_devices()) == (None, None)


def test_name_classification():
    d, _ = _load(_fake_sd(MAC))
    for name in ("MacBook Pro Speakers", "Speakers (Realtek(R) Audio)",
                 "Microsoft Sound Mapper - Output", "Built-in Output"):
        assert d.looks_like_loudspeaker(name), name
    for name in ("External Headphones", "Headphones (WH-1000XM4)", "BlackHole 2ch",
                 "CABLE Input (VB-Audio Virtual C"):
        assert not d.looks_like_loudspeaker(name), name
    assert d.call_app_mic_name("BlackHole 2ch") == "BlackHole 2ch"
    assert d.call_app_mic_name("CABLE Input (VB-Audio Virtual C") == "CABLE Output"


# ── real engine loop + real audio callback ───────────────────────────────────
class _Engine:
    """Runs run_engine on a thread with a fake sounddevice and no DSP models."""

    def __init__(self, sd):
        _FakeStream.last = None
        _, self.mod = _load(sd)
        self.mod._HAS_PIPELINE = False
        from app.config import AppConfig
        from app.ipc.messages import CmdType, Command, EvtType
        self.CmdType, self.Command, self.EvtType = CmdType, Command, EvtType
        self.cmd_q, self.evt_q = queue.Queue(), queue.Queue()
        self.thread = threading.Thread(
            target=self.mod.run_engine, args=(self.cmd_q, self.evt_q, AppConfig()), daemon=True)
        self.thread.start()
        self._wait(lambda: _FakeStream.last is not None, "stream never opened")

    def _wait(self, cond, why, timeout=20.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return
            time.sleep(0.01)
        raise AssertionError(why)

    def send(self, kind, value=None):
        self.cmd_q.put(self.Command(kind, value))

    def play_block(self):
        """Run one real audio callback; return the peak sent to the output."""
        s = _FakeStream.last
        indata = np.full((4800, 1), 0.1, dtype=np.float32)
        outdata = np.ones((4800, s.channels[1]), dtype=np.float32)
        s.callback(indata, outdata, 4800, None, None)
        return float(np.max(np.abs(outdata)))

    def latest_status(self):
        status = None
        end = time.monotonic() + 5.0
        while time.monotonic() < end:
            try:
                e = self.evt_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if e.kind == self.EvtType.STATUS:
                status = e.payload
                if self.evt_q.empty():
                    return status
        raise AssertionError("no STATUS event")

    def stop(self):
        self.send(self.CmdType.SHUTDOWN)
        self.thread.join(timeout=10)
        assert not self.thread.is_alive(), "engine did not shut down"


def test_engine_mutes_speakers_until_user_allows():
    e = _Engine(_fake_sd(MAC))
    try:
        assert e.play_block() == 0.0, "first launch must not play mic into the speakers"
        assert e.latest_status().output_muted is True
        e.send(e.CmdType.ALLOW_SPEAKERS)
        e._wait(lambda: e.play_block() > 0.0, "Unmute did not restore audio")
        assert e.latest_status().output_muted is False
    finally:
        e.stop()


def test_engine_passthrough_also_respects_mute():
    e = _Engine(_fake_sd(MAC))
    try:
        e.send(e.CmdType.SET_PASSTHROUGH, True)
        time.sleep(0.3)
        assert e.play_block() == 0.0, "the raw-mic A/B button bypassed the feedback guard"
    finally:
        e.stop()


def test_engine_call_mode_is_live_immediately():
    e = _Engine(_fake_sd(MAC_WITH_CABLE))
    try:
        assert _FakeStream.last.device == (0, 2)
        assert e.play_block() > 0.0
        assert e.latest_status().output_muted is False
    finally:
        e.stop()


def test_engine_rescan_finds_plugged_in_headphones():
    sd = _fake_sd(MAC)
    e = _Engine(sd)
    try:
        assert e.play_block() == 0.0
        sd.rows = MAC_ROWS_WITH_HEADPHONES(sd)
        sd.default.device = (0, 2)
        e.send(e.CmdType.REFRESH_DEVICES)
        e._wait(lambda: _FakeStream.last.device == (0, 2), "rescan did not switch output")
        assert sd.rescans == 1, "PortAudio must re-initialise to see new hardware"
        assert e.play_block() > 0.0, "headphones must not stay muted"
    finally:
        e.stop()


def MAC_ROWS_WITH_HEADPHONES(sd):
    return sd.rows + [dict(name="External Headphones", max_input_channels=0,
                           max_output_channels=2, default_samplerate=48000.0, hostapi=0)]


def test_engine_recovers_after_a_failed_device_switch():
    sd = _fake_sd(MAC + [("Broken", 0, 2, 0)])
    e = _Engine(sd)
    try:
        good = _FakeStream.last
        real_stream = sd.Stream

        def explode(*a, **k):
            raise RuntimeError("device busy")

        sd.Stream = explode
        e.send(e.CmdType.SET_OUTPUT_DEVICE, 2)
        time.sleep(0.3)
        sd.Stream = real_stream
        e.send(e.CmdType.SET_OUTPUT_DEVICE, 1)
        e._wait(lambda: _FakeStream.last is not good, "stuck on the dead stream")
        assert _FakeStream.last.device == (0, 1)
    finally:
        e.stop()


def _run_all():
    saved = sys.modules.get("sounddevice")
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
    if saved is not None:
        sys.modules["sounddevice"] = saved
    print(f"\n{npass} passed, {nskip} skipped, {nfail} failed")
    return nfail


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)

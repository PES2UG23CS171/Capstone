"""
Audio-device enumeration and first-launch device choice.

Wraps ``sounddevice.query_devices()`` and returns serialisable ``DeviceInfo``
instances that can be sent over the IPC queue.

Only devices of the default host API are listed.  On Windows PortAudio reports
every device once per API (MME, DirectSound, WASAPI, WDM-KS), and a stream whose
mic and output come from different APIs fails to open, so offering the full list
would let a user pick a combination that cannot work.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import sounddevice as sd

from app.ipc.messages import DeviceInfo

# Name fragments of virtual audio cables.  Windows MME truncates names to 31
# characters ("CABLE Input (VB-Audio Virtual C"), hence "vb-audio".
_VIRTUAL_MARKERS = ("blackhole", "loopback", "vb-cable", "vb-audio", "soundflower")

# Outputs whose sound can reach the microphone.  "sound mapper" is the Windows
# MME alias for the default device: it could be anything, so treat it as a
# loudspeaker rather than risk howling at someone on first launch.
_LOUDSPEAKER_MARKERS = ("speaker", "built-in output", "sound mapper")

# Free virtual-microphone drivers, offered in the GUI when none is installed.
DRIVER_URLS = {
    "darwin": ("BlackHole", "https://existential.audio/blackhole/"),
    "win32": ("VB-CABLE", "https://vb-audio.com/Cable/"),
}


def is_virtual_device(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _VIRTUAL_MARKERS)


def looks_like_loudspeaker(name: str) -> bool:
    if is_virtual_device(name):
        return False
    lowered = name.lower()
    return any(marker in lowered for marker in _LOUDSPEAKER_MARKERS)


def call_app_mic_name(sink_name: str) -> str:
    """The device to choose as the microphone in Meet/Zoom for this output.

    BlackHole is one device with both directions.  VB-CABLE is a pair: the app
    plays into "CABLE Input" and the call app must record from "CABLE Output".
    """
    if "vb-audio" in sink_name.lower() or "vb-cable" in sink_name.lower():
        return "CABLE Output"
    return sink_name


def query_devices(selected_in: Optional[int] = None,
                  selected_out: Optional[int] = None) -> List[DeviceInfo]:
    """Return the devices of the default host API."""
    hostapi = sd.default.hostapi
    defaults = sd.default.device           # (input_idx, output_idx)
    default_in, default_out = defaults[0], defaults[1]

    devices: List[DeviceInfo] = []
    for idx, d in enumerate(sd.query_devices()):
        if d["hostapi"] != hostapi:
            continue
        devices.append(
            DeviceInfo(
                index=idx,
                name=d["name"],
                max_input_channels=d["max_input_channels"],
                max_output_channels=d["max_output_channels"],
                default_samplerate=d["default_samplerate"],
                is_default_input=(idx == default_in),
                is_default_output=(idx == default_out),
                is_selected_input=(idx == selected_in),
                is_selected_output=(idx == selected_out),
            )
        )
    return devices


def pick_devices(devices: List[DeviceInfo],
                 keep_input: Optional[str] = None,
                 keep_output: Optional[str] = None,
                 ) -> Tuple[Optional[int], Optional[int]]:
    """Choose (input, output) so that a fresh install works without setup.

    * A device the user explicitly chose (``keep_*``, by name, because indices
      change after a rescan) wins while it is still present.
    * Output: a virtual cable if one is installed — someone who installed the
      driver wants call mode, and it cannot feed back.  Otherwise the system
      default.
    * Input: the system default, unless that is a virtual cable.  Pointing the
      system mic at BlackHole for a call is a common setup, and capturing our
      own output would produce silence or a loop.
    """
    inputs = [d for d in devices if d.max_input_channels > 0]
    outputs = [d for d in devices if d.max_output_channels > 0]

    def named(pool: List[DeviceInfo], name: Optional[str]) -> Optional[DeviceInfo]:
        return next((d for d in pool if d.name == name), None) if name else None

    chosen_in = named(inputs, keep_input)
    if chosen_in is None:
        real = [d for d in inputs if not is_virtual_device(d.name)]
        chosen_in = next((d for d in real if d.is_default_input), real[0] if real else None)

    chosen_out = named(outputs, keep_output)
    if chosen_out is None:
        chosen_out = next((d for d in outputs if is_virtual_device(d.name)), None)
    if chosen_out is None:
        chosen_out = next((d for d in outputs if d.is_default_output),
                          outputs[0] if outputs else None)

    return (chosen_in.index if chosen_in else None,
            chosen_out.index if chosen_out else None)

"""
Audio-device enumeration helpers.

Wraps ``sounddevice.query_devices()`` and returns serialisable
``DeviceInfo`` dataclass instances that can be sent over the IPC queue.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import sounddevice as sd

from app.ipc.messages import DeviceInfo


def query_devices() -> List[DeviceInfo]:
    """Return a list of all available audio devices."""
    raw = sd.query_devices()
    defaults = sd.default.device           # (input_idx, output_idx)
    default_in: Optional[int] = defaults[0] if isinstance(defaults, (list, tuple)) else defaults
    default_out: Optional[int] = defaults[1] if isinstance(defaults, (list, tuple)) else defaults

    devices: List[DeviceInfo] = []
    for idx, d in enumerate(raw):
        devices.append(
            DeviceInfo(
                index=idx,
                name=d["name"],
                max_input_channels=d["max_input_channels"],
                max_output_channels=d["max_output_channels"],
                default_samplerate=d["default_samplerate"],
                is_default_input=(idx == default_in),
                is_default_output=(idx == default_out),
            )
        )
    return devices


def input_devices() -> List[DeviceInfo]:
    """Return only devices that can capture audio."""
    return [d for d in query_devices() if d.max_input_channels > 0]


def output_devices() -> List[DeviceInfo]:
    """Return only devices that can play audio."""
    return [d for d in query_devices() if d.max_output_channels > 0]


def default_input_index() -> Optional[int]:
    """Return the OS-default input device index, or *None*."""
    defaults = sd.default.device
    idx = defaults[0] if isinstance(defaults, (list, tuple)) else defaults
    return int(idx) if idx is not None and idx >= 0 else None


def default_output_index() -> Optional[int]:
    """Return the OS-default output device index, or *None*."""
    defaults = sd.default.device
    idx = defaults[1] if isinstance(defaults, (list, tuple)) else defaults
    return int(idx) if idx is not None and idx >= 0 else None

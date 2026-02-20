"""
IPC message definitions for GUI ↔ Audio-Engine communication.

All messages are plain dataclasses so they are pickle-safe and can cross
process boundaries via ``multiprocessing.Queue``.

Direction legend:
    CMD  → GUI  → Engine   (command queue)
    EVT  ← Engine → GUI    (status / event queue)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


# ── Command types (GUI → Engine) ────────────────────────────────────────────


class CmdType(Enum):
    """Exhaustive list of commands the GUI can send to the engine."""

    SET_ENABLED = auto()          # bool
    SET_STRENGTH = auto()         # float  [0.0, 1.0]
    SET_GAIN = auto()             # float  dB
    SET_INPUT_DEVICE = auto()     # Optional[int]
    SET_OUTPUT_DEVICE = auto()    # Optional[int]
    GET_DEVICES = auto()          # request device list
    SHUTDOWN = auto()             # graceful stop


@dataclass
class Command:
    """A single command sent from the GUI to the audio engine."""

    kind: CmdType
    value: Any = None


# ── Event types (Engine → GUI) ──────────────────────────────────────────────


class EvtType(Enum):
    """Exhaustive list of events the engine can send back to the GUI."""

    STATUS = auto()               # periodic heartbeat with levels
    DEVICE_LIST = auto()          # response to GET_DEVICES
    ERROR = auto()                # non-fatal error string
    ENGINE_STOPPED = auto()       # engine has shut down


@dataclass
class DeviceInfo:
    """Minimal info about one audio device."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default_input: bool = False
    is_default_output: bool = False


@dataclass
class StatusPayload:
    """Periodic status snapshot from the audio engine."""

    running: bool = False
    input_level_db: float = -120.0   # dBFS peak of last block
    output_level_db: float = -120.0
    xruns: int = 0                    # cumulative buffer over/under-runs
    cpu_percent: float = 0.0          # rough engine CPU usage


@dataclass
class Event:
    """A single event sent from the audio engine back to the GUI."""

    kind: EvtType
    payload: Any = None               # StatusPayload | List[DeviceInfo] | str

"""
Virtual Audio Device Routing
=============================
Routes cleaned audio output to a virtual audio device so that Zoom, Teams,
and other conferencing applications can pick it up as a microphone input.

Supported platforms:
* **Windows** — VB-Audio Virtual Cable (``CABLE Input``)
* **Linux**   — PipeWire / PulseAudio null sink
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
except ImportError:
    sd = None


def get_virtual_output_device_index() -> Optional[int]:
    """Scan for a virtual audio cable device and return its index, or None."""
    if sd is None:
        logger.warning("sounddevice not installed — cannot detect virtual device.")
        return None

    keywords = ("virtual", "cable", "vb-audio", "voicemeeter")
    for idx, dev in enumerate(sd.query_devices()):
        name_lower = dev["name"].lower()
        if dev["max_output_channels"] > 0 and any(kw in name_lower for kw in keywords):
            logger.info("Virtual output device found: [%d] %s", idx, dev["name"])
            return idx

    logger.info("No virtual output device detected.")
    return None


def setup_virtual_device() -> bool:
    """Attempt to set up a virtual audio device.  Returns True on success."""
    if sys.platform == "win32":
        idx = get_virtual_output_device_index()
        if idx is not None:
            logger.info("Virtual cable available at device index %d.", idx)
            return True
        logger.warning(
            "No virtual audio cable found on Windows.  "
            "Install VB-Audio Virtual Cable: https://vb-audio.com/Cable/"
        )
        return False

    elif sys.platform.startswith("linux"):
        # Try PipeWire / PulseAudio null sink
        try:
            result = subprocess.run(
                [
                    "pactl",
                    "load-module",
                    "module-null-sink",
                    "sink_name=NoiseFilterOutput",
                    'sink_properties=device.description="Noise_Filter_Output"',
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.info("PulseAudio null sink created: NoiseFilterOutput")
                return True
            else:
                logger.warning("pactl failed: %s", result.stderr.strip())
                return False
        except FileNotFoundError:
            logger.warning("pactl not found — PulseAudio/PipeWire may not be installed.")
            return False
        except Exception as exc:
            logger.warning("Virtual device setup failed: %s", exc)
            return False

    else:
        logger.warning("Virtual device routing not supported on %s.", sys.platform)
        return False

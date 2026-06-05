"""
voiceiso — CPU-only real-time voice isolation pipeline.

A commercial-style speech-enhancement stack (in the spirit of Apple Voice
Isolation / Krisp) built around the DeepFilterNet3 enhancement core, with a
full surrounding pipeline:

    mic → preprocessing → AEC → VAD → noise-classification → enhancement
        → speech-preservation → post-filter → output

The capstone novelty is the *integrated system architecture* and the dynamic
controller that fuses VAD / noise-class / SNR signals to steer suppression —
not the enhancement network itself.

Sub-packages
------------
* ``voiceiso.stages``   — one module per pipeline stage (each a ``Stage``).
* ``voiceiso.io``       — audio stream + buffering.
* ``voiceiso.bench``    — evaluation + profiling harness.
* ``voiceiso.data``     — dynamic-mixing dataset pipeline.
"""

from __future__ import annotations

__version__ = "0.1.0"

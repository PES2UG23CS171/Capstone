"""
Central configuration for the Real-Time AI Audio Filter.

Every magic number lives here — nothing hardcoded elsewhere.
"""

from __future__ import annotations

# ── Audio I/O ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48_000          # Hz — standard VoIP
CHANNELS = 1                  # Monaural
BIT_DEPTH = 16                # PCM 16-bit input
BLOCK_SIZE = 256              # Frames per sounddevice callback (~5.3 ms @ 48 kHz)
RING_BUFFER_SECONDS = 0.5     # 0.5 s of audio context (~24 000 samples)
CONTEXT_WINDOW_SAMPLES = 512  # Sliding window fed to model layers

# ── Audio passthrough (presentation demo) ────────────────────────────────────
DIRECT_PASSTHROUGH = False    # True → skip model, raw mic → headphones
OUTPUT_DEVICE_INDEX = None    # None = system default; int to pin a device
INPUT_DEVICE_INDEX = None     # None = system default; int to pin a device
PASSTHROUGH_BLOCK_SIZE = 256  # Block size for low-latency passthrough
WAVEFORM_QUEUE_MAXSIZE = 50   # Max frames queued for GUI waveform display

# ── Model ────────────────────────────────────────────────────────────────────
FIR_FILTER_LENGTH = 64        # Number of FIR taps predicted by DeepFIR
MAMBA_D_MODEL = 64            # Mamba hidden state dimension
MAMBA_D_STATE = 16            # SSM state size
MAMBA_N_LAYERS = 4            # Number of Mamba blocks

# ── Quantization ─────────────────────────────────────────────────────────────
QUANTIZE_INT8 = True
PRUNE_RATIO = 0.50            # Remove 50 % of weights by magnitude

# ── Performance targets ──────────────────────────────────────────────────────
TARGET_RTF = 0.8              # Real-Time Factor must be < 0.8 on i5 CPU
MAX_ALGORITHMIC_LATENCY_MS = 100
MAX_END_TO_END_LATENCY_MS = 300

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset"
LIBRISPEECH_DIR = "data/raw/librispeech"
FREESOUND_DIR = "data/raw/freesound"
RIR_DIR = "data/raw/openair_rirs"
CHECKPOINT_DIR = "checkpoints"
ONNX_MODEL_PATH = "model/filter_model.onnx"

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
SNR_RANGE_DB = (-5, 20)       # Dynamic mixing SNR range
NUM_SYNTHETIC_PAIRS = 10_000
TRAIN_CONTEXT_WINDOW = 64     # Shorter window for training speed (model is length-agnostic)
TRAIN_EPOCHS = 50             # Number of training epochs

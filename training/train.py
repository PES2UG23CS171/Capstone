"""
Training Loop
=============
Full training pipeline: data loading → forward pass → combined loss →
backward → gradient clipping → optimiser step.

After training: prune → quantize → benchmark RTF → export ONNX.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import config as cfg
from model.combined_model import CombinedModel
from model.quantize import apply_magnitude_pruning, quantize_model_int8, benchmark_rtf
from model.export_onnx import export_to_onnx
from training.losses import si_sdr_loss


def _select_device() -> str:
    """Pick the best available device: MPS (Apple Silicon) > CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Device: MPS (Apple Silicon GPU)")
        return "mps"
    print("  Device: CPU")
    return "cpu"


def train(resume: bool = False, epochs: int | None = None) -> None:
    """Full training loop."""

    print("=" * 60)
    print("  Training — Real-Time Transient Noise Suppressor")
    print("=" * 60)

    num_epochs = epochs if epochs is not None else cfg.EPOCHS

    # ── Device ───────────────────────────────────────────────────────────
    device = _select_device()

    # ── Dataset ──────────────────────────────────────────────────────────
    try:
        from dataset.dataset_loader import TransientNoiseDataset

        train_ds = TransientNoiseDataset(
            cfg.DATASET_DIR, split="train",
            context_window=cfg.TRAIN_CONTEXT_WINDOW,
        )
        val_ds = TransientNoiseDataset(
            cfg.DATASET_DIR, split="val", augment=False,
            context_window=cfg.TRAIN_CONTEXT_WINDOW,
        )

        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=False,
        )
        print(f"  Train samples: {len(train_ds)}")
        print(f"  Val samples  : {len(val_ds)}")
    except Exception as exc:
        print(f"  ⚠ Dataset not available ({exc})")
        print("  → Using synthetic dummy data for structure validation")
        train_loader = _dummy_loader(200)
        val_loader = _dummy_loader(50)

    # ── Model ────────────────────────────────────────────────────────────
    model = CombinedModel()
    model.count_parameters()
    model = model.to(device)

    # ── Optimiser & scheduler ────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=num_epochs)

    # ── Resume ───────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_val_loss = float("inf")

    if resume:
        latest = ckpt_dir / "latest.pt"
        if latest.exists():
            state = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimiser.load_state_dict(state["optimiser"])
            start_epoch = state["epoch"] + 1
            best_val_loss = state.get("best_val_loss", float("inf"))
            print(f"  Resumed from epoch {start_epoch}")

    # ── Training log ─────────────────────────────────────────────────────
    log_path = ckpt_dir / "training_log.csv"
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if start_epoch == 0:
        log_writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        model.train()
        train_losses = []

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") if tqdm else train_loader

        for noisy, clean in iterator:
            noisy, clean = noisy.to(device), clean.to(device)

            estimated = model(noisy)
            loss = si_sdr_loss(clean, estimated)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            train_losses.append(loss.item())

        scheduler.step()

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                estimated = model(noisy)
                loss = si_sdr_loss(clean, estimated)
                val_losses.append(loss.item())

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch+1:3d} │ train_loss={train_loss:.4f} │ "
              f"val_loss={val_loss:.4f} │ lr={lr:.6f} │ {elapsed:.1f}s")

        log_writer.writerow([epoch + 1, train_loss, val_loss, lr, round(elapsed, 1)])
        log_file.flush()

        # Save checkpoint
        # Move model to CPU for saving to ensure portability
        model_cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_cpu_state, ckpt_dir / "best.pt")

        torch.save({
            "epoch": epoch,
            "model": model_cpu_state,
            "optimiser": optimiser.state_dict(),
            "best_val_loss": best_val_loss,
        }, ckpt_dir / "latest.pt")

    log_file.close()

    # ── Post-training ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Post-Training: Prune → Quantize → Benchmark → Export")
    print("=" * 60)

    # Move model to CPU for post-processing (quantization requires CPU)
    model = model.to("cpu")

    # Load best
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))

    # Prune
    print("\n  Applying magnitude pruning…")
    apply_magnitude_pruning(model)

    # Quantize
    print("\n  Applying INT8 quantization…")
    quantized = quantize_model_int8(model)

    # Benchmark
    print("\n  Benchmarking RTF…")
    benchmark_rtf(model)

    # Export ONNX
    print("\n  Exporting to ONNX…")
    export_to_onnx(model)

    print("\n  Training complete! ✓")


def _dummy_loader(n: int):
    """Create a dummy DataLoader for structure validation."""
    noisy = torch.randn(n, cfg.CONTEXT_WINDOW_SAMPLES)
    clean = torch.randn(n, cfg.CONTEXT_WINDOW_SAMPLES)
    ds = torch.utils.data.TensorDataset(noisy, clean)
    return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()
    train(resume=args.resume, epochs=args.epochs)

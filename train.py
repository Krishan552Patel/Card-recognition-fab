"""
Standalone training script for the card recognition model.

Usage:
    python train.py --image-dir /path/to/flat/card/images
    python train.py --image-dir /path/to/images --output-dir ./checkpoints --epochs 50

Imports from the project modules (backbone, dataset, losses, augmentation, config)
so that the same code runs identically locally and on cloud GPU (Modal / Colab).
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import config
from augmentation import get_controlled_augmentation, get_val_transforms
from backbone import CardNet
from dataset import CardDataset
from losses import CosFaceLoss


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            loss = criterion(model(images), labels)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  val  ", leave=False):
            images, labels = images.to(device), labels.to(device)
            total_loss += criterion(model(images), labels).item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train card recognition embedding model")
    parser.add_argument("--image-dir", required=True, help="Flat folder of card images")
    parser.add_argument("--output-dir", default="./checkpoints", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--check-interval", type=int, default=config.CHECK_INTERVAL,
                        help="Epochs between top-k accuracy checks")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = CardDataset(args.image_dir, get_controlled_augmentation(config.IMAGE_SIZE))
    val_ds = CardDataset(args.image_dir, get_val_transforms(config.IMAGE_SIZE), rotations=[0])

    indices = np.random.permutation(len(train_ds.images))
    split = int(config.TRAIN_SPLIT * len(train_ds.images))
    train_card_idx = set(indices[:split])
    val_card_idx = set(indices[split:])

    train_samples = [i for i, (c, _) in enumerate(train_ds.samples) if c in train_card_idx]
    val_samples = [i for i, (c, _) in enumerate(val_ds.samples) if c in val_card_idx]

    train_loader = DataLoader(
        Subset(train_ds, train_samples),
        batch_size=args.batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_samples),
        batch_size=args.batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    num_classes = train_ds.get_num_classes()
    print(f"Cards: {num_classes} | Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    # ── Model, loss, optimiser ─────────────────────────────────────────────────
    model = CardNet(emb_dim=config.EMBEDDING_DIM).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    criterion = CosFaceLoss(num_classes, config.EMBEDDING_DIM,
                            config.COSFACE_SCALE, config.COSFACE_MARGIN).to(device)

    optimizer = torch.optim.AdamW(
        trainable + list(criterion.parameters()),
        lr=args.lr, weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Training loop ──────────────────────────────────────────────────────────
    best_loss = float("inf")
    patience_counter = 0
    best_path = output_dir / "best_model.pth"

    print("=" * 60)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{args.epochs} | train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_loss": val_loss, "num_classes": num_classes},
                best_path,
            )
            print(f"  => Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping after {epoch} epochs (patience={args.patience})")
                break

    print("=" * 60)
    print(f"Done. Best val loss: {best_loss:.4f}")
    print(f"Model saved to: {best_path}")
    return str(best_path)


if __name__ == "__main__":
    main()

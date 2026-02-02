"""
Training Script for FAB Card Siamese Network
With Early Stopping Support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import os
import argparse

from backbone import create_model
from arcface_loss import ArcFaceLoss
from dataset import create_dataloaders, FABCardDataset
import config


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass with mixed precision
        with autocast(enabled=scaler is not None):
            embeddings = model(images)
            loss = criterion(embeddings, labels)
        
        # Backward pass
        optimizer.zero_grad()
        
        if scaler:
            # Mixed precision backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard backward
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            embeddings = model(images)
            loss = criterion(embeddings, labels)
            
            running_loss += loss.item()
    
    avg_loss = running_loss / len(val_loader)
    return avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FAB card embedding model")
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=config.NUM_WORKERS, help='Number of workers')
    parser.add_argument('--image-size', type=int, default=config.IMAGE_SIZE, help='Image size')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    checkpoint_dir = Path(config.MODEL_SAVE_DIR)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    Path(config.LOGS_DIR).mkdir(exist_ok=True, parents=True)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        root_dir=config.IMAGE_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    # Get number of classes
    temp_dataset = FABCardDataset(config.IMAGE_DIR)
    num_classes = temp_dataset.get_num_classes()
    print(f"Number of unique cards: {num_classes}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        embedding_dim=config.EMBEDDING_DIM,
        gem_p=config.GEM_P_INIT,
        pretrained=True
    )
    model = model.to(device)
    
    # Freeze backbone
    print("Freezing backbone parameters...")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Create loss
    criterion = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=config.EMBEDDING_DIM,
        scale=config.ARCFACE_SCALE,
        margin=config.ARCFACE_MARGIN
    ).to(device)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler:
        print("Using mixed precision training (AMP)")
    
    # Tensorboard
    writer = SummaryWriter(config.LOGS_DIR)
    
    # Early stopping
    patience = args.patience
    patience_counter = 0
    print(f"Early stopping enabled with patience: {patience}")
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epoch(s)")
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered! No improvement for {patience} epochs.")
                print(f"Best validation loss: {best_val_loss:.4f} at epoch {epoch - patience}")
                break
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
    
    writer.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {checkpoint_dir}")

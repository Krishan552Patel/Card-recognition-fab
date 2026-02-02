"""
Training Script for FAB Card Siamese Network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import os

from backbone import create_model
from arcface_loss import ArcFaceLoss
from dataset import create_dataloaders, FABCardDataset
import config


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
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


def train(
    root_dir=config.IMAGE_DIR,
    num_epochs=config.NUM_EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    save_dir=config.MODEL_SAVE_DIR,
    log_dir=config.LOGS_DIR
):
    """
    Main training function.
    
    Args:
        root_dir: Root directory containing card folders
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints
        log_dir: Directory for tensorboard logs
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        image_size=config.IMAGE_SIZE
    )
    
    # Get number of classes from dataset
    temp_dataset = FABCardDataset(root_dir)
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
    
    # Freeze backbone (Critical for small datasets to prevent overfitting)
    print("Freezing backbone parameters...")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Only train the GeM pooling (p parameter) and the FC head
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
"""
Training Script for FAB Card Siamese Network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import os
import argparse # Added import for argparse

from backbone import create_model
from arcface_loss import ArcFaceLoss
from dataset import create_dataloaders, FABCardDataset
import config


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
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


def train(
    root_dir=config.IMAGE_DIR,
    num_epochs=config.NUM_EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    save_dir=config.MODEL_SAVE_DIR,
    log_dir=config.LOGS_DIR
):
    """
    Main training function.
    
    Args:
        root_dir: Root directory containing card folders
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints
        log_dir: Directory for tensorboard logs
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        image_size=config.IMAGE_SIZE
    )
    
    # Get number of classes from dataset
    temp_dataset = FABCardDataset(root_dir)
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
    
    # Freeze backbone (Critical for small datasets to prevent overfitting)
    print("Freezing backbone parameters...")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Only train the GeM pooling (p parameter) and the FC head
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Create loss
    criterion = ArcFaceLoss()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train FAB card embedding model")
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=config.NUM_WORKERS, help='Number of data loading workers')
    parser.add_argument('--image-size', type=int, default=config.IMAGE_SIZE, help='Input image size')
    
    args = parser.parse_args()
    
    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

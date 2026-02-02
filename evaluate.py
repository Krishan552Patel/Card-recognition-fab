"""
Evaluate Model Accuracy
Calculates Top-1 and Top-5 accuracy on the validation set.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np

from backbone import create_model
from dataset import create_dataloaders, FABCardDataset
import config

def evaluate(
    model_path,
    root_dir=config.IMAGE_DIR,
    batch_size=config.BATCH_SIZE,
    image_size=config.IMAGE_SIZE
):
    """
    Evaluate model accuracy.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    _, val_loader = create_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        image_size=image_size
    )
    
    # Get number of classes
    temp_dataset = FABCardDataset(root_dir)
    num_classes = temp_dataset.get_num_classes()
    print(f"Number of unique cards: {num_classes}")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = create_model(
        embedding_dim=config.EMBEDDING_DIM,
        gem_p=config.GEM_P_INIT,
        pretrained=False  # Weights loaded from checkpoint
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create prototypes (mean embedding for each class)
    print("Computing class prototypes...")
    prototypes = torch.zeros(num_classes, config.EMBEDDING_DIM).to(device)
    class_counts = torch.zeros(num_classes).to(device)
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Computing prototypes"):
            images = images.to(device)
            labels = labels.to(device)
            
            embeddings = model(images)
            
            for i in range(len(labels)):
                label = labels[i]
                prototypes[label] += embeddings[i]
                class_counts[label] += 1
    
    # Normalize prototypes
    # Avoid division by zero
    class_counts = class_counts.unsqueeze(1).clamp(min=1)
    prototypes = prototypes / class_counts
    prototypes = F.normalize(prototypes, p=2, dim=1)
    
    # Evaluate accuracy
    print("\nEvaluating accuracy...")
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            embeddings = model(images)
            
            # Cosine similarity to prototypes
            # (B, E) @ (C, E)^T -> (B, C)
            similarity = torch.matmul(embeddings, prototypes.T)
            
            # Top-k
            _, indices = torch.topk(similarity, k=5, dim=1)
            
            for i in range(len(labels)):
                label = labels[i]
                pred_1 = indices[i, 0]
                pred_5 = indices[i, :]
                
                if pred_1 == label:
                    correct_1 += 1
                if label in pred_5:
                    correct_5 += 1
                
                total += 1
    
    acc_1 = 100.0 * correct_1 / total
    acc_5 = 100.0 * correct_5 / total
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Top-1 Accuracy: {acc_1:.2f}%")
    print(f"Top-5 Accuracy: {acc_5:.2f}%")
    print("=" * 60)
    
    return acc_1, acc_5

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model/checkpoints/best_model.pth')
    args = parser.parse_args()
    
    evaluate(args.model)

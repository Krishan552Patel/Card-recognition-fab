"""
ArcFace Loss Function
Angular margin loss for metric learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    
    Paper: https://arxiv.org/abs/1801.07698
    
    Forces the model to maximize angular (cosine) distance between classes.
    More aggressive than Triplet Loss for fine-grained distinctions.
    """
    
    def __init__(self, num_classes, embedding_dim, scale=30.0, margin=0.5):
        """
        Args:
            num_classes: Number of unique cards
            embedding_dim: Dimension of embeddings
            scale: Scale parameter (s in paper)
            margin: Angular margin (m in paper, in radians)
        """
        super(ArcFaceLoss, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin
        
        # Weight matrix for classification
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute cos(m) and sin(m)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        
        # Threshold for numerical stability
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: L2-normalized embeddings (B, embedding_dim)
            labels: Class labels (B,)
            
        Returns:
            loss: ArcFace loss
        """
        # Normalize weights
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings, weight_norm)  # (B, num_classes)
        
        # Get cosine for ground truth class
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Avoid numerical issues
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encode labels
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale
        output *= self.scale
        
        # Cross entropy loss
        loss = F.cross_entropy(output, labels)
        
        return loss


class TripletLoss(nn.Module):
    """
    Alternative: Triplet Loss with hard negative mining.
    Simpler than ArcFace but may be less effective.
    """
    
    def __init__(self, margin=0.5):
        """
        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Anchor embeddings (B, D)
            positive: Positive embeddings (B, D)
            negative: Negative embeddings (B, D)
            
        Returns:
            loss: Triplet loss
        """
        # Euclidean distance
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


if __name__ == "__main__":
    # Test ArcFace loss
    print("Testing ArcFace Loss...")
    
    num_classes = 100
    embedding_dim = 512
    batch_size = 32
    
    # Create loss
    arcface = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        scale=30.0,
        margin=0.5
    )
    
    # Dummy data
    embeddings = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    loss = arcface(embeddings, labels)
    
    print(f"Batch size: {batch_size}")
    print(f"Num classes: {num_classes}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test Triplet loss
    print("\nTesting Triplet Loss...")
    
    triplet = TripletLoss(margin=0.5)
    
    anchor = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    positive = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    negative = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    
    loss = triplet(anchor, positive, negative)
    print(f"Loss: {loss.item():.4f}")

"""
Model Backbone: MobileNetV3-Small + GeM Pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling
    
    Instead of average pooling, uses: (1/n * Σ(x^p))^(1/p)
    where p is learnable. Focuses on high-activation features.
    """
    
    def __init__(self, p=3.0, eps=1e-6):
        """
        Args:
            p: Initial power parameter (learnable)
            eps: Small constant for numerical stability
        """
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Pooled tensor (B, C)
        """
        # Clamp to avoid negative values
        x = x.clamp(min=self.eps)
        
        # Apply power
        x = x.pow(self.p)
        
        # Average pool
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # Apply inverse power
        x = x.pow(1.0 / self.p)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        return x
    
    def __repr__(self):
        return f"GeM(p={self.p.data.item():.4f}, eps={self.eps})"


class CardEmbeddingNet(nn.Module):
    """
    Card Embedding Network
    
    Architecture:
        - MobileNetV3-Small (pretrained)
        - GeM Pooling
        - FC layer to embedding_dim
        - L2 normalization
    """
    
    def __init__(self, embedding_dim=512, gem_p=3.0, pretrained=True):
        """
        Args:
            embedding_dim: Dimension of output embedding
            gem_p: Initial GeM pooling parameter
            pretrained: Use ImageNet pretrained weights
        """
        super(CardEmbeddingNet, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Load MobileNetV3-Small backbone
        self.backbone = timm.create_model(
            'mobilenetv3_small_100',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling (we'll use GeM)
        )
        
        # Get number of features from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.num_features = features.shape[1]
        
        # GeM Pooling
        self.gem = GeM(p=gem_p)
        
        # Embedding layer
        self.fc = nn.Linear(self.num_features, embedding_dim)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(embedding_dim)
        
        # Dropout for regularization (prevent overfitting)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            L2-normalized embeddings (B, embedding_dim)
        """
        # Extract features
        features = self.backbone(x)  # (B, C, H, W)
        
        # GeM pooling
        pooled = self.gem(features)  # (B, C)
        
        # Embedding
        embedding = self.fc(pooled)  # (B, embedding_dim)
        embedding = self.bn(embedding)
        embedding = self.dropout(embedding)  # Apply dropout
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def get_embedding_dim(self):
        """Return embedding dimension."""
        return self.embedding_dim


def create_model(embedding_dim=512, gem_p=3.0, pretrained=True):
    """
    Factory function to create model.
    
    Args:
        embedding_dim: Dimension of output embedding
        gem_p: Initial GeM pooling parameter
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        CardEmbeddingNet model
    """
    model = CardEmbeddingNet(
        embedding_dim=embedding_dim,
        gem_p=gem_p,
        pretrained=pretrained
    )
    return model


if __name__ == "__main__":
    # Test model
    print("Testing CardEmbeddingNet...")
    
    model = create_model(embedding_dim=512, gem_p=3.0)
    print(f"\nModel created:")
    print(f"  Embedding dim: {model.get_embedding_dim()}")
    print(f"  GeM pooling: {model.gem}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output norm: {output.norm(dim=1)}")  # Should be ~1.0 (L2 normalized)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

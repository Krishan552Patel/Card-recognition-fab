"""
CardNet backbone — MobileNetV3-Large with frozen weights, GeM pooling, 2-layer embedding head.
Source of truth: Card_Recognition_Training.ipynb (V6).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """Generalized Mean Pooling with learnable power parameter."""

    def __init__(self, p: float = 3.0):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.adaptive_avg_pool2d(x.clamp(min=1e-6).pow(self.p), 1)
            .pow(1.0 / self.p)
            .view(x.size(0), -1)
        )


class CardNet(nn.Module):
    """
    Card recognition embedding network.

    Backbone: MobileNetV3-Large (pretrained ImageNet, permanently frozen).
    Pooling:  GeM (learnable exponent).
    Head:     Linear(feat -> 1024) -> BN -> ReLU -> Dropout(0.5) -> Linear(1024 -> emb_dim) -> BN.
    Output:   L2-normalised embedding vector.
    """

    def __init__(self, emb_dim: int = 512):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_large_100", pretrained=True, num_classes=0, global_pool=""
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        with torch.no_grad():
            n_feat = self.backbone(torch.randn(1, 3, 224, 224)).shape[1]

        self.gem = GeM()
        self.head = nn.Sequential(
            nn.Linear(n_feat, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.gem(self.backbone(x))
        emb = self.head(features)
        return F.normalize(emb, p=2, dim=1)

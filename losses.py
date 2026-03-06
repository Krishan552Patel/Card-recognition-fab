"""
CosFace loss for card recognition training.
Source of truth: Card_Recognition_Training.ipynb (V6), cell 19.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFaceLoss(nn.Module):
    """
    CosFace (Large Margin Cosine Loss).

    Applies a cosine margin to the target class before scaling and cross-entropy:
        logit_i = (cos(theta_i) - margin * 1[i == y]) * scale

    Args:
        num_classes: Number of card identities.
        emb_dim:     Embedding dimension (must match CardNet output).
        scale:       Feature scale (default 30.0).
        margin:      Cosine margin (default 0.35).
    """

    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        scale: float = 30.0,
        margin: float = 0.35,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        W = F.normalize(self.weight, p=2, dim=1)
        cos = F.linear(emb, W)
        one_hot = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1.0)
        logits = (cos - one_hot * self.margin) * self.scale
        return F.cross_entropy(logits, labels)

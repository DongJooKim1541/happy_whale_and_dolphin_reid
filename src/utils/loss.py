import torch
import torch.nn as nn
from torch.nn.modules.distance import PairwiseDistance


class TripletLoss(nn.Module):
    """Triplet loss with margin."""

    def __init__(self, margin: float = 0.0001) -> None:
        """Initialize TripletLoss.

        Args:
            margin: Minimum difference between positive and negative distances
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(p=2)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """Calculate triplet loss.

        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)

        Returns:
            Scalar loss value
        """
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)
        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)
        return loss

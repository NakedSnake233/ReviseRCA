"""Losses for node-level service RCA."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from dyrca_rca.models.config import PAIRWISE_RANKING_COEF


def masked_node_cross_entropy(logits: torch.Tensor, label: torch.Tensor,
                              candidate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cross entropy over candidate service nodes."""
    label_idx = int(label.item() if hasattr(label, "item") else label)
    if logits.dim() != 1:
        logits = logits.view(-1)

    if candidate_mask is None:
        target = torch.tensor([label_idx], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits.unsqueeze(0), target)

    mask = candidate_mask.to(device=logits.device, dtype=torch.bool)
    candidate_indices = torch.nonzero(mask, as_tuple=False).view(-1)
    if candidate_indices.numel() == 0:
        raise ValueError("candidate_mask selects no nodes")
    matches = torch.nonzero(candidate_indices == label_idx, as_tuple=False).view(-1)
    if matches.numel() == 0:
        raise ValueError("label is not included in candidate_mask")
    target = matches[:1].to(dtype=torch.long)
    return F.cross_entropy(logits[candidate_indices].unsqueeze(0), target)


def pairwise_ranking_loss(logits: torch.Tensor, label: torch.Tensor,
                          candidate_mask: Optional[torch.Tensor] = None,
                          margin: float = 1.0) -> torch.Tensor:
    """Margin ranking loss that scores the root node above other candidates."""
    label_idx = int(label.item() if hasattr(label, "item") else label)
    if logits.dim() != 1:
        logits = logits.view(-1)
    if candidate_mask is None:
        candidate_indices = torch.arange(logits.numel(), device=logits.device)
    else:
        candidate_indices = torch.nonzero(
            candidate_mask.to(device=logits.device, dtype=torch.bool),
            as_tuple=False,
        ).view(-1)
    negatives = candidate_indices[candidate_indices != label_idx]
    if negatives.numel() == 0:
        return logits.sum() * 0.0
    positive = logits[label_idx]
    return torch.relu(margin - (positive - logits[negatives])).mean()


def rca_node_loss(logits: torch.Tensor, label: torch.Tensor,
                  candidate_mask: Optional[torch.Tensor] = None,
                  use_pairwise: bool = True) -> Tuple[torch.Tensor, dict]:
    """Node-level RCA loss with optional fixed-coefficient pairwise term."""
    ce = masked_node_cross_entropy(logits, label, candidate_mask)
    if use_pairwise:
        rank = pairwise_ranking_loss(logits, label, candidate_mask)
        total = ce + PAIRWISE_RANKING_COEF * rank
    else:
        rank = ce.new_tensor(0.0)
        total = ce
    return total, {
        "cross_entropy": float(ce.detach().cpu().item()),
        "pairwise_ranking": float(rank.detach().cpu().item()),
        "pairwise_coef": PAIRWISE_RANKING_COEF,
    }

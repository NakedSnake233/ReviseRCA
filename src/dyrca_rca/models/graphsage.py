"""Weighted GraphSAGE for small RCA graphs."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyrca_rca.models.config import DROPOUT, HIDDEN_DIM, NUM_LAYERS


class WeightedSAGELayer(nn.Module):
    """GraphSAGE layer with weighted mean neighbor aggregation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_nodes = x.size(0)
        if edge_index.numel() == 0:
            neigh = torch.zeros_like(x)
        else:
            src = edge_index[0].long()
            dst = edge_index[1].long()
            if edge_weight is None:
                weight = x.new_ones(src.numel())
            else:
                weight = edge_weight.to(device=x.device, dtype=x.dtype).view(-1)
                if weight.numel() != src.numel():
                    raise ValueError("edge_weight length must match edge count")
            weighted_messages = x[src] * weight.unsqueeze(-1)
            neigh = x.new_zeros((num_nodes, x.size(1)))
            denom = x.new_zeros((num_nodes, 1))
            neigh.index_add_(0, dst, weighted_messages)
            denom.index_add_(0, dst, weight.unsqueeze(-1))
            neigh = neigh / denom.clamp_min(1e-8)
        return self.self_linear(x) + self.neigh_linear(neigh)


class WeightedGraphSAGE(nn.Module):
    """Fixed-config weighted GraphSAGE that outputs one score per node."""

    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        layers = []
        current_dim = in_dim
        for _ in range(num_layers):
            layers.append(WeightedSAGELayer(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if edge_weight is None and edge_attr is not None and edge_attr.numel() > 0:
            edge_weight = edge_attr[:, -1]
        h = x
        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
            if layer_idx < len(self.layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return self.scorer(h).squeeze(-1)

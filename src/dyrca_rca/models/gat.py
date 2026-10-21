"""Edge-aware GAT for small RCA graphs."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyrca_rca.models.config import DROPOUT, HIDDEN_DIM, NUM_LAYERS


class EdgeAwareGATLayer(nn.Module):
    """Single-head GAT layer with optional edge-attribute attention bias."""

    def __init__(self, in_dim: int, out_dim: int, edge_attr_dim: int = 0):
        super().__init__()
        self.node_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(out_dim))
        self.attn_dst = nn.Parameter(torch.empty(out_dim))
        self.edge_bias = nn.Linear(edge_attr_dim, 1, bias=False) if edge_attr_dim > 0 else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_linear.weight)
        nn.init.xavier_uniform_(self.attn_src.view(1, -1))
        nn.init.xavier_uniform_(self.attn_dst.view(1, -1))
        if self.edge_bias is not None:
            nn.init.xavier_uniform_(self.edge_bias.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.node_linear(x)
        num_nodes = h.size(0)
        if edge_index.numel() == 0:
            return h

        src = edge_index[0].long()
        dst = edge_index[1].long()
        logits = (h[src] * self.attn_src).sum(dim=-1) + (h[dst] * self.attn_dst).sum(dim=-1)
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = edge_attr.to(device=x.device, dtype=x.dtype)
            if self.edge_bias is not None:
                logits = logits + self.edge_bias(edge_attr).squeeze(-1)
            # The final edge weight is the last column in materialized edge_attr.
            logits = logits + edge_attr[:, -1]
        logits = F.leaky_relu(logits, negative_slope=0.2)

        out = h.new_zeros((num_nodes, h.size(1)))
        for node_idx in range(num_nodes):
            mask = dst == node_idx
            if not bool(mask.any()):
                continue
            alpha = torch.softmax(logits[mask], dim=0)
            messages = h[src[mask]] * alpha.unsqueeze(-1)
            out[node_idx] = messages.sum(dim=0)
        # Residual self signal prevents isolated nodes from becoming zero.
        return out + h


class EdgeAwareGAT(nn.Module):
    """Fixed-config edge-aware GAT that outputs one score per node."""

    def __init__(self, in_dim: int, edge_attr_dim: int = 0,
                 hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        layers = []
        current_dim = in_dim
        for _ in range(num_layers):
            layers.append(EdgeAwareGATLayer(current_dim, hidden_dim, edge_attr_dim=edge_attr_dim))
            current_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x
        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_attr=edge_attr)
            h = F.elu(h)
            if layer_idx < len(self.layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return self.scorer(h).squeeze(-1)

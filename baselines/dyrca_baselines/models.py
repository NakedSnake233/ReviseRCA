"""PyTorch baseline reproductions adapted from Eadro and HolisticRCA."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class MeanGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return F.elu(self.self_linear(x))
        source, target = edge_index[0], edge_index[1]
        messages = x[:, source, :]
        agg = torch.zeros_like(x)
        agg.index_add_(1, target, messages)
        degree = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
        agg = agg / degree.clamp_min(1.0).view(1, -1, 1)
        return F.elu(self.self_linear(x) + self.neigh_linear(agg))


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, steps, dims = x.shape
        flat = x.reshape(batch * nodes, steps, dims)
        _, h = self.net(flat)
        return h[-1].reshape(batch, nodes, -1)


class EadroRCA(nn.Module):
    """Eadro-style multi-source encoder plus service dependency graph locator."""

    def __init__(self, metric_dim: int, trace_dim: int, log_dim: int,
                 topology_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.metric_encoder = TemporalEncoder(metric_dim, hidden_dim)
        self.trace_encoder = TemporalEncoder(trace_dim, hidden_dim)
        self.log_encoder = nn.Sequential(nn.Linear(log_dim, hidden_dim), nn.ReLU())
        self.topology_encoder = nn.Sequential(nn.Linear(topology_dim, hidden_dim), nn.ReLU())
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GLU(dim=-1),
        )
        fused_dim = hidden_dim // 2
        self.graph1 = MeanGraphConv(fused_dim, hidden_dim)
        self.graph2 = MeanGraphConv(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)
        self.locator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        metrics = self.metric_encoder(batch["metrics"])
        traces = self.trace_encoder(batch["traces"])
        logs = self.log_encoder(batch["logs"])
        topology = self.topology_encoder(batch["topology"])
        x = self.fuse(torch.cat([metrics, traces, logs, topology], dim=-1))
        x = self.graph1(x, edge_index)
        x = self.graph2(x, edge_index)
        attn = torch.softmax(self.attn(x).squeeze(-1), dim=-1).unsqueeze(-1)
        graph_emb = (attn * x).sum(dim=1, keepdim=True).expand_as(x)
        return self.locator(torch.cat([x, graph_emb], dim=-1)).squeeze(-1)


class ModalTransformer(nn.Module):
    def __init__(self, input_steps: int, out_dim: int, heads: int = 4):
        super().__init__()
        self.project = nn.Linear(input_steps, out_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=heads,
            dim_feedforward=out_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, nodes, steps, features] -> tokens are modal features.
        batch, nodes, steps, features = x.shape
        tokens = x.permute(0, 1, 3, 2).reshape(batch * nodes, features, steps)
        out = self.encoder(self.project(tokens))
        return out.mean(dim=1).reshape(batch, nodes, -1)


class HolisticRCABaseline(nn.Module):
    """HolisticRCA-style modality learning, entity integration, and graph fusion."""

    def __init__(self, metric_steps: int, trace_steps: int, metric_dim: int,
                 trace_dim: int, log_dim: int, topology_dim: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.metric_repr = ModalTransformer(metric_steps, hidden_dim)
        self.trace_repr = ModalTransformer(trace_steps, hidden_dim)
        self.log_repr = nn.Sequential(nn.Linear(log_dim, hidden_dim), nn.ELU())
        self.topology_repr = nn.Sequential(nn.Linear(topology_dim, hidden_dim), nn.ELU())
        integration_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.integration = nn.TransformerEncoder(integration_layer, num_layers=1)
        self.entity_align = nn.Linear(hidden_dim * 4, hidden_dim)
        self.graph1 = MeanGraphConv(hidden_dim, hidden_dim)
        self.graph2 = MeanGraphConv(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        metric = self.metric_repr(batch["metrics"])
        trace = self.trace_repr(batch["traces"])
        log = self.log_repr(batch["logs"])
        topo = self.topology_repr(batch["topology"])
        batch_size, node_count, hidden = metric.shape
        modal_tokens = torch.stack([metric, trace, log, topo], dim=2)
        modal_tokens = modal_tokens.reshape(batch_size * node_count, 4, hidden)
        integrated = self.integration(modal_tokens).reshape(batch_size, node_count, 4 * hidden)
        x = F.elu(self.entity_align(integrated))
        x = self.graph1(x, edge_index)
        x = self.graph2(x, edge_index)
        return self.classifier(x).squeeze(-1)


def create_baseline_model(name: str, sample: Dict[str, torch.Tensor]) -> nn.Module:
    metric_dim = int(sample["metrics"].shape[-1])
    trace_dim = int(sample["traces"].shape[-1])
    log_dim = int(sample["logs"].shape[-1])
    topology_dim = int(sample["topology"].shape[-1])
    if name == "eadro":
        return EadroRCA(metric_dim, trace_dim, log_dim, topology_dim)
    if name == "holisticrca":
        return HolisticRCABaseline(
            metric_steps=int(sample["metrics"].shape[-2]),
            trace_steps=int(sample["traces"].shape[-2]),
            metric_dim=metric_dim,
            trace_dim=trace_dim,
            log_dim=log_dim,
            topology_dim=topology_dim,
        )
    raise ValueError(f"unknown baseline model: {name}")

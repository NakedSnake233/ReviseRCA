"""
Dynamic graph builder.
Constructs PyG-compatible graphs for pre/during/post phases.
"""
import logging

import torch
import numpy as np

from dataset.config_provider import get_config_from_metadata

log = logging.getLogger(__name__)


class DynamicServiceGraph:
    """Three-phase dynamic service graph with topology change encoding."""

    def __init__(self):
        # One PyG-like dict per phase
        self.graphs = [None, None, None]  # [pre, during, post]
        self.topo_change = None           # Tensor [C]
        self.change_mask = None           # Tensor [N] — which nodes changed
        self.edge_change_mask = None      # Tensor [E] — which edges changed

    def to_dict(self) -> dict:
        """Serialize to a plain dict for torch.save."""
        result = {}
        for i, phase in enumerate(["pre", "during", "post"]):
            g = self.graphs[i]
            if g is not None:
                result[f"{phase}_edge_index"] = g["edge_index"]
                result[f"{phase}_node_names"] = g["node_names"]
        result["topo_change"] = self.topo_change
        result["change_mask"] = self.change_mask
        result["edge_change_mask"] = self.edge_change_mask
        return result


def _build_phase_graph(topo: dict, node_feats_at_phase: torch.Tensor, cfg) -> dict:
    """Build a single-phase graph from topology snapshot.

    Args:
        topo: {nodes: [...], edges: [...]} from loader
        node_feats_at_phase: [N, F] node features for this phase
    Returns:
        dict with edge_index [2, E], edge_attr [E, F_e], node_names
    """
    SERVICES = cfg.SERVICES
    SERVICE_TO_IDX = cfg.SERVICE_TO_IDX
    KNOWN_EDGES = cfg.KNOWN_EDGES
    NUM_SERVICES = cfg.NUM_SERVICES
    # Use KNOWN_EDGES as the base edge set
    edge_index = []
    node_names = list(SERVICES)

    for src, dst in KNOWN_EDGES:
        si = SERVICE_TO_IDX[src]
        di = SERVICE_TO_IDX[dst]
        edge_index.append([si, di])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return {
        "edge_index": edge_index,
        "node_names": node_names,
        "num_nodes": NUM_SERVICES,
    }


def _compute_change_masks(topo_pre: dict, topo_during: dict, cfg) -> tuple:
    """Compute which nodes and edges changed between pre and during.

    Returns:
        change_mask: [N] — 1 if node changed
        edge_change_mask: [E] — 1 if edge changed
    """
    SERVICES = cfg.SERVICES
    SERVICE_TO_IDX = cfg.SERVICE_TO_IDX
    KNOWN_EDGES = cfg.KNOWN_EDGES
    NUM_SERVICES = cfg.NUM_SERVICES
    NUM_EDGES = cfg.NUM_EDGES

    change_mask = torch.zeros(NUM_SERVICES, dtype=torch.float32)
    edge_change_mask = torch.zeros(NUM_EDGES, dtype=torch.float32)

    if not topo_pre or not topo_during:
        return change_mask, edge_change_mask

    # Build lookup
    pre_nodes = {n["name"]: n for n in topo_pre.get("nodes", [])}
    during_nodes = {n["name"]: n for n in topo_during.get("nodes", [])}

    # Node changes
    for svc in SERVICES:
        pn = pre_nodes.get(svc)
        dn = during_nodes.get(svc)
        if pn is None or dn is None:
            if pn is not None or dn is not None:
                change_mask[SERVICE_TO_IDX[svc]] = 1.0
            continue

        # Check replica change
        if pn.get("replicas", 0) != dn.get("replicas", 0):
            change_mask[SERVICE_TO_IDX[svc]] = 1.0
            continue

        # Check version change / split
        pv = pn.get("version", "unknown")
        dv = dn.get("version", "unknown")
        if pv != dv:
            change_mask[SERVICE_TO_IDX[svc]] = 1.0
            continue

        # Check pod changes (different pod names)
        pre_pods = sorted(p.get("name", "") for p in pn.get("pods", []))
        during_pods = sorted(p.get("name", "") for p in dn.get("pods", []))
        if pre_pods != during_pods:
            change_mask[SERVICE_TO_IDX[svc]] = 1.0
            continue

        # Check IP changes
        pre_ips = set(pn.get("ips", []))
        during_ips = set(dn.get("ips", []))
        if pre_ips != during_ips:
            change_mask[SERVICE_TO_IDX[svc]] = 1.0

    # Edge changes: compare endpoint sets
    pre_eps = {}
    for svc in topo_pre.get("services", []):
        pre_eps[svc["name"]] = set(svc.get("endpoint_ips", []))
    during_eps = {}
    for svc in topo_during.get("services", []):
        during_eps[svc["name"]] = set(svc.get("endpoint_ips", []))

    for ei, (src, dst) in enumerate(KNOWN_EDGES):
        pre_e = pre_eps.get(dst, set())
        during_e = during_eps.get(dst, set())
        if pre_e != during_e:
            edge_change_mask[ei] = 1.0

    return change_mask, edge_change_mask


def build_dynamic_graph(sample, features, cfg=None) -> DynamicServiceGraph:
    """Build a DynamicServiceGraph from an ExperimentSample and its features."""
    if cfg is None:
        cfg = get_config_from_metadata(getattr(sample, '_exp_dir', ''))

    graph = DynamicServiceGraph()

    # Phase time boundaries (using time_window)
    tw = sample.time_window
    t_collect = tw.get("t_collect", "")
    t_inject = tw.get("t_inject", "")
    t_recover = tw.get("t_recover", "")
    t_end = tw.get("t_end", "")

    # Compute phase averages for node features
    phase_labels = features.phase_labels.numpy()
    node_feats = features.node_features  # [T, N, F]
    T, N, F = node_feats.shape

    # Average features per phase
    phase_node_feats = []
    for phase_id in [0, 1, 2]:
        mask = phase_labels == phase_id
        if mask.any():
            phase_avg = node_feats[mask].mean(dim=0)  # [N, F]
        else:
            phase_avg = torch.zeros(N, F)
        phase_node_feats.append(phase_avg)

    # Build graphs
    topos = [sample.topo_pre, sample.topo_during, sample.topo_post]
    for i, topo in enumerate(topos):
        if topo and topo.get("nodes"):
            g = _build_phase_graph(topo, phase_node_feats[i], cfg)
        else:
            g = _build_phase_graph({"nodes": [], "edges": []}, phase_node_feats[i], cfg)
        graph.graphs[i] = g

    # Change masks
    graph.change_mask, graph.edge_change_mask = _compute_change_masks(
        sample.topo_pre, sample.topo_during, cfg
    )

    # Topology change features
    graph.topo_change = features.topo_change

    return graph

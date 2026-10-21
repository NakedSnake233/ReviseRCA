import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

torch = pytest.importorskip("torch")

from dyrca_rca.models.config import (
    BATCH_SIZE,
    DROPOUT,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    FIXED_GNN_CONFIG,
    HIDDEN_DIM,
    LR,
    NUM_LAYERS,
    PAIRWISE_RANKING_COEF,
    SEED,
    WEIGHT_DECAY,
    set_seed,
)
from dyrca_rca.models.gat import EdgeAwareGAT
from dyrca_rca.models.graphsage import WeightedGraphSAGE
from dyrca_rca.models.losses import masked_node_cross_entropy, rca_node_loss


def _synthetic_graph():
    x = torch.tensor([
        [1.0, 0.0, 0.0, 0.2],
        [0.0, 1.0, 0.0, 0.8],
        [0.0, 0.0, 1.0, 0.1],
        [0.5, 0.0, 0.5, 0.4],
    ], dtype=torch.float32)
    edge_index = torch.tensor([
        [0, 1, 2, 0, 3],
        [1, 2, 1, 3, 1],
    ], dtype=torch.long)
    edge_attr = torch.tensor([
        [0.8, 0.2, 1.0, 0.3, 0.55],
        [0.3, 0.7, 0.6, 0.8, 0.61],
        [0.4, 0.1, 0.9, 0.2, 0.43],
        [0.2, 0.2, 1.0, 0.2, 0.39],
        [1.0, 0.8, 0.5, 1.0, 0.805],
    ], dtype=torch.float32)
    candidate_mask = torch.tensor([True, True, True, False])
    label = torch.tensor(1, dtype=torch.long)
    return x, edge_index, edge_attr, candidate_mask, label


def test_fixed_config_constants_are_not_search_space():
    assert HIDDEN_DIM == 128
    assert NUM_LAYERS == 2
    assert DROPOUT == 0.1
    assert LR == 3e-3
    assert WEIGHT_DECAY == 1e-5
    assert EPOCHS == 200
    assert BATCH_SIZE == 1
    assert SEED == 42
    assert EARLY_STOP_PATIENCE == 30
    assert PAIRWISE_RANKING_COEF == 1.0
    assert FIXED_GNN_CONFIG.hidden_dim == HIDDEN_DIM


def test_weighted_graphsage_forward_on_synthetic_graph():
    set_seed()
    x, edge_index, edge_attr, candidate_mask, label = _synthetic_graph()
    model = WeightedGraphSAGE(in_dim=x.size(1))

    scores = model(x, edge_index, edge_attr=edge_attr)
    loss, parts = rca_node_loss(scores, label, candidate_mask)

    assert scores.shape == (x.size(0),)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(loss)
    assert parts["pairwise_coef"] == 1.0


def test_edge_aware_gat_forward_on_synthetic_graph():
    set_seed()
    x, edge_index, edge_attr, candidate_mask, label = _synthetic_graph()
    model = EdgeAwareGAT(in_dim=x.size(1), edge_attr_dim=edge_attr.size(1))

    scores = model(x, edge_index, edge_attr=edge_attr)
    loss = masked_node_cross_entropy(scores, label, candidate_mask)

    assert scores.shape == (x.size(0),)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(loss)


def test_rca_loss_decreases_on_tiny_synthetic_example():
    set_seed()
    x, edge_index, edge_attr, candidate_mask, label = _synthetic_graph()
    model = WeightedGraphSAGE(in_dim=x.size(1))
    model.eval()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FIXED_GNN_CONFIG.lr,
        weight_decay=FIXED_GNN_CONFIG.weight_decay,
    )

    with torch.no_grad():
        initial_scores = model(x, edge_index, edge_attr=edge_attr)
        initial_loss, _ = rca_node_loss(initial_scores, label, candidate_mask)
        initial_value = float(initial_loss.item())

    for _ in range(60):
        optimizer.zero_grad()
        scores = model(x, edge_index, edge_attr=edge_attr)
        loss, _ = rca_node_loss(scores, label, candidate_mask)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_scores = model(x, edge_index, edge_attr=edge_attr)
        final_loss, _ = rca_node_loss(final_scores, label, candidate_mask)
        final_value = float(final_loss.item())

    assert final_value < initial_value

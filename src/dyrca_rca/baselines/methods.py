"""Source-guided RCA methods over the shared dynamic-evidence schema.

These are explicit adapters, not byte-for-byte executions of upstream pipelines.
They retain the ranking mechanisms while replacing dataset-specific readers and
graph discovery with the frozen observation-only evidence supplied to every method.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dyrca_rca.baselines.evidence import CandidateCase
from dyrca_rca.models.config import FIXED_GNN_CONFIG, set_seed
from dyrca_rca.models.losses import rca_node_loss
from dyrca_rca.interaction.temporal import FEATURES as TEMPORAL_FEATURES


BASELINES = ["nsigma", "baro", "circa", "cloudranger", "eadro", "holisticrca", "changeonly", "nodeonly"]
DISPLAY_NAMES = {
    "nsigma": "NSigma-Dyn",
    "baro": "BARO-Dyn",
    "circa": "CIRCA-Dyn",
    "cloudranger": "CloudRanger-Dyn",
    "eadro": "Eadro-Dyn",
    "holisticrca": "HolisticRCA-Dyn",
    "changeonly": "Change-only",
    "nodeonly": "Node-only",
}


def ranked_prediction(case: CandidateCase, scores: Mapping[str, float]) -> Dict[str, Any]:
    values = {name: float(scores.get(name, float("-inf"))) for name in case.node_names}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("non-finite baseline score for %s" % case.name)
    ranked = sorted(case.node_names, key=lambda name: (-values[name], name))
    return {
        "experiment": case.name,
        "incident_id": case.incident_id,
        "task": case.task,
        "prediction": {
            "experiment_id": case.incident_id,
            "ranked_node_ids": ranked,
            "scores": values,
            "top_score": values[ranked[0]] if ranked else None,
        },
    }


class FeatureReference:
    """Training-only standard and robust reference distributions."""

    def __init__(self) -> None:
        self.by_node: Dict[str, Dict[str, np.ndarray]] = {}
        self.global_stats: Dict[str, np.ndarray] = {}

    def fit(self, cases: Sequence[CandidateCase]) -> "FeatureReference":
        if not cases:
            raise ValueError("empty baseline training evidence")
        grouped: Dict[str, List[np.ndarray]] = {}
        all_rows = []
        for case in cases:
            for name, row in zip(case.node_names, case.x.numpy()):
                values = np.asarray(row, dtype=np.float64)
                grouped.setdefault(name, []).append(values)
                all_rows.append(values)
        self.global_stats = _stats(np.stack(all_rows, axis=0))
        self.by_node = {name: _stats(np.stack(rows, axis=0)) for name, rows in grouped.items()}
        return self

    def standard(self, name: str, values: np.ndarray) -> np.ndarray:
        stats = self.by_node.get(name, self.global_stats)
        return (np.asarray(values, dtype=np.float64) - stats["mean"]) / stats["std"]

    def robust(self, name: str, values: np.ndarray) -> np.ndarray:
        stats = self.by_node.get(name, self.global_stats)
        return (np.asarray(values, dtype=np.float64) - stats["median"]) / stats["iqr"]


def _stats(values: np.ndarray) -> Dict[str, np.ndarray]:
    std = values.std(axis=0)
    q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0], axis=0)
    iqr = q3 - q1
    std[std < 1e-6] = 1.0
    iqr[iqr < 1e-6] = 1.0
    return {
        "mean": values.mean(axis=0),
        "std": std,
        "median": median,
        "iqr": iqr,
    }


class NSigmaBaseline:
    """RCAEval NSigma mechanism: maximum positive standard score."""

    def __init__(self) -> None:
        self.reference = FeatureReference()

    def fit(self, cases: Sequence[CandidateCase]) -> "NSigmaBaseline":
        self.reference.fit(cases)
        return self

    def predict(self, case: CandidateCase) -> Dict[str, float]:
        return {name: float(np.max(self.reference.standard(name, row.numpy())))
                for name, row in zip(case.node_names, case.x)}


class BAROBaseline:
    """RCAEval BARO ranking head: maximum positive robust-scaled score."""

    def __init__(self) -> None:
        self.reference = FeatureReference()

    def fit(self, cases: Sequence[CandidateCase]) -> "BAROBaseline":
        self.reference.fit(cases)
        return self

    def predict(self, case: CandidateCase) -> Dict[str, float]:
        return {name: float(np.max(self.reference.robust(name, row.numpy())))
                for name, row in zip(case.node_names, case.x)}


class CIRCABaseline:
    """Regression-based hypothesis testing on the observed dependency graph."""

    def __init__(self) -> None:
        self.reference = FeatureReference()
        self.train_cases: Sequence[CandidateCase] = []
        self.models: Dict[Tuple[str, Tuple[str, ...]], Optional[Tuple[np.ndarray, np.ndarray]]] = {}

    def fit(self, cases: Sequence[CandidateCase]) -> "CIRCABaseline":
        self.reference.fit(cases)
        self.train_cases = list(cases)
        return self

    def predict(self, case: CandidateCase) -> Dict[str, float]:
        values = case.feature_map()
        children: Dict[str, List[str]] = {name: [] for name in case.node_names}
        for source, target, _ in case.directed_edges:
            if source in children and target in values:
                children[source].append(target)
        scores = {}
        for name in case.node_names:
            parents = tuple(sorted(set(children.get(name, []))))
            model = self._model(name, parents)
            if model is None:
                zscores = np.abs(self.reference.standard(name, values[name].numpy()))
            else:
                beta, residual_scale = model
                parent_values = np.stack([values[parent].numpy() for parent in parents], axis=0)
                design = np.concatenate([np.ones((1, parent_values.shape[1])), parent_values], axis=0)
                prediction = np.sum(beta * design, axis=0)
                zscores = np.abs(values[name].numpy() - prediction) / residual_scale
            scores[name] = float(np.max(zscores))
        return scores

    def _model(self, node: str, parents: Tuple[str, ...]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        key = (node, parents)
        if key in self.models:
            return self.models[key]
        if not parents:
            self.models[key] = None
            return None
        rows = []
        targets = []
        for case in self.train_cases:
            feature_map = case.feature_map()
            if node in feature_map and all(parent in feature_map for parent in parents):
                rows.append(np.stack([feature_map[parent].numpy() for parent in parents], axis=0))
                targets.append(feature_map[node].numpy())
        if len(rows) < max(5, len(parents) + 2):
            self.models[key] = None
            return None
        x = np.stack(rows, axis=0).astype(np.float64)  # cases x parents x features
        y = np.stack(targets, axis=0).astype(np.float64)
        feature_count = y.shape[1]
        beta = np.zeros((len(parents) + 1, feature_count), dtype=np.float64)
        residual_scale = np.ones(feature_count, dtype=np.float64)
        ridge = 1e-3
        for feature in range(feature_count):
            design = np.concatenate([np.ones((x.shape[0], 1)), x[:, :, feature]], axis=1)
            penalty = np.eye(design.shape[1]) * ridge
            penalty[0, 0] = 0.0
            coef = np.linalg.pinv(design.T.dot(design) + penalty).dot(design.T).dot(y[:, feature])
            residual = y[:, feature] - design.dot(coef)
            scale = float(residual.std())
            beta[:, feature] = coef
            residual_scale[feature] = scale if scale >= 1e-6 else 1.0
        self.models[key] = (beta, residual_scale)
        return self.models[key]


class CloudRangerBaseline:
    """Relation-weighted second-order random walk on each observed graph."""

    def __init__(self, seed: int = 42, beta: float = 0.3, rho: float = 0.2) -> None:
        self.seed = seed
        self.beta = beta
        self.rho = rho
        self.reference = FeatureReference()
        self.series: Dict[str, Dict[str, float]] = {}

    def fit(self, cases: Sequence[CandidateCase]) -> "CloudRangerBaseline":
        self.reference.fit(cases)
        for case in cases:
            self.series[case.name] = {
                name: float(np.linalg.norm(self.reference.standard(name, row.numpy())))
                for name, row in zip(case.node_names, case.x)
            }
        return self

    def predict(self, case: CandidateCase) -> Dict[str, float]:
        names = list(case.node_names)
        size = len(names)
        index = {name: idx for idx, name in enumerate(names)}
        anomaly = np.array([
            float(np.max(np.abs(self.reference.robust(name, row.numpy()))))
            for name, row in zip(names, case.x)
        ], dtype=np.float64)
        if anomaly.max() > 0:
            anomaly /= anomaly.max()
        sli = "frontend" if "frontend" in index else names[int(np.argmax(anomaly))]
        relation = np.array([self._correlation(sli, name) for name in names], dtype=np.float64)
        relation[index[sli]] = 1.0
        suspiciousness = np.maximum(relation, 0.05) * (0.05 + anomaly)

        access = np.zeros((size, size), dtype=np.float64)
        for source, target, weight in case.directed_edges:
            access[index[source], index[target]] = max(access[index[source], index[target]], weight)
        matrix = self._transition(access, suspiciousness)
        # Experiment identifiers can encode injection targets. Stochasticity
        # must depend only on the declared seed, never on that identifier.
        rng = np.random.RandomState(self.seed)
        counts = np.zeros(size, dtype=np.float64)
        start = index[sli]
        for _ in range(10):
            previous = current = start
            for _ in range(1000):
                probabilities = matrix[previous, current]
                total = probabilities.sum()
                if total <= 0:
                    probabilities = np.full(size, 1.0 / size)
                else:
                    probabilities = probabilities / total
                nxt = int(rng.choice(size, p=probabilities))
                counts[nxt] += 1.0
                previous, current = current, nxt
        return {name: float(counts[index[name]] / counts.sum()) for name in names}

    def _correlation(self, left: str, right: str) -> float:
        pairs = [(values[left], values[right]) for values in self.series.values()
                 if left in values and right in values]
        if len(pairs) < 5:
            return 0.0
        a, b = np.asarray(pairs, dtype=np.float64).T
        if a.std() < 1e-8 or b.std() < 1e-8:
            return 0.0
        value = float(np.corrcoef(a, b)[0, 1])
        return abs(value) if np.isfinite(value) else 0.0

    def _transition(self, access: np.ndarray, scores: np.ndarray) -> np.ndarray:
        size = access.shape[0]
        p = np.zeros((size, size), dtype=np.float64)
        for source in range(size):
            for target in range(size):
                if access[source, target] > 0:
                    p[source, target] = scores[target] * access[source, target]
            if p[source].sum() > 0:
                p[source] /= p[source].sum()
        matrix = np.zeros((size, size, size), dtype=np.float64)
        for previous in range(size):
            for current in range(size):
                for target in range(size):
                    if access[current, target] > 0:
                        matrix[previous, current, target] = (
                            (1.0 - self.beta) * p[previous, current]
                            + self.beta * p[current, target]
                        ) * access[current, target]
                forward = matrix[previous, current].sum()
                if forward > 0:
                    matrix[previous, current] /= forward
                incoming = [node for node in range(size)
                            if access[current, node] == 0 and access[node, current] > 0]
                for node in incoming:
                    matrix[previous, current, node] = self.rho * (
                        (1.0 - self.beta) * p[previous, current]
                        + self.beta * p[node, current]
                    ) * access[node, current]
                other = [node for node in range(size) if node != current]
                maximum = max([matrix[previous, current, node] for node in other] or [0.0])
                matrix[previous, current, current] = max(0.0, scores[current] - maximum)
                total = matrix[previous, current].sum()
                if total > 0:
                    matrix[previous, current] /= total
        return matrix


class ChangeOnlyBaseline:
    """Rank by observed temporal change magnitudes only.

    The positive scales are fitted on training evidence.  No graph adjacency,
    root-cause label, target service, or injection configuration is consumed.
    The implementation follows the frozen TOSEM E2 definition and records the
    temporal feature names in its training manifest.
    """

    def __init__(self) -> None:
        self.scales: Optional[np.ndarray] = None
        self.feature_names = list(TEMPORAL_FEATURES)
        self.indices = [i for i, name in enumerate(self.feature_names)
                        if name in {
                            "endpoint_added", "endpoint_removed", "endpoint_became_empty",
                            "pod_added", "pod_removed", "version_added", "version_removed",
                            "ready_drop", "ready_recovery", "service_appeared", "service_disappeared",
                        }]

    def fit(self, cases: Sequence[CandidateCase]) -> "ChangeOnlyBaseline":
        values = []
        for case in cases:
            if case.x.shape[1] < 25 + len(self.feature_names):
                values.append(np.zeros(len(self.indices), dtype=np.float64))
            else:
                values.append(case.x[:, [25 + i for i in self.indices]].numpy().reshape(-1, len(self.indices)))
        matrix = np.concatenate(values, axis=0) if values else np.zeros((1, len(self.indices)))
        positive = []
        for j in range(matrix.shape[1]):
            observed = matrix[:, j][matrix[:, j] > 0]
            positive.append(float(np.median(observed)) if observed.size else 1.0)
        self.scales = np.maximum(np.asarray(positive, dtype=np.float64), 1e-8)
        return self

    def predict(self, case: CandidateCase) -> Dict[str, float]:
        if self.scales is None:
            raise RuntimeError("change-only baseline is not fitted")
        if case.x.shape[1] < 25 + len(self.feature_names):
            scores = np.zeros(len(case.node_names), dtype=np.float64)
        else:
            raw = case.x[:, [25 + i for i in self.indices]].numpy()
            scores = np.maximum(raw, 0.0).dot(1.0 / self.scales)
        return {name: float(scores[index]) for index, name in enumerate(case.node_names)}


class WeightedMeanGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return F.elu(self.self_linear(x))
        source, target = edge_index[0].long(), edge_index[1].long()
        weight = edge_weight.to(dtype=x.dtype).view(-1)
        aggregate = x.new_zeros(x.shape)
        denominator = x.new_zeros((x.shape[0], 1))
        aggregate.index_add_(0, target, x[source] * weight.unsqueeze(-1))
        denominator.index_add_(0, target, weight.unsqueeze(-1))
        aggregate = aggregate / denominator.clamp_min(1e-8)
        return F.elu(self.self_linear(x) + self.neigh_linear(aggregate))


class EadroAdapterModel(nn.Module):
    """Multi-source encoding, graph aggregation, and node localization."""

    def __init__(self, hidden: int, extra_features: int = 0) -> None:
        super().__init__()
        self.structure = nn.Sequential(nn.Linear(20, hidden), nn.ReLU())
        self.trace = nn.Sequential(nn.Linear(5 + extra_features, hidden), nn.ReLU())
        self.temporal = nn.Sequential(nn.Linear(18, hidden), nn.ReLU())
        self.fuse = nn.Linear(hidden * 3, hidden * 2)
        self.graph1 = WeightedMeanGraphConv(hidden, hidden)
        self.graph2 = WeightedMeanGraphConv(hidden, hidden)
        self.attention = nn.Linear(hidden, 1)
        self.locator = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        modalities = torch.cat([
            self.structure(x[:, :20]), self.trace(torch.cat([x[:, 20:25], x[:, 43:]], dim=1)), self.temporal(x[:, 25:43])
        ], dim=-1)
        node = F.glu(self.fuse(modalities), dim=-1)
        node = self.graph1(node, edge_index, edge_weight)
        node = F.dropout(node, p=FIXED_GNN_CONFIG.dropout, training=self.training)
        node = self.graph2(node, edge_index, edge_weight)
        weights = torch.softmax(self.attention(node).squeeze(-1), dim=0).unsqueeze(-1)
        graph = (weights * node).sum(dim=0, keepdim=True).expand_as(node)
        return self.locator(torch.cat([node, graph], dim=-1)).squeeze(-1)


class HolisticRCAAdapterModel(nn.Module):
    """Modality representation/integration followed by relation-graph fusion."""

    def __init__(self, hidden: int, extra_features: int = 0) -> None:
        super().__init__()
        self.structure = nn.Sequential(nn.Linear(20, hidden), nn.ELU())
        self.trace = nn.Sequential(nn.Linear(5 + extra_features, hidden), nn.ELU())
        self.temporal = nn.Sequential(nn.Linear(18, hidden), nn.ELU())
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=FIXED_GNN_CONFIG.dropout,
            batch_first=True,
        )
        self.integration = nn.TransformerEncoder(layer, num_layers=1)
        self.align = nn.Linear(hidden * 3, hidden)
        self.graph1 = WeightedMeanGraphConv(hidden, hidden)
        self.graph2 = WeightedMeanGraphConv(hidden, hidden)
        self.classifier = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([
            self.structure(x[:, :20]), self.trace(torch.cat([x[:, 20:25], x[:, 43:]], dim=1)), self.temporal(x[:, 25:43])
        ], dim=1)
        node = F.elu(self.align(self.integration(tokens).reshape(x.shape[0], -1)))
        node = self.graph1(node, edge_index, edge_weight)
        node = F.dropout(node, p=FIXED_GNN_CONFIG.dropout, training=self.training)
        node = self.graph2(node, edge_index, edge_weight)
        return self.classifier(node).squeeze(-1)


class NodeOnlyAdapterModel(nn.Module):
    """Two-layer fixed-width MLP with no neighbor aggregation."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(FIXED_GNN_CONFIG.dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(FIXED_GNN_CONFIG.dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        del edge_index, edge_weight
        return self.net(x).squeeze(-1)


class NeuralBaseline:
    def __init__(self, name: str, seed: int = 42, epochs: Optional[int] = None) -> None:
        if name not in ("eadro", "holisticrca", "nodeonly"):
            raise ValueError("unknown neural baseline: %s" % name)
        self.name = name
        self.seed = seed
        self.epochs = FIXED_GNN_CONFIG.epochs if epochs is None else int(epochs)
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.model: Optional[nn.Module] = None
        self.training_record: Dict[str, Any] = {}

    def fit(self, train_cases: Sequence[CandidateCase], val_cases: Sequence[CandidateCase],
            train_labels: Mapping[str, str], val_labels: Mapping[str, str]) -> "NeuralBaseline":
        if not train_cases or not val_cases:
            raise ValueError("neural baseline requires train and validation evidence")
        torch.set_num_threads(1)
        set_seed(self.seed)
        values = torch.cat([case.x for case in train_cases], dim=0)
        self.mean = values.mean(dim=0)
        self.std = values.std(dim=0, unbiased=False)
        self.std[self.std < 1e-6] = 1.0
        hidden = FIXED_GNN_CONFIG.hidden_dim
        if self.name == "eadro":
            self.model = EadroAdapterModel(hidden, extra_features=max(0, int(train_cases[0].x.shape[1]) - 43))
        elif self.name == "holisticrca":
            self.model = HolisticRCAAdapterModel(hidden, extra_features=max(0, int(train_cases[0].x.shape[1]) - 43))
        else:
            self.model = NodeOnlyAdapterModel(int(train_cases[0].x.shape[1]), hidden)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=FIXED_GNN_CONFIG.lr,
            weight_decay=FIXED_GNN_CONFIG.weight_decay,
        )
        rng = random.Random(self.seed)
        best_state = None
        best_loss = None
        best_epoch = 0
        stale = 0
        epochs_run = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            shuffled = list(train_cases)
            rng.shuffle(shuffled)
            for case in shuffled:
                optimizer.zero_grad()
                scores = self._forward(case)
                label = _local_label(case, train_labels[case.name])
                loss, _ = rca_node_loss(scores, torch.tensor(label, dtype=torch.long))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
            val_loss = self._loss(val_cases, val_labels)
            epochs_run = epoch
            if best_loss is None or val_loss < best_loss - 1e-12:
                best_loss = val_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone()
                              for key, value in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= FIXED_GNN_CONFIG.early_stop_patience:
                    break
        if best_state is None:
            raise RuntimeError("neural baseline did not produce a checkpoint")
        self.model.load_state_dict(best_state, strict=True)
        self.model.eval()
        self.training_record = {
            "epochs_cap": self.epochs,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_loss),
            "fixed_config": dict(FIXED_GNN_CONFIG.__dict__),
        }
        return self

    def predict(self, case: CandidateCase) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("neural baseline is not fitted")
        self.model.eval()
        with torch.no_grad():
            scores = self._forward(case).detach().cpu().numpy()
        return {name: float(scores[index]) for index, name in enumerate(case.node_names)}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("neural baseline is not fitted")
        return {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

    def _forward(self, case: CandidateCase) -> torch.Tensor:
        if self.model is None or self.mean is None or self.std is None:
            raise RuntimeError("neural baseline is not initialized")
        x = (case.x - self.mean) / self.std
        return self.model(x, case.edge_index, case.edge_weight)

    def _loss(self, cases: Sequence[CandidateCase], labels: Mapping[str, str]) -> float:
        if self.model is None:
            raise RuntimeError("neural baseline is not initialized")
        self.model.eval()
        losses = []
        with torch.no_grad():
            for case in cases:
                scores = self._forward(case)
                label = _local_label(case, labels[case.name])
                loss, _ = rca_node_loss(scores, torch.tensor(label, dtype=torch.long))
                losses.append(float(loss.item()))
        return sum(losses) / len(losses)


def _local_label(case: CandidateCase, truth: str) -> int:
    if truth not in case.node_names:
        raise ValueError("training target %s missing from candidates for %s" % (truth, case.name))
    return case.node_names.index(truth)


def create_classical(name: str, seed: int = 42) -> Any:
    if name == "nsigma":
        return NSigmaBaseline()
    if name == "baro":
        return BAROBaseline()
    if name == "circa":
        return CIRCABaseline()
    if name == "cloudranger":
        return CloudRangerBaseline(seed=seed)
    if name == "changeonly":
        return ChangeOnlyBaseline()
    raise ValueError("unknown classical baseline: %s" % name)

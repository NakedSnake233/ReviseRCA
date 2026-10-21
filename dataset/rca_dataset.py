"""
RCA Dataset — PyTorch/PyG compatible dataset for root cause analysis.
"""
import json
import logging
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from dataset.config import DATA_DIR, PROCESSED_DIR, SERVICES, SERVICE_TO_IDX, CHANGE_TYPES
from dataset.config_provider import get_config_from_metadata
from dataset.loader import load_experiment
from dataset.feature_extractor import extract_features
from dataset.graph_builder import build_dynamic_graph

log = logging.getLogger(__name__)


class RCADataset(Dataset):
    """
    Dynamic topology fault RCA dataset.

    Each item contains:
      - node_features:   [T, N, F]   per-service time-series features
      - edge_features:   [T, E, Fe]  per-edge time-series features
      - topo_change:     [C]         topology change vector
      - phase_labels:    [T]         0=pre, 1=during, 2=post
      - change_mask:     [N]         which services changed
      - edge_change_mask:[E]         which edges changed
      - edge_index:      [2, E]      graph structure (shared across phases)
      - root_cause:      int         index of root cause service
      - change_type:     int         index of change type
      - exp_id:          str         experiment identifier
      - scenario:        str         T1/T2/T3/T4
    """

    def __init__(self, data_dir: str | Path = None, split: str | None = None,
                 train_ratio: float = 0.7, val_ratio: float = 0.15,
                 seed: int = 42):
        """
        Args:
            data_dir: Directory containing processed .pt files or raw experiment dirs.
            split: 'train', 'val', 'test', or None (all data).
            train_ratio, val_ratio: Split ratios.
            seed: Random seed for reproducibility.
        """
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.split = split

        # Find experiment directories
        self.exp_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith(("t1-", "t2-", "t3-", "t4-"))
        ])

        if not self.exp_dirs:
            raise FileNotFoundError(f"No experiment dirs found in {self.data_dir}")

        # Stratified split by scenario
        random.seed(seed)
        by_scenario = {}
        for d in self.exp_dirs:
            sid = d.name.split("-")[0].upper()  # t1 → T1
            by_scenario.setdefault(sid, []).append(d)

        train, val, test = [], [], []
        for sid, dirs in sorted(by_scenario.items()):
            random.shuffle(dirs)
            n = len(dirs)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train.extend(dirs[:n_train])
            val.extend(dirs[n_train:n_train + n_val])
            test.extend(dirs[n_train + n_val:])

        if split == "train":
            self.exp_dirs = train
        elif split == "val":
            self.exp_dirs = val
        elif split == "test":
            self.exp_dirs = test
        else:
            self.exp_dirs = train + val + test

        log.info(f"RCADataset: split={split}, size={len(self.exp_dirs)}")

    def __len__(self) -> int:
        return len(self.exp_dirs)

    def __getitem__(self, idx: int) -> dict:
        exp_dir = self.exp_dirs[idx]

        # Check for cached .pt file
        cache_path = exp_dir / "processed.pt"
        if cache_path.exists():
            return torch.load(cache_path, weights_only=False)

        # Load and process on the fly
        sample = load_experiment(exp_dir)
        features = extract_features(sample)
        graph = build_dynamic_graph(sample, features)

        item = {
            "node_features": features.node_features,       # [T, N, F]
            "edge_features": features.edge_features,       # [T, E, Fe]
            "topo_change": features.topo_change,            # [C]
            "phase_labels": features.phase_labels,          # [T]
            "change_mask": graph.change_mask,               # [N]
            "edge_change_mask": graph.edge_change_mask,     # [E]
            "edge_index": graph.graphs[0]["edge_index"],    # [2, E]
            "root_cause": features.root_cause_idx,          # int
            "change_type": features.change_type_idx,        # int
            "exp_id": sample.exp_id,
            "scenario": sample.scenario,
        }

        return item

    def get_label_distribution(self) -> dict:
        """Count labels across the dataset (without full processing)."""
        from collections import Counter
        scenarios = Counter()
        root_causes = Counter()
        for d in self.exp_dirs:
            meta_path = d / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    m = json.load(f)
                scenarios[m.get("scenario_id", "?")] += 1
                root_causes[m.get("target_service", "?")] += 1
        return {"scenarios": dict(scenarios), "root_causes": dict(root_causes)}

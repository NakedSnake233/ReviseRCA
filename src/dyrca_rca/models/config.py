"""Fixed model configuration for all RCA graph variants.

These constants are not a search space. They must remain shared across graph
variants so experiments compare graph update behavior rather than tuning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1
LR = 3e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 200
BATCH_SIZE = 1
SEED = 42
EARLY_STOP_PATIENCE = 30
PAIRWISE_RANKING_COEF = 1.0


@dataclass(frozen=True)
class FixedGNNConfig:
    hidden_dim: int = HIDDEN_DIM
    num_layers: int = NUM_LAYERS
    dropout: float = DROPOUT
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    seed: int = SEED
    early_stop_patience: int = EARLY_STOP_PATIENCE
    pairwise_ranking_coef: float = PAIRWISE_RANKING_COEF


FIXED_GNN_CONFIG = FixedGNNConfig()


def set_seed(seed: int = SEED) -> None:
    """Set deterministic seeds for small-model smoke tests and runs."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

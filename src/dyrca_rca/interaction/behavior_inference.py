"""Frozen inference for the versioned behavior checkpoints (no label reads)."""
from pathlib import Path
import json

import torch

from dyrca_rca.baselines.evidence import file_hash
from dyrca_rca.dataset.materialize import feature_names
from dyrca_rca.experiments.inference import predict_observed
from dyrca_rca.experiments.train import create_model
from dyrca_rca.interaction.behavior import FEATURES, PROTOCOL, augment, load_behavior
from dyrca_rca.interaction.observation import load_observation
from dyrca_rca.interaction.pipeline import replay
from dyrca_rca.interaction.state import new_session
from dyrca_rca.interaction.temporal import PROTOCOL as OLD_PROTOCOL
from dyrca_rca.models.config import FIXED_GNN_CONFIG


def load_behavior_checkpoint(path):
    path = Path(path)
    manifest_path = path.parent / "protocol.json"
    protocol = json.loads(manifest_path.read_text())
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    expected, _ = feature_names(OLD_PROTOCOL)
    expected += FEATURES
    if (protocol.get("protocol") != PROTOCOL or protocol.get("fixed_config") != FIXED_GNN_CONFIG.__dict__
            or checkpoint.get("protocol_sha256") != file_hash(manifest_path)
            or checkpoint.get("node_features") != expected
            or checkpoint.get("group") not in ("DY-RCA+behavior", "DY-RCA+behavior-separated")):
        raise ValueError("behavior checkpoint/schema/protocol mismatch")
    group = checkpoint["group"]
    report_path = path.parent / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["training"][group]["checkpoint_sha256"] != file_hash(path):
            raise ValueError("behavior checkpoint content hash mismatch")
    model = create_model("weighted_graphsage", len(expected), 8)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, group == "DY-RCA+behavior-separated", protocol


def predict_behavior_case(checkpoint, case_dir, dataset_id, system):
    """Rebuild the cutoff-scoped input and score it with a frozen checkpoint."""
    model, separated, protocol = load_behavior_checkpoint(checkpoint)
    if system != protocol["application"]:
        raise ValueError("application differs from checkpoint")
    view = load_observation(case_dir, dataset_id, system, max_rows=protocol["max_rows"])
    _, original = replay(new_session(view), feature_protocol=OLD_PROTOCOL)
    behavior = load_behavior(case_dir, max_rows=protocol["max_rows"])
    item = augment(original, behavior, separated)
    return predict_observed(model, item, "weighted_graphsage"), behavior["coverage"]

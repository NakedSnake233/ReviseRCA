"""Frozen-model inference, independent of labels and model training."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch

from dyrca_rca.dataset.materialize import feature_names
from dyrca_rca.experiments.train import create_model, _candidate_mask, _forward
from dyrca_rca.models.config import FIXED_GNN_CONFIG


def save_checkpoint(model, path, model_name, feature_protocol, training_metadata=None):
    node_names, edge_names = feature_names(feature_protocol)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": 1, "model_name": model_name,
                "fixed_config": asdict(FIXED_GNN_CONFIG),
                "node_features": node_names, "edge_features": edge_names,
                "feature_protocol": feature_protocol,
                "training_metadata": training_metadata or {}}
    # PyTorch 1.13's restricted unpickler does not support pickled floats.
    # Keep metadata in JSON, so weights_only=True works on both old and new torch.
    torch.save({"manifest_json": json.dumps(manifest, sort_keys=True, allow_nan=False),
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, str(target))


def load_checkpoint(path, feature_protocol):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    data = json.loads(checkpoint["manifest_json"])
    if feature_protocol == "auto":
        from dyrca_rca.dataset.materialize import INTERACTIVE_FEATURE_PROTOCOL
        from dyrca_rca.interaction.temporal import PROTOCOL
        if data.get("feature_protocol") not in (INTERACTIVE_FEATURE_PROTOCOL, PROTOCOL):
            raise ValueError("unsupported automatic checkpoint protocol")
        feature_protocol = data["feature_protocol"]
    if data.get("feature_protocol") != feature_protocol:
        raise ValueError("checkpoint feature protocol mismatch; train under the same observation protocol")
    node_names, edge_names = feature_names(feature_protocol)
    if (data.get("format_version") != 1 or data.get("node_features") != node_names
            or data.get("edge_features") != edge_names
            or data.get("fixed_config") != asdict(FIXED_GNN_CONFIG)):
        raise ValueError("checkpoint schema/config does not match fixed model")
    model = create_model(data["model_name"], len(node_names), len(edge_names))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.feature_protocol = feature_protocol
    model.eval()
    return model, data["model_name"]


def predict_observed(model, item, model_name):
    if getattr(model, "feature_protocol", item["metadata"].get("feature_protocol")) != item["metadata"].get("feature_protocol"):
        raise ValueError("model/input feature protocol mismatch")
    model.eval()
    with torch.no_grad():
        scores = _forward(model, item, model_name).detach().cpu()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("non-finite model scores")
    indices = torch.nonzero(_candidate_mask(item), as_tuple=False).view(-1).tolist()
    indices.sort(key=lambda i: (-float(scores[i]), item["node_ids"][i]))
    ranked = [item["node_ids"][i] for i in indices]
    return {"experiment_id": item["metadata"]["experiment_id"], "ranked_node_ids": ranked,
            "scores": {item["node_ids"][i]: float(scores[i]) for i in indices},
            "top_score": float(scores[indices[0]]) if indices else None}

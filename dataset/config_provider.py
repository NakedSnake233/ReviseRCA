"""
Config provider: returns the correct config module based on namespace.
"""
import sys
from pathlib import Path


def get_config(namespace: str = "boutique"):
    """Get the dataset config module for the given namespace."""
    if namespace == "trainticket":
        from dataset import config_trainticket
        return config_trainticket
    else:
        from dataset import config
        return config


def get_config_from_metadata(exp_dir):
    """Auto-detect config from experiment metadata.json."""
    import json
    meta_path = Path(exp_dir) / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        ns = meta.get("environment", {}).get("namespace", "boutique")
        return get_config(ns)
    return get_config("boutique")

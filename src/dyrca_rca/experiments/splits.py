"""Dataset splitting utilities for RCA experiments."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import pandas as pd


def make_split(frame: pd.DataFrame, split: str = "random", seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Create train/val/test splits from an index frame."""
    if split == "random":
        return random_stratified_split(frame, seed=seed)
    if split == "leave_service_out":
        return leave_service_out_split(frame, seed=seed)
    if split == "leave_scenario_out":
        return leave_scenario_out_split(frame, seed=seed)
    raise ValueError(f"unknown split: {split}")


def random_stratified_split(frame: pd.DataFrame, seed: int = 42,
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    train_idx = []
    val_idx = []
    test_idx = []
    scenario_col = "scenario_id" if "scenario_id" in frame.columns else None
    groups = frame.groupby(scenario_col).groups if scenario_col else {"all": list(frame.index)}
    for _, indices in sorted(groups.items(), key=lambda item: str(item[0])):
        idxs = list(indices)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n >= 3 and n_train == 0:
            n_train = 1
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])
    return {
        "train": frame.loc[train_idx].reset_index(drop=True),
        "val": frame.loc[val_idx].reset_index(drop=True),
        "test": frame.loc[test_idx].reset_index(drop=True),
    }


def leave_service_out_split(frame: pd.DataFrame, seed: int = 42) -> Dict[str, pd.DataFrame]:
    service_col = "target_service" if "target_service" in frame.columns else "root_cause_service"
    services = sorted(str(v) for v in frame[service_col].dropna().unique())
    if not services:
        return random_stratified_split(frame, seed=seed)
    rng = random.Random(seed)
    held_out = rng.choice(services)
    test_mask = frame[service_col].astype(str) == held_out
    train_val = frame.loc[~test_mask].reset_index(drop=True)
    base = random_stratified_split(train_val, seed=seed, train_ratio=0.85, val_ratio=0.15)
    return {
        "train": base["train"],
        "val": base["val"],
        "test": frame.loc[test_mask].reset_index(drop=True),
    }


def leave_scenario_out_split(frame: pd.DataFrame, seed: int = 42) -> Dict[str, pd.DataFrame]:
    if "scenario_id" not in frame.columns:
        return random_stratified_split(frame, seed=seed)
    scenarios = sorted(str(v) for v in frame["scenario_id"].dropna().unique())
    if not scenarios:
        return random_stratified_split(frame, seed=seed)
    rng = random.Random(seed)
    held_out = rng.choice(scenarios)
    test_mask = frame["scenario_id"].astype(str) == held_out
    train_val = frame.loc[~test_mask].reset_index(drop=True)
    base = random_stratified_split(train_val, seed=seed, train_ratio=0.85, val_ratio=0.15)
    return {
        "train": base["train"],
        "val": base["val"],
        "test": frame.loc[test_mask].reset_index(drop=True),
    }

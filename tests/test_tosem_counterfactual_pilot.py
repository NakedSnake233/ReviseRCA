import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyrca_rca.experiments.run_tosem_counterfactual_pilot import (
    change_ranking, feature_coverage, grouped_split,
)


def test_grouped_split_keeps_matched_change_patterns_together():
    records = [
        {"case_id": "a", "application": "boutique", "mode": "scale", "decoy": "d1", "decoy2": ""},
        {"case_id": "b", "application": "boutique", "mode": "scale", "decoy": "d1", "decoy2": ""},
        {"case_id": "c", "application": "boutique", "mode": "drift", "decoy": "d2", "decoy2": ""},
        {"case_id": "d", "application": "boutique", "mode": "split", "decoy": "d3", "decoy2": ""},
    ]
    split = grouped_split(records, 42)
    assert split == grouped_split(records, 42)
    assert sorted(sum(split.values(), [])) == ["a", "b", "c", "d"]
    assert next(part for part, cases in split.items() if "a" in cases) == next(
        part for part, cases in split.items() if "b" in cases)


def test_change_only_uses_only_structural_magnitudes_and_stable_ties():
    candidates = ["root", "decoy", "alpha"]
    features = {"root": {"trace_anomaly": 100.0}, "decoy": {"pod_added": 2.0}}
    scales = {field: 1.0 for field in (
        "endpoint_added", "endpoint_removed", "endpoint_became_empty",
        "pod_added", "pod_removed", "version_added", "version_removed",
        "ready_drop", "ready_recovery", "service_appeared", "service_disappeared")}
    result = change_ranking(candidates, features, scales)
    assert result["ranked_node_ids"] == ["decoy", "alpha", "root"]
    assert result["scores"]["root"] == 0.0


def test_feature_coverage_separates_eligible_objects_from_active_cases():
    features = {
        "a": {"s1": {"endpoint_pair_observed": 1, "endpoint_added": 1},
              "s2": {"endpoint_pair_observed": 1, "endpoint_added": 0}},
        "b": {"s1": {"endpoint_pair_observed": 0, "endpoint_added": 5}},
    }
    row = feature_coverage(features)["endpoint_change"]
    assert row == {"eligible_object_comparisons": 2, "eligible_cases": 1, "active_cases": 1}

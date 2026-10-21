import pytest

from dyrca_rca.experiments.rebalance_shortcut import bound_split, control_split, summarize


def test_related_controls_stay_together_across_modes_copies_and_repeats():
    metadata = {f"c{mode}-{service}-copy{copy}": {"target_service": service}
                for service in "abcde" for mode in range(1, 7) for copy in range(4)}
    split = control_split(metadata, 42)
    assert split == control_split(dict(reversed(list(metadata.items()))), 42)
    seen = {}
    for part, names in split.items():
        for n in names:
            root = metadata[n]["target_service"]
            assert seen.setdefault(root, part) == part
    assert sorted(sum(split.values(), [])) == sorted(metadata)
    assert all(split.values())


def test_invalid_grouping_fails_instead_of_splitting_repeated_incidents():
    with pytest.raises(ValueError):
        control_split({"c1-a": {"target_service": "a"}, "c2-a": {"target_service": "a"}}, 42)
    with pytest.raises(ValueError):
        control_split({"c1-a": {}}, 42)


def test_smoke_limits_retain_all_partitions_and_do_not_mutate_plan():
    original = {"train": ["a", "b", "c"], "val": ["d", "e"], "test": ["f", "g"]}
    selected = bound_split(original, 4)
    assert sum(map(len, selected.values())) == 4
    assert all(selected.values())
    assert original["train"] == ["a", "b", "c"]
    assert bound_split(original, None) == original
    with pytest.raises(ValueError):
        bound_split(original, 2)


def test_macro_average_does_not_treat_repeats_as_independent_configurations():
    rows = [{"experiment": name, "training_cohort": "mixed", "method": "DY-RCA",
             "hit1": hit, "hit3": 1., "mrr": 1. if hit else .5}
            for name, hit in [("c1-a-run1", 1.), ("c1-a-run2", 1.), ("c2-b-run1", 0.)]]
    value = next(r for r in summarize(rows) if r["test_stratum"] == "C1-C5")
    assert value["hit1"] == pytest.approx(2/3)
    assert value["configuration_macro_hit1"] == .5
    assert value["configuration_groups"] == 2

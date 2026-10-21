from dyrca_rca.experiments.optimize_harness import service_grouped_split


def test_service_group_split_is_deterministic_and_disjoint():
    names = [
        f"t1-{service}-run{i}" for service in ("a", "b", "c") for i in range(3)
    ] + [
        f"t2-{service}-run{i}" for service in ("a", "b", "c") for i in range(3)
    ]
    targets = {name: name.split("-")[1] for name in names}
    first = service_grouped_split(names, [], targets, 42, tasks=("t1", "t2"))
    second = service_grouped_split(names, [], targets, 42, tasks=("t1", "t2"))
    assert first == second
    service_sets = {
        part: {targets[name] for name in first[part]}
        for part in ("train", "val", "test")
    }
    assert service_sets["train"].isdisjoint(service_sets["val"])
    assert service_sets["train"].isdisjoint(service_sets["test"])
    assert service_sets["val"].isdisjoint(service_sets["test"])


def test_service_group_split_keeps_regression_out_of_training_partitions():
    names = [f"t1-{service}-run{i}" for service in ("a", "b", "c") for i in range(3)]
    targets = {name: name.split("-")[1] for name in names}
    regression = ["t1-a-run0"]
    split = service_grouped_split(names, regression, targets, 7, tasks=("t1",))
    assert split["regression"] == regression
    assert all(regression[0] not in split[part] for part in ("train", "val", "test"))

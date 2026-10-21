import pytest

from dyrca_rca.experiments.measure_tosem_stability import COMPARATORS, paired_interval, select_comparator


def test_validation_selects_comparator_without_using_test_outcomes():
    rows = []
    for index, method in enumerate(COMPARATORS):
        rows += [{"method": method, "split": "val", "experiment": "t3-cfg-run1", "mrr": .1 * index, "hit1": 0},
                 {"method": method, "split": "test", "experiment": "t3-test-run1", "mrr": 1 - .1 * index, "hit1": 1}]
    assert select_comparator(rows) == "HolisticRCA-Dyn"


def test_paired_bootstrap_resamples_configuration_groups_not_duplicate_runs():
    rows = []
    for repeat in range(20):
        rows += [{"method": "DY-RCA", "split": "test", "experiment": f"t3-a-run{repeat}", "mrr": 1},
                 {"method": "BARO-Dyn", "split": "test", "experiment": f"t3-a-run{repeat}", "mrr": 0}]
    rows += [{"method": "DY-RCA", "split": "test", "experiment": "t3-b-run1", "mrr": 0},
             {"method": "BARO-Dyn", "split": "test", "experiment": "t3-b-run1", "mrr": 0}]
    result = paired_interval(rows, "BARO-Dyn", seed=42)
    assert result["configuration_groups"] == 2
    assert result["delta_mrr"] == .5
    assert result == paired_interval(rows, "BARO-Dyn", seed=42)


def test_missing_paired_method_case_is_rejected():
    rows = [{"method": "DY-RCA", "split": "test", "experiment": "t3-a-run1", "mrr": 1}]
    with pytest.raises(ValueError, match="identical complete"):
        paired_interval(rows, "BARO-Dyn", seed=42)

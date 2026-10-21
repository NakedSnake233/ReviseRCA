from pathlib import Path

from dyrca_rca.collection_contamination import overlapping_failed_stress


def _report(tmp_path, success_namespace="boutique-c1", success_start="2026-09-30T11:13:00Z"):
    failure = {"scenario_id": "T1", "status": "failed", "fault_exit_code": 1,
               "output_dir": str(tmp_path / "failed"), "start_time": "2026-09-30T11:10:00Z",
               "end_time": "2026-09-30T11:12:00Z"}
    success = {"scenario_id": "T4", "status": "success", "output_dir": str(tmp_path / "success"),
               "start_time": success_start, "end_time": "2026-09-30T11:20:00Z"}
    return [{"settings": {"namespace": "boutique-c1", "seed": 42}, "experiments": [failure]},
            {"settings": {"namespace": success_namespace, "seed": 42}, "experiments": [success]}]


def test_failed_stress_overlap_quarantines_the_successful_export(tmp_path):
    rows = overlapping_failed_stress(_report(tmp_path))
    assert len(rows) == 1
    assert rows[0]["output_dir"] == str(tmp_path / "success")


def test_a_fault_in_another_copy_does_not_contaminate_this_copy(tmp_path):
    assert overlapping_failed_stress(_report(tmp_path, success_namespace="boutique-c2")) == []


def test_an_export_after_the_conservative_expiry_is_not_excluded(tmp_path):
    assert overlapping_failed_stress(_report(tmp_path, success_start="2026-09-30T11:16:00Z")) == []


def test_confirmed_cleanup_prevents_false_quarantine(tmp_path):
    directory = tmp_path / "failed"
    directory.mkdir()
    (directory / "fault_cleanup.done").touch()
    assert overlapping_failed_stress(_report(tmp_path)) == []

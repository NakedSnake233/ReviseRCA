import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dyrca_rca.experiments import publish_tosem_results as module
from dyrca_rca.experiments.run_tosem_downstream import _ranking_summary
from dyrca_rca.system_config import application_id


def test_shortcut_distinguishes_total_rows_from_scorable_faults():
    from dyrca_rca.experiments.run_tosem_downstream import _shortcut_summary
    rows = [{"experiment": "fault", "method": "DY-RCA", "split": "control", "hit1": 0, "mrr": .5},
            {"experiment": "benign", "method": "DY-RCA", "split": "control", "hit1": None, "mrr": None}]
    audits = {n: {"scenario_flags": {"multiple_changed_services": True}} for n in ("fault", "benign")}
    result = next(r for r in _shortcut_summary(rows, audits)["rows"] if r["method"] == "DY-RCA")
    assert result["n"] == result["n_total"] == 2
    assert result["scored_n"] == 1
    assert result["hit1"] == 0


def test_copy_identity_is_application_specific():
    assert application_id("boutique-c4") == "boutique"
    assert application_id("trainticket-c2") == "trainticket"
    assert application_id("boutique-c5") == "boutique-c5"
    assert application_id("unknown-c2") != "boutique"


def test_table_ranking_excludes_validation_and_controls():
    rows = [{"method": "DY-RCA", "split": "val", "hit1": 1, "hit3": 1, "mrr": 1},
            {"method": "DY-RCA", "split": "control", "hit1": 1, "hit3": 1, "mrr": 1},
            {"method": "DY-RCA", "split": "test", "hit1": 0, "hit3": 1, "mrr": .5}]
    report = _ranking_summary(rows)
    ours = next(row for row in report["methods"] if row["method"] == "DY-RCA")
    assert ours["n"] == 1 and ours["hit1"] == 0 and ours["mrr"] == .5


def test_pytest_testsuites_xml_preserves_actual_fixture_count(tmp_path):
    from dyrca_rca.experiments.run_tosem_downstream import _contract_summary
    path = tmp_path / "state.xml"
    path.write_text('<testsuites><testsuite tests="81" failures="0" errors="0"/></testsuites>')
    report = _contract_summary(path)
    assert report["tests_total"] == 81 and report["tests_passed"] == 81


def test_partial_collection_cannot_replace_manuscript_tables(tmp_path):
    index = tmp_path / "index"
    index.mkdir()
    (index / "collection_index.json").write_text('{"formal_ready":false,"limit":4}')
    paper = tmp_path / "paper"
    paper.mkdir()
    args = SimpleNamespace(data_root=str(tmp_path / "raw"), out_dir=str(tmp_path / "out"),
                           paper_dir=str(paper), system="boutique", seed=42, planner="rule",
                           limit=None, artifacts={"index": str(index)})
    with pytest.raises(ValueError, match="partial collection"):
        module.run(args)
    assert not (paper / "generated").exists()


def test_unavailable_ranking_is_not_filled_with_zero(tmp_path):
    index, audit = tmp_path / "index", tmp_path / "audit"
    index.mkdir()
    audit.mkdir()
    (index / "collection_index.json").write_text('{"formal_ready":true,"limit":null}')
    values = {"application": "boutique", "intended_n": 4, "scheduler_success_n": 4,
              "verified_n": 3, "unverified_n": 1, "services_median": 13,
              "nodes_median_pre": 13, "nodes_median_during": 14, "trace_rows_median": 200,
              "scenario_counts": {"T3": 3}}
    for family in ("endpoint", "pod", "version"):
        values["eligible_" + family + "_comparisons"] = 39
        values["active_" + family + "_cases"] = 2
    (audit / "e0_e1_summary.json").write_text(json.dumps({"applications": [values]}))
    (audit / "collection_audit.jsonl").write_text("")
    args = SimpleNamespace(data_root=str(tmp_path / "raw"), out_dir=str(tmp_path / "out"),
                           paper_dir=str(tmp_path / "paper"), system="boutique", seed=42, planner="rule",
                           limit=None, artifacts={"index": str(index), "audit": str(audit)})
    report = module.run(args)
    ranking = next(row for row in report["tables"] if row["table"] == 13)
    assert ranking["status"] == "pending"
    assert not any("ranking" in key for key in report["fields"])
    assert not (Path(args.out_dir) / "table13_new_ranking.tex").exists()
    assert (Path(args.paper_dir) / "generated/tosem_four_copies/boutique").is_symlink()
    assert report["source_hashes"]

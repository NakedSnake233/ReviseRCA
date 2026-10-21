import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dyrca_rca.experiments.backfill_revised_source import (
    Integrator, cluster_interval, frozen_stage, replace_registry, run, sha,
)


def test_registry_preserves_nested_tex_and_pending_values():
    text = "\\SetResult{ob.rank}{old}\n\\SetResult{tt.rank}{}\n"
    updated = replace_registry(text, {"ob.rank":r"95.56\%", "audit.binding":r"\texttt{cache\_id}"})
    assert r"\SetResult{ob.rank}{95.56\%}" in updated
    assert r"\SetResult{tt.rank}{}" in updated
    assert r"\SetResult{audit.binding}{\texttt{cache\_id}}" in updated
    assert replace_registry(updated, {"ob.rank":r"95.56\%", "audit.binding":r"\texttt{cache\_id}"}) == updated


def test_legacy_shortcut_denominator_is_rebuilt_from_scorable_predictions(tmp_path):
    downstream, ranking, out = (tmp_path / n for n in ("downstream", "ranking", "out"))
    for p in (downstream, ranking, out):
        p.mkdir()
    (downstream / "shortcut_summary.json").write_text(json.dumps({"rows": [
        {"subset": "multiple_changed_services", "method": "DY-RCA", "n": 2, "hit1": 0}]}))
    predictions = [{"experiment": n, "method": "DY-RCA", "split": "control", "hit1": h, "mrr": h}
                   for n, h in [("fault", 0), ("healthy", None)]]
    (ranking / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
    audits = [{"case_id": n, "verified": True, "scenario_flags": {
        "multiple_changed_services": True, "no_major_topology_change": False}}
              for n in ("fault", "healthy")]
    binder = Integrator(SimpleNamespace(paper_dir=tmp_path / "paper", out_dir=out))
    (downstream / "collection_audit.jsonl").write_text("".join(json.dumps(r) + "\n" for r in audits))
    binder.shortcuts("tt", {"11_downstream": downstream, "ranking_s42_t42": ranking, "02_audit": downstream}, audits)
    field = binder.fields["tt.shortcut_multi_change_dy_h1"]
    assert field["denominator"] == 1
    assert field["value"] == "0.00"
    assert "total rows=2" in field["note"]


def test_cluster_bootstrap_keeps_cluster_members_and_repeats():
    values = {"one":[1,1,1], "two":[0]}
    first = cluster_interval(values, seed=42)
    assert first == cluster_interval(values, seed=42)
    assert first["estimate"] == .75
    assert first["groups"] == 2 and first["replicates"] == 2000
    assert first["low"] == 0 and first["high"] == 1
    with pytest.raises(ValueError, match="eligible clusters"):
        cluster_interval({})


def test_missing_stage_is_pending_and_corrupt_receipt_rejected(tmp_path):
    assert frozen_stage(tmp_path, "boutique", "ranking_s42_t42") is None
    stage = tmp_path / "boutique/ranking_s42_t42"
    output = stage / "attempt-0001"
    output.mkdir(parents=True)
    source = output / "report.json"
    source.write_text('{"value":1}')
    (stage / "receipt.json").write_text(json.dumps({"seed":42,"app":"boutique", "stage":"ranking_s42_t42",
        "out":str(output),"outputs":{"report.json":sha(source)}}))
    assert frozen_stage(tmp_path,"boutique","ranking_s42_t42") == output
    source.write_text('{"value":2}')
    with pytest.raises(ValueError, match="Frozen result"):
        frozen_stage(tmp_path,"boutique","ranking_s42_t42")


def test_bounded_inventory_never_writes_manuscript(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    registry = paper / "pending_values.tex"
    registry.write_text(r"\SetResult{tt.rank}{}")
    args = SimpleNamespace(data_root=tmp_path/"raw", out_dir=tmp_path/"out", paper_dir=paper,
                           experiment_root=tmp_path/"experiments", seed=42, planner="rule", limit=2)
    report = run(args)
    assert report["manuscript_modified"] is False
    assert registry.read_text() == r"\SetResult{tt.rank}{}"
    assert not args.out_dir.exists()


def test_backfill_refuses_outputs_under_raw_data(tmp_path):
    args = SimpleNamespace(data_root=tmp_path, out_dir=tmp_path/"unsafe", paper_dir=tmp_path/"paper",
                           seed=42, planner="rule", limit=1)
    with pytest.raises(ValueError, match="outside raw data"):
        run(args)


@pytest.mark.parametrize('key,value',[('seed',7),('paired_conditions',5),('checkpoint_unchanged',False),('schema_version','unknown')])
def test_supplement_rejects_wrong_protocol_under_installed_schema_validator(tmp_path,key,value):
    from jsonschema import ValidationError
    stage=tmp_path/'stage';stage.mkdir()
    record={'status':'completed','system':'boutique','seed':42,'limit':None,'test_cases':45,'audit_cases':222,
            'paired_conditions':6,'paired_strata':2,'checkpoint_unchanged':True,'artifact_hashes':{'missing.json':'0'*64}}
    record[key]=value
    (stage/'completion_summary.json').write_text(json.dumps(record))
    binder=Integrator(SimpleNamespace(paper_dir=tmp_path/'paper',out_dir=tmp_path/'out'))
    # Reject before following any untrusted artifact paths, including on Draft4.
    with pytest.raises(ValidationError):binder.supplement('boutique',stage,None)

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dyrca_rca.experiments import index_four_copy_collection as indexer


def _fixture(tmp_path, monkeypatch, count=4):
    raw = tmp_path / "raw"
    raw.mkdir()
    identity = {"scenario_id": "T3", "target": "demo", "params": {}, "repeat": 1}
    monkeypatch.setattr(indexer, "_plan", lambda path, seed: {i: identity for i in range(1, 5)})
    plan = tmp_path / "plan.yaml"
    plan.write_text("frozen plan\n")
    for copy_id in range(1, count + 1):
        namespace = f"boutique-c{copy_id}"
        case = raw / namespace / f"t3-demo-c{copy_id}"
        for rel in ("metadata.json", "ground_truth.json", "topology/pre.json", "topology/during.json",
                    "topology/post.json", "traces/traces_raw.json", "logs/all.jsonl"):
            path = case / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
        for rel in ("metrics/request_rate.csv", "traces/trace_summary.csv"):
            path = case / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("timestamp,value\n1,2\n")
        record = dict(identity, index=copy_id, experiment_id=case.name, output_dir=str(case), status="success")
        report = {"settings": {"seed": 42, "namespace": namespace, "shard_index": copy_id - 1, "shard_count": 4},
                  "failed": 0, "pending": 0, "experiments": [record]}
        (raw / namespace / "batch_summary.json").write_text(json.dumps(report))
    return SimpleNamespace(data_root=raw, out_dir=tmp_path / "derived", system="boutique", seed=42,
                           planner="rule", limit=None, plan=plan, require_complete=False)


def test_complete_index_is_repeatable_and_leaves_sources_unchanged(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    snapshots = {path: path.read_bytes() for path in args.data_root.rglob("*") if path.is_file()}
    original = Path.read_text
    def bounded_read(path, *a, **kw):
        if path.name == "traces_raw.json":
            pytest.fail("Indexing must not materialize raw trace data")
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, "read_text", bounded_read)
    for _ in range(2):
        result = indexer.run(args)
        assert result["formal_ready"]
        assert result["complete_exported_cases"] == 4
    links = list((args.out_dir / "cases").glob("t3-*"))
    assert len(links) == 4 and all(path.is_symlink() for path in links)
    assert all(path.read_bytes() == before for path, before in snapshots.items())


def test_validation_subset_cannot_be_promoted_to_formal_input(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch, count=1)
    args.require_complete = True
    with pytest.raises(RuntimeError, match="incomplete"):
        indexer.run(args)
    assert not args.out_dir.exists()
    args.require_complete = False
    result = indexer.run(args)
    assert not result["formal_ready"]
    summary = json.loads((args.out_dir / "cases" / "batch_summary.json").read_text())
    assert summary["pending"] == 3


@pytest.mark.parametrize("field,value", [("seed", 123), ("namespace", "boutique-c2")])
def test_index_refuses_mixed_protocol_or_wrong_copy(tmp_path, monkeypatch, field, value):
    args = _fixture(tmp_path, monkeypatch)
    path = args.data_root / "boutique-c1" / "batch_summary.json"
    report = json.loads(path.read_text())
    report["settings"][field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Incompatible"):
        indexer.run(args)


def test_case_limit_remains_preview_even_if_all_shards_are_complete(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    args.limit = 1
    result = indexer.run(args)
    assert not result["formal_ready"]
    assert result["complete_exported_cases"] == 1


def test_quarantined_case_blocks_formal_indexing(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    args.exclusions = tmp_path / "exclusions.json"
    args.exclusions.write_text(json.dumps({"excluded_output_dirs": [str(args.data_root / "boutique-c1" / "t3-demo-c1")]}))
    result = indexer.run(args)
    assert not result["formal_ready"]
    assert result["complete_exported_cases"] == 3

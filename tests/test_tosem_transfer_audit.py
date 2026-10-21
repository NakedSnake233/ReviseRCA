import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyrca_rca.experiments.audit_tosem_transfer import audit


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary():
    return {"normal": {"n": 33, "hit1": 0.0, "hit3": 0.0, "mrr": 0.1,
                       "decoy_top1_rate": 0.5},
            "benign": {"n": 8, "decoy_top1_rate": 0.5}}


def test_transfer_audit_rejects_candidate_mismatch(tmpdir):
    root = Path(str(tmpdir))
    pilot, transfer = root / "pilot", root / "transfer"
    ids = ["incident-%02d" % i for i in range(41)]
    anchors = [{"case_id": item, "method": "DY-RCA-frozen-v3",
                "observation_version": item + "-view", "ranked_node_ids": ["a", "b"]}
               for item in ids]
    anchor_hash = _jsonl(pilot / "ranking" / "predictions.jsonl", anchors)
    _write(pilot / "collection" / "integrity_checks.json",
           {"prediction_sha256_before_labels": anchor_hash})
    _write(pilot / "ranking" / "summary.json",
           {"DY-RCA-frozen-v3": _summary(), "Change-only": _summary()})
    methods = ["NSigma-Dyn", "BARO-Dyn", "CIRCA-Dyn", "CloudRanger-Dyn",
               "Eadro-Dyn", "HolisticRCA-Dyn"]
    predictions = [{"incident_id": item, "method": method,
                    "observation_version": item + "-view",
                    "prediction": {"ranked_node_ids": ["a", "b"]}}
                   for method in methods for item in ids]
    transfer_hash = _jsonl(transfer / "predictions.jsonl", predictions)
    _write(transfer / "report.json", {
        "prediction_sha256_before_labels": transfer_hash,
        "summary": {method: _summary() for method in methods},
    })
    assert audit(pilot, transfer, 41)["shared_candidate_inventories"]

    predictions[0]["prediction"]["ranked_node_ids"] = ["a", "c"]
    transfer_hash = _jsonl(transfer / "predictions.jsonl", predictions)
    _write(transfer / "report.json", {
        "prediction_sha256_before_labels": transfer_hash,
        "summary": {method: _summary() for method in methods},
    })
    with pytest.raises(ValueError, match="candidate inventory"):
        audit(pilot, transfer, 41)

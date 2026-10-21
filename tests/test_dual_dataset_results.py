import csv
import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "aggregate_dual_dataset_results", Path(__file__).parents[1] / "scripts" / "aggregate_dual_dataset_results.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)
METHODS, load_run, make_tex = _MODULE.METHODS, _MODULE.load_run, _MODULE.make_tex


def _run(root: Path, system: str) -> Path:
    path = root / system
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"system": system}))
    fields = ["method", "test_n", "test_hit1", "test_hit3", "test_mrr",
              "regression_n", "regression_hit1", "regression_hit3", "regression_mrr"]
    with (path / "main_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, method in enumerate(METHODS):
            writer.writerow({"method": method, "test_n": 10,
                             "test_hit1": 0.5 + i / 100.0,
                             "test_hit3": 0.6 + i / 100.0,
                             "test_mrr": 0.4 + i / 100.0,
                             "regression_n": 10, "regression_hit1": 0.5,
                             "regression_hit3": 0.6, "regression_mrr": 0.4})
    return path


def test_dual_dataset_loader_and_tex(tmpdir):
    root = Path(str(tmpdir))
    boutique = load_run(_run(root, "boutique"), "boutique")
    train = load_run(_run(root, "trainticket"), "trainticket")
    tex = make_tex([boutique, train])
    assert "Online Boutique" in tex
    assert "Train Ticket" in tex
    assert tex.count("DY-RCA") == 1

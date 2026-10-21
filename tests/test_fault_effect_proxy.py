import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyrca_rca.experiments.audit_fault_effect_proxy import phase


def test_phase_requires_span_entirely_within_observation_interval():
    assert phase(2.0, 1.0, 0.0, 10.0, 20.0) == "pre"
    assert phase(10.0, 2.0, 0.0, 10.0, 20.0) == "during"
    assert phase(9.9, 0.2, 0.0, 10.0, 20.0) is None
    assert phase(19.9, 0.2, 0.0, 10.0, 20.0) is None
    assert phase(-1.0, 0.1, 0.0, 10.0, 20.0) is None

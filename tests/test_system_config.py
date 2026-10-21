import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyrca_rca.interaction.observation import dependency_edges


def test_train_ticket_registry_is_distinct_and_stable():
    boutique = dependency_edges("boutique")
    trainticket = dependency_edges("trainticket")
    assert len(boutique) == 13
    assert len(trainticket) >= 70
    assert all(source.startswith("ts-") and target.startswith("ts-")
               for source, target in trainticket)
    assert not set(map(tuple, boutique)) & set(map(tuple, trainticket))


def test_unknown_system_is_rejected():
    try:
        dependency_edges("unknown")
    except ValueError as exc:
        assert "no dependency configuration" in str(exc)
    else:
        raise AssertionError("unknown system unexpectedly accepted")

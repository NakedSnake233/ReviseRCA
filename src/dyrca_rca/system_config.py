"""Dependency registries shared by the two supported microservice systems.

Keeping the registry in one small module prevents experiment runners from
silently falling back to Online Boutique edges when they process TrainTicket.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from dyrca_rca.planner.signals import KNOWN_EDGES as BOUTIQUE_EDGES

try:
    # Load the standalone configuration file directly.  Importing the
    # ``dataset`` package would execute its legacy loader, which requires a
    # newer interpreter than some of the repository's lightweight test jobs.
    _config_path = Path(__file__).resolve().parents[2] / "dataset" / "config_trainticket.py"
    _spec = importlib.util.spec_from_file_location("_dyrca_trainticket_config", _config_path)
    _module = importlib.util.module_from_spec(_spec)
    assert _spec and _spec.loader
    _spec.loader.exec_module(_module)
    TRAINTICKET_EDGES = list(_module.KNOWN_EDGES)
except (ImportError, OSError, AttributeError, AssertionError):  # pragma: no cover
    TRAINTICKET_EDGES = []


SYSTEM_EDGES = {
    "boutique": [list(edge) for edge in BOUTIQUE_EDGES],
    "trainticket": [list(edge) for edge in TRAINTICKET_EDGES],
}


def application_id(namespace: str) -> str:
    """Treat an isolated collection copy as its application, retaining raw IDs."""
    return re.sub(r"-c[1-4]$", "", str(namespace))


def dependency_edges(system_id: str, override: dict | None = None) -> list[list[str]]:
    """Return a validated, stable edge list for ``system_id``."""
    registry = dict(SYSTEM_EDGES)
    if override:
        registry.update(override)
    edges = registry.get(system_id)
    if edges is None:
        raise ValueError("no dependency configuration for system: " + system_id)
    clean = []
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError("dependency configuration must contain service pairs")
        source, target = edge
        if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
            raise ValueError("dependency configuration must contain non-empty service names")
        clean.append([source, target])
    return [list(edge) for edge in sorted(set(tuple(e) for e in clean))]

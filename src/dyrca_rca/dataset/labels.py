"""Label helpers for service-level RCA materialization."""

from __future__ import annotations

from typing import Any, Dict, List


def root_service_from_row(row: Dict[str, Any]) -> str:
    """Resolve the service-level RCA label from an index row."""
    return str(
        row.get("root_cause_service")
        or row.get("target_service")
        or ""
    )


def node_service(node: Dict[str, Any]) -> str:
    """Best-effort service identity for a materialized node."""
    node_id = str(node.get("id", "") or "")
    attrs = node.get("attrs", {}) if isinstance(node.get("attrs"), dict) else {}
    if attrs.get("service"):
        return str(attrs["service"])
    if node_id.startswith("blackhole:"):
        return node_id.split(":", 1)[1]
    if node_id.startswith("pod:"):
        return node_id[4:].rsplit("-", 2)[0]
    if ":" in node_id:
        return node_id.split(":", 1)[0]
    return node_id


def root_node_index(nodes: List[Dict[str, Any]], root_service: str) -> int:
    """Return service node index for a root cause service, falling back to matches."""
    if not root_service:
        return -1
    for idx, node in enumerate(nodes):
        if str(node.get("id", "")) == root_service:
            return idx
    for idx, node in enumerate(nodes):
        if node_service(node) == root_service:
            return idx
    return -1

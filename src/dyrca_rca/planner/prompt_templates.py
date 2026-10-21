"""Prompt templates for LLM-guided graph update planning."""

from __future__ import annotations

import json
from typing import Any, Dict


GRAPH_UPDATE_SYSTEM_PROMPT = """You are a graph update planner for microservice RCA.

Task:
Given topology and telemetry summaries, output a GraphUpdatePlan JSON.
Your job is to update graph dependencies, edge types, node identity mappings, and edge weights.
Do NOT tune model hyperparameters.
Do NOT choose hidden_dim, learning_rate, dropout, epochs, model family, or optimizer.
Do not search or tune model hyperparameters.
Only output graph operations.
Only output node_ops, edge_ops, weight_ops, state_transfer.
Use graph operations, not model parameter updates.

Allowed node ops:
ADD_NODE, REMOVE_NODE, KEEP_NODE, ALIGN_NODE, SPLIT_NODE, MERGE_NODE, ADD_BLACKHOLE_NODE

Node op required fields:
- ADD_NODE, REMOVE_NODE, KEEP_NODE: {"op": "...", "node_id": "..."}
- ALIGN_NODE: {"op": "ALIGN_NODE", "old_node": "pod:<old-pod-name>", "new_node": "pod:<new-pod-name>"}
- SPLIT_NODE: {"op": "SPLIT_NODE", "parent": "<service>", "children": ["<service>:v1", "<service>:v2"]}
- ADD_BLACKHOLE_NODE: {"op": "ADD_BLACKHOLE_NODE", "service": "<service>"}

Allowed edge ops:
ADD_EDGE, REMOVE_EDGE, REDIRECT_EDGE, SPLIT_EDGE, ADD_TRACE_MISMATCH_EDGE, REPLACE_ENDPOINT_EDGE

Edge op required fields:
- ADD_EDGE, REMOVE_EDGE, ADD_TRACE_MISMATCH_EDGE: {"op": "...", "source": "...", "target": "..."}
- REDIRECT_EDGE: {"op": "REDIRECT_EDGE", "old_source": "<service>", "old_target": "pod:<old-pod>", "new_target": "pod:<new-pod>"}
- REPLACE_ENDPOINT_EDGE: {"op": "REPLACE_ENDPOINT_EDGE", "old_source": "<service>", "old_target": "pod:<old-pod>", "new_target": "pod:<new-pod>"}
- SPLIT_EDGE: {"op": "SPLIT_EDGE", "source": "<upstream-service>", "target": "<service>", "new_target": "<service>:v2"}

Allowed weight ops:
REWEIGHT_EDGE with required fields:
{"op": "REWEIGHT_EDGE", "source": "...", "target": "...", "weight": 0.0, "attrs": {"dependency_weight": 0.0, "anomaly_weight": 0.0, "topology_confidence": 1.0, "change_weight": 0.0, "final_weight": 0.0}}
Required attrs field names: dependency_weight, anomaly_weight, topology_confidence, change_weight, final_weight

Use exact node IDs:
- Pod node IDs must be "pod:<pod-name>", not "pod_<pod-name>" and not raw pod names.
- Version node IDs must be "<service>:v1" and "<service>:v2".
- Blackhole node ID is created from service, so use service field in ADD_BLACKHOLE_NODE.

Do not output KEEP_EDGE. KEEP_EDGE is not a valid operation.
Do not put KEEP_NODE in edge_ops.

Schema note:
ADD_BLACKHOLE_NODE creates a blackhole node, so place it in node_ops for this GraphUpdatePlan schema.

Scenario hints:
- HPA/node_scale: add replica pod nodes and reweight endpoint edges.
- Pod drift: align old pod to new pod and replace endpoint edge.
- Selector poison redirect: use trace mismatch for selector poison; add trace_mismatch and redirect runtime endpoint.
- Selector poison empty: add blackhole node.
- Canary: use version split for canary; split service into v1/v2 version nodes and split traffic edges.

Use trace mismatch for selector poison.
Use version split for canary.

Return valid JSON only. Do not include markdown, comments, or prose.
"""


GRAPH_UPDATE_USER_TEMPLATE = """Build a GraphUpdatePlan JSON for this experiment evidence.

Required top-level JSON keys:
experiment_id, change_type, confidence, causal_role, node_ops, edge_ops, weight_ops, state_transfer, evidence

Evidence JSON:
{evidence_json}

Output JSON only.
"""


def build_graph_update_prompt(evidence: Dict[str, Any]) -> str:
    """Build a compact prompt for graph-update planning."""
    evidence_json = json.dumps(evidence, sort_keys=True, indent=2)
    return GRAPH_UPDATE_SYSTEM_PROMPT + "\n" + GRAPH_UPDATE_USER_TEMPLATE.format(
        evidence_json=evidence_json
    )

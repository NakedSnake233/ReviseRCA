"""Rule-based GraphUpdatePlan generation from topology change signals."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dyrca_rca.graph.update_ops import (
    ADD_BLACKHOLE_NODE,
    ADD_EDGE,
    ADD_NODE,
    ADD_TRACE_MISMATCH_EDGE,
    ALIGN_NODE,
    REDIRECT_EDGE,
    REPLACE_ENDPOINT_EDGE,
    REWEIGHT_EDGE,
    SPLIT_EDGE,
    SPLIT_NODE,
    EdgeOp,
    GraphUpdatePlan,
    NodeOp,
    WeightOp,
)
from dyrca_rca.planner.signals import (
    detect_edge_redirect,
    detect_edge_removal,
    detect_node_drift,
    detect_node_scale,
    detect_version_split,
    pod_node_id,
    read_json,
    scenario_hint,
    service_name_from_metadata,
    trace_mismatch_edges,
    upstream_services,
)


def plan_experiment(experiment_dir: str, g_pre: Optional[Dict[str, Any]] = None) -> GraphUpdatePlan:
    """Build a rule-based graph update plan for one experiment directory."""
    exp_dir = Path(experiment_dir)
    metadata = read_json(exp_dir / "metadata.json", required=True)
    topology_diff = read_json(exp_dir / "topology_diff.json", required=False)
    pre = read_json(exp_dir / "topology" / "pre.json", required=True)
    during = read_json(exp_dir / "topology" / "during.json", required=True)
    # Ground truth is loaded only as evaluation metadata, not used in decisions.
    ground_truth = read_json(exp_dir / "ground_truth.json", required=False)

    experiment_id = str(metadata.get("experiment_id") or exp_dir.name)
    target_service = service_name_from_metadata(metadata)
    hints = scenario_hint(metadata, topology_diff)
    trace_path = exp_dir / "traces" / "trace_summary.csv"

    node_ops = []
    edge_ops = []
    weight_ops = []
    evidence = [{
        "source": "metadata",
        "scenario_id": metadata.get("scenario_id", ""),
        "scenario_name": metadata.get("scenario_name", ""),
        "target_service": target_service,
    }]
    if ground_truth:
        evidence.append({
            "source": "ground_truth_eval_metadata",
            "experiment_id": ground_truth.get("experiment_id", ""),
            "root_cause_service": ground_truth.get("root_cause_service", ""),
            "topology_change_type": ground_truth.get("topology_change_type", ""),
        })

    change_type = "no_change"
    confidence = 0.0
    causal_role = "unknown"

    # T3 empty endpoints should be checked before redirect because an empty
    # backend is the stronger service-level failure mode.
    removal = detect_edge_removal(during, target_service) if target_service else {"matched": False}
    redirect = detect_edge_redirect(pre, during, target_service) if target_service else {"matched": False}
    split = detect_version_split(during, metadata, target_service) if target_service else {"matched": False}
    drift = detect_node_drift(pre, during, target_service) if target_service else {"matched": False}
    scale = detect_node_scale(pre, during, target_service) if target_service else {"matched": False}

    if _prefer_edge_removal(hints, removal):
        change_type = "edge_removal"
        confidence = 0.9
        causal_role = "cause_of_fault"
        node_ops.append(NodeOp(
            op=ADD_BLACKHOLE_NODE,
            service=target_service,
            attrs={"reason": "empty_endpoints_or_unreachable_backend"},
        ))
        edge_ops.append(EdgeOp(
            op=ADD_EDGE,
            source=target_service,
            target=f"blackhole:{target_service}",
            edge_type="blackhole",
            weight=1.0,
            attrs={"reason": "empty_endpoints_or_unreachable_backend"},
        ))
        evidence.append({"source": "topology", "signal": "edge_removal", "details": removal})

    elif _prefer_edge_redirect(hints, redirect):
        change_type = "edge_redirect"
        confidence = 0.9
        causal_role = "cause_of_fault"
        old_target = _first_old_backend_node(redirect)
        for owner, pod in redirect.get("wrong_backends", []):
            wrong_node = pod_node_id(str(pod.get("name", "")))
            edge_ops.append(EdgeOp(
                op=REDIRECT_EDGE,
                old_source=target_service,
                old_target=old_target,
                new_target=wrong_node,
                edge_type="endpoint",
                attrs={
                    "declared_service": target_service,
                    "actual_service": owner,
                    "actual_pod_ip": pod.get("ip", ""),
                },
            ))
        for mismatch in trace_mismatch_edges(trace_path):
            edge_ops.append(EdgeOp(
                op=ADD_TRACE_MISMATCH_EDGE,
                source=mismatch["source"],
                target=mismatch["observed"],
                edge_type="trace_mismatch",
                attrs={"declared_dst": mismatch["declared"], "observed_dst": mismatch["observed"]},
            ))
        evidence.append({"source": "topology", "signal": "edge_redirect", "details": redirect})

    elif _prefer_version_split(hints, split):
        change_type = "edge_split"
        confidence = 0.88
        causal_role = "cause_or_mixed"
        children = split.get("children", [])
        route_weights = split.get("route_weights", {})
        node_ops.append(NodeOp(
            op=SPLIT_NODE,
            parent=target_service,
            children=children,
            attrs={"reason": "version_split_or_canary"},
        ))
        for upstream in upstream_services(target_service, g_pre):
            for child in children:
                version = child.rsplit(":", 1)[-1] if ":" in child else child
                weight = route_weights.get(version)
                edge_ops.append(EdgeOp(
                    op=SPLIT_EDGE,
                    source=upstream,
                    target=target_service,
                    new_target=child,
                    edge_type="dependency",
                    weight=weight,
                    attrs={"route_version": version, "route_weight": weight},
                ))
                if weight is not None:
                    weight_ops.append(WeightOp(
                        op=REWEIGHT_EDGE,
                        source=upstream,
                        target=child,
                        weight=weight,
                        attrs={"weight_source": "metadata.fault_params"},
                    ))
        evidence.append({"source": "topology", "signal": "edge_split", "details": split})

    elif _prefer_node_drift(hints, drift):
        change_type = "node_drift"
        confidence = 0.85
        causal_role = "topology_event"
        for old_pod, new_pod in drift.get("pairs", []):
            old_node = pod_node_id(str(old_pod.get("name", "")))
            new_node = pod_node_id(str(new_pod.get("name", "")))
            node_ops.append(NodeOp(
                op=ALIGN_NODE,
                old_node=old_node,
                new_node=new_node,
                service=target_service,
                attrs={
                    "service": target_service,
                    "old_ip": old_pod.get("ip", ""),
                    "new_ip": new_pod.get("ip", ""),
                    "reason": "pod_replacement",
                },
            ))
            edge_ops.append(EdgeOp(
                op=REPLACE_ENDPOINT_EDGE,
                old_source=target_service,
                old_target=old_node,
                new_target=new_node,
                edge_type="endpoint",
                attrs={
                    "old_ip": old_pod.get("ip", ""),
                    "new_ip": new_pod.get("ip", ""),
                    "reason": "pod_replacement",
                },
            ))
        evidence.append({"source": "topology", "signal": "node_drift", "details": _drift_details(drift)})

    elif _prefer_node_scale(hints, scale):
        change_type = "node_scale"
        confidence = 0.8
        causal_role = "effect_or_mixed"
        replica_count = max(int(scale.get("during_replicas", 0) or 0), 1)
        weight = 1.0 / replica_count
        for pod in scale.get("added_pods", []):
            node_id = pod_node_id(str(pod.get("name", "")))
            node_ops.append(NodeOp(
                op=ADD_NODE,
                node_id=node_id,
                service=target_service,
                attrs={
                    "service": target_service,
                    "pod_ip": pod.get("ip", ""),
                    "node": pod.get("node", ""),
                    "reason": "replica_increase",
                },
            ))
            edge_ops.append(EdgeOp(
                op=ADD_EDGE,
                source=target_service,
                target=node_id,
                edge_type="endpoint",
                weight=weight,
                attrs={"replica_count": replica_count},
            ))
        for pod in scale.get("during_pods", []):
            weight_ops.append(WeightOp(
                op=REWEIGHT_EDGE,
                source=target_service,
                target=pod_node_id(str(pod.get("name", ""))),
                weight=weight,
                attrs={"weight_source": "uniform_replica_share", "replica_count": replica_count},
            ))
        evidence.append({"source": "topology", "signal": "node_scale", "details": _scale_details(scale)})

    return GraphUpdatePlan(
        experiment_id=experiment_id,
        change_type=change_type,
        confidence=confidence,
        causal_role=causal_role,
        node_ops=node_ops,
        edge_ops=edge_ops,
        weight_ops=weight_ops,
        state_transfer={},
        evidence=evidence,
    )


def plan_to_dict(plan: GraphUpdatePlan) -> Dict[str, Any]:
    return asdict(plan)


def write_plan(plan: GraphUpdatePlan, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(plan_to_dict(plan), f, indent=2, sort_keys=True)


def _prefer_edge_removal(hints: set, signal: Dict[str, Any]) -> bool:
    return bool(signal.get("matched")) and _hint_matches(hints, ["t3", "edge_removal", "selector"])


def _prefer_edge_redirect(hints: set, signal: Dict[str, Any]) -> bool:
    return bool(signal.get("matched")) and _hint_matches(hints, ["t3", "edge_redirect", "selector"])


def _prefer_version_split(hints: set, signal: Dict[str, Any]) -> bool:
    return bool(signal.get("matched")) and _hint_matches(hints, ["t4", "version_split", "canary", "edge_split"])


def _prefer_node_drift(hints: set, signal: Dict[str, Any]) -> bool:
    return bool(signal.get("matched")) and _hint_matches(hints, ["t2", "drift", "pod_kill", "instance_drift"])


def _prefer_node_scale(hints: set, signal: Dict[str, Any]) -> bool:
    return bool(signal.get("matched")) and _hint_matches(hints, ["t1", "scale", "node_scale", "instance_count"])


def _hint_matches(hints: set, needles: List[str]) -> bool:
    if not hints:
        return True
    for hint in hints:
        for needle in needles:
            if needle in hint:
                return True
    return False


def _first_old_backend_node(redirect: Dict[str, Any]) -> str:
    old = redirect.get("old_service_pods", [])
    if old:
        return pod_node_id(str(old[0].get("name", "")))
    return f"pod:{redirect.get('service', 'unknown')}:previous"


def _scale_details(scale: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "service": scale.get("service", ""),
        "pre_replicas": scale.get("pre_replicas", 0),
        "during_replicas": scale.get("during_replicas", 0),
        "added_pods": [pod.get("name", "") for pod in scale.get("added_pods", [])],
    }


def _drift_details(drift: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "service": drift.get("service", ""),
        "pairs": [
            {"old": old.get("name", ""), "new": new.get("name", "")}
            for old, new in drift.get("pairs", [])
        ],
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a rule-based graph update plan.")
    parser.add_argument("--experiment-dir", default=None,
                        help="Experiment directory containing metadata/topology files.")
    parser.add_argument("--out", default=None,
                        help="Output GraphUpdatePlan JSON path.")
    parser.add_argument("--index", default=None,
                        help="Optional index artifact for batch plan generation.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for batch plan generation.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional max index rows for batch mode.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.index:
        if not args.out_dir:
            raise SystemExit("--out-dir is required with --index")
        outputs = write_plans_from_index(args.index, args.out_dir, limit=args.limit)
        print(f"wrote {len(outputs)} rule plans -> {args.out_dir}")
        return 0
    if not args.experiment_dir or not args.out:
        raise SystemExit("--experiment-dir and --out are required outside batch mode")
    plan = plan_experiment(args.experiment_dir)
    write_plan(plan, args.out)
    print(f"wrote rule plan for {plan.experiment_id} -> {args.out}")
    return 0


def write_plans_from_index(index_path: str, out_dir: str,
                           limit: Optional[int] = None) -> List[str]:
    """Write rule plans for indexed experiments without using ground truth decisions."""
    from dyrca_rca.dataset.materialize import read_index

    frame = read_index(index_path)
    if limit is not None:
        frame = frame.head(limit)
    outputs = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        error = row_dict.get("error_message")
        if isinstance(error, str) and error.strip():
            continue
        exp_dir = row_dict.get("data_path")
        if not exp_dir:
            continue
        experiment_id = str(row_dict.get("experiment_id") or Path(str(exp_dir)).name)
        out_path = Path(out_dir) / f"{experiment_id}.json"
        plan = plan_experiment(str(exp_dir))
        write_plan(plan, str(out_path))
        outputs.append(str(out_path))
    return outputs


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

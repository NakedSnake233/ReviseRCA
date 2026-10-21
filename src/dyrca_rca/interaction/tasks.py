"""Operator-facing task forms translated to existing validated feedback."""
from copy import deepcopy
from math import isfinite

from dyrca_rca.interaction.observation import epoch
from dyrca_rca.interaction.state import effective_state


def task_options(session):
    view, _ = effective_state(session)
    snapshots = [e for e in view["evidence"] if e["kind"] == "topology"]
    latest = max(snapshots, key=lambda e: epoch(e["data"]["timestamp"]))
    return {"snapshot_id": latest["id"], "services": latest["data"]["services"],
            "pods": latest["data"]["pods"]}


def task_feedback(session, result, task, service, values, event_id, actor, reason):
    options = task_options(session)
    matches = [s for s in options["services"] if s.get("name") == service]
    if len(matches) != 1:
        raise ValueError("请选择当前快照中唯一的服务")
    if task in ("T2", "T3"):
        operation, target = "patch_topology", options["snapshot_id"]
        payload = {"collection": "services", "entity": service, "field": "endpoint_ips",
                   "expected": matches[0].get("endpoint_ips"), "value": values["endpoint_ips"]}
    elif task == "T4":
        versions, weights = values["versions"], values["weights"]
        if (not isinstance(versions, list) or len(versions) < 2 or len(versions) != len(weights)
                or any(not isinstance(v, str) or not v or ":" in v for v in versions)
                or len(set(versions)) != len(versions)
                or any(isinstance(w, bool) or not isinstance(w, (int, float)) or not isfinite(w) or w < 0 or w > 1 for w in weights)
                or abs(sum(weights) - 1) > 1e-8):
            raise ValueError("版本需唯一且至少两个，权重必须在 [0,1] 且总和为 1")
        observed = {p.get("version") for p in options["pods"] if p.get("service") == service}
        if not set(versions) <= observed:
            raise ValueError("版本必须有当前 Pod 快照证据")
        payload = deepcopy(result["plan"])
        if any(n["id"] in {service + ":" + v for v in versions} for n in result["graph"]["nodes"]):
            raise ValueError("版本节点已存在，请使用单操作编辑调整现有计划")
        payload["node_ops"].append({"op": "SPLIT_NODE", "parent": service,
                                    "children": [service + ":" + v for v in versions],
                                    "attrs": {"service": service}, "evidence": [options["snapshot_id"]]})
        for version, weight in zip(versions, weights):
            payload["weight_ops"].append({"op": "REWEIGHT_EDGE", "source": service,
                                          "target": service + ":" + version, "weight": weight,
                                          "attrs": {"route_weight": weight}, "evidence": [options["snapshot_id"]]})
        operation, target = "replace_plan", "graph_plan"
        # Existing local edits are already included in result.plan. Replacing it
        # would apply them twice; ask the operator to retire these first.
        if effective_state(session)[0].get("plan_edits"):
            raise ValueError("请先撤回局部计划编辑，再提交版本拆分")
    else:
        raise ValueError("unknown task")
    view = session["observation"]
    return {"event_id": event_id, "event_kind": "feedback", "incident_id": session["incident_id"],
            "base_revision": session["state_revision"], "operation": operation, "target_id": target,
            "payload": payload, "actor": actor, "reason": reason,
            "scope": {"system_id": view["system_id"], "as_of": view["as_of"]}}

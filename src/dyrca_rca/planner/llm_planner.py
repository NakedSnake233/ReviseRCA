"""LLM Graph Update Planner interface with rule-based fallback."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dyrca_rca.graph.update_ops import GraphUpdatePlan
from dyrca_rca.planner.plan_validator import parse_json_object, validate_plan_dict, validate_plan_json
from dyrca_rca.planner.prompt_templates import build_graph_update_prompt
from dyrca_rca.planner.rule_planner import plan_experiment as rule_plan_experiment
from dyrca_rca.planner.signals import (
    KNOWN_EDGES,
    endpoint_backend_service,
    pod_node_id,
    pods_by_service,
    read_json,
    service_endpoint_ips,
    service_name_from_metadata,
    trace_mismatch_edges,
)


LLMCallable = Callable[[str], str]


class LLMUnavailable(RuntimeError):
    """Raised when no LLM backend is configured."""


class UnavailableLLMClient:
    """Default no-network client used by CLI unless a caller injects a client."""

    def __call__(self, prompt: str) -> str:
        raise LLMUnavailable("no LLM client configured")


class ResponsesAPIClient:
    """Minimal OpenAI Responses-compatible client.

    Configuration is intentionally environment-driven so API credentials do not
    need to be written into repository files.
    """

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 60.0, max_output_tokens: int = 2000,
                 session_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.session_id = session_id or str(uuid.uuid4())

    @classmethod
    def from_env(cls) -> "ResponsesAPIClient":
        api_key = os.environ.get("DYRCA_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = (
            os.environ.get("DYRCA_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )
        model = os.environ.get("DYRCA_LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
        timeout = _env_float("DYRCA_LLM_TIMEOUT", 60.0)
        default_tokens = 8000 if str(model).startswith("gpt-5") else 3000
        max_output_tokens = int(_env_float("DYRCA_LLM_MAX_OUTPUT_TOKENS", default_tokens))
        if not api_key or not base_url:
            raise LLMUnavailable("DYRCA_LLM_API_KEY and DYRCA_LLM_BASE_URL are required")
        return cls(base_url=base_url, api_key=api_key, model=model,
                   timeout=timeout, max_output_tokens=max_output_tokens)

    def __call__(self, prompt: str) -> str:
        endpoint = self.base_url if self.base_url.endswith("/responses") else f"{self.base_url}/responses"
        payload = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
        }
        if str(self.model).startswith("gpt-5"):
            payload["reasoning"] = {"effort": os.environ.get("DYRCA_LLM_REASONING_EFFORT", "low")}
            payload["text"] = {
                "format": {"type": "text"},
                "verbosity": os.environ.get("DYRCA_LLM_VERBOSITY", "low"),
            }
        else:
            payload["temperature"] = 0
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if endpoint.startswith("https://opencode.ai/zen/go/"):
            # Go requires a stable session identifier for each conversation.
            headers["x-opencode-session"] = self.session_id
            headers["User-Agent"] = "dyrca-research-agent/0.1"
        try:
            import httpx
        except ImportError:
            # Some supported experiment environments have requests but no
            # httpx. Both transports send the same Responses API payload.
            import requests
            response = requests.post(endpoint, headers=headers, json=payload,
                                     timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        else:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        text = _extract_response_text(data)
        if not text:
            raise LLMUnavailable("LLM response did not contain text")
        return text


def build_evidence_json(experiment_dir: str) -> Dict[str, Any]:
    """Build compact evidence for an experiment without loading heavy raw data."""
    exp_dir = Path(experiment_dir)
    metadata = read_json(exp_dir / "metadata.json", required=True)
    topology_diff = read_json(exp_dir / "topology_diff.json", required=False)
    pre = read_json(exp_dir / "topology" / "pre.json", required=True)
    during = read_json(exp_dir / "topology" / "during.json", required=True)
    experiment_id = str(metadata.get("experiment_id") or exp_dir.name)
    target_service = service_name_from_metadata(metadata)

    return {
        "experiment_id": experiment_id,
        "scenario_name": metadata.get("scenario_name", ""),
        "scenario_id": metadata.get("scenario_id", ""),
        "target_service": target_service,
        "topology_diff_summary": _topology_diff_summary(topology_diff),
        "dependency_edge_summary": _dependency_edge_summary(pre, target_service),
        "endpoint_diff_summary": _endpoint_diff_summary(pre, during, target_service),
        "pod_diff_summary": _pod_diff_summary(pre, during, target_service),
        "deployment_diff_summary": _deployment_diff_summary(pre, during, target_service),
        "trace_route_summary": _trace_route_summary(exp_dir / "traces" / "trace_summary.csv"),
        "metric_summary": _metric_summary(exp_dir / "metrics"),
        "metadata_params": metadata.get("fault_params", {}) if isinstance(metadata.get("fault_params"), dict) else {},
    }


def plan_experiment_with_llm(experiment_dir: str,
                             llm_client: Optional[LLMCallable] = None,
                             fallback: str = "rule",
                             cache_dir: str = "artifacts/llm_plans",
                             use_cache: bool = True) -> GraphUpdatePlan:
    """Plan with an LLM response when valid, otherwise fall back deterministically."""
    evidence = build_evidence_json(experiment_dir)
    experiment_id = str(evidence["experiment_id"])
    cache_path = Path(cache_dir) / f"{experiment_id}.json"

    if use_cache and cache_path.exists():
        cached_text = cache_path.read_text(encoding="utf-8")
        cached = validate_plan_json(cached_text)
        if cached.valid and cached.plan is not None:
            return cached.plan

    prompt = build_graph_update_prompt(evidence)
    client = llm_client or default_llm_client()
    try:
        response_text = client(prompt)
        validation = validate_llm_plan_response(response_text, evidence)
        if validation.valid and validation.plan is not None:
            plan = validation.plan
            plan.evidence.append({
                "source": "llm_planner",
                "fallback_used": False,
                "evidence": evidence,
            })
            _write_cache(cache_path, plan)
            return plan
        fallback_reason = "; ".join(validation.errors)
    except Exception as exc:
        fallback_reason = str(exc)

    if fallback != "rule":
        raise LLMUnavailable(f"LLM planner failed and fallback is disabled: {fallback_reason}")

    plan = rule_plan_experiment(experiment_dir)
    plan.evidence.append({
        "source": "llm_planner",
        "fallback_used": True,
        "fallback_reason": fallback_reason,
        "evidence": evidence,
    })
    _write_cache(cache_path, plan)
    return plan


def default_llm_client() -> LLMCallable:
    """Return an environment-configured LLM client or an unavailable stub."""
    try:
        return ResponsesAPIClient.from_env()
    except LLMUnavailable:
        return UnavailableLLMClient()


def _write_cache(path: Path, plan: GraphUpdatePlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(plan), f, indent=2, sort_keys=True)


def write_plan(plan: GraphUpdatePlan, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(plan), f, indent=2, sort_keys=True)


def write_llm_plans_from_index(index_path: str, out_dir: str,
                               fallback: str = "rule",
                               cache_dir: Optional[str] = None,
                               limit: Optional[int] = None,
                               use_cache: bool = True,
                               concurrency: int = 1) -> List[str]:
    """Generate LLM-guided plans for indexed experiments."""
    from dyrca_rca.dataset.materialize import read_index

    frame = read_index(index_path)
    if limit is not None:
        frame = frame.head(limit)
    jobs = []
    cache_root = cache_dir or out_dir
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        error = row_dict.get("error_message")
        if isinstance(error, str) and error.strip():
            continue
        exp_dir = row_dict.get("data_path")
        if not exp_dir:
            continue
        experiment_id = str(row_dict.get("experiment_id") or Path(str(exp_dir)).name)
        jobs.append((experiment_id, str(exp_dir), str(Path(out_dir) / f"{experiment_id}.json")))

    def run_job(job):
        _, exp_dir, out_path = job
        client = default_llm_client()
        plan = plan_experiment_with_llm(
            exp_dir,
            llm_client=client,
            fallback=fallback,
            cache_dir=cache_root,
            use_cache=use_cache,
        )
        write_plan(plan, out_path)
        return out_path

    if concurrency <= 1:
        return [run_job(job) for job in jobs]

    outputs = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            outputs.append(future.result())
            if len(outputs) % 25 == 0:
                print(f"generated {len(outputs)}/{len(jobs)} LLM graph plans", flush=True)
    return sorted(outputs)


def validate_llm_plan_response(response_text: str, evidence: Dict[str, Any]):
    """Parse, normalize common LLM field variants, then validate."""
    try:
        data = parse_json_object(response_text)
    except Exception:
        return validate_plan_json(response_text)
    return validate_plan_dict(normalize_llm_plan_dict(data, evidence))


def normalize_llm_plan_dict(data: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common valid-intent LLM variants into GraphUpdatePlan schema."""
    out = dict(data)
    target_service = str(evidence.get("target_service", "") or "")
    out.setdefault("evidence", [])
    if not isinstance(out.get("evidence"), list):
        out["evidence"] = [out.get("evidence")]
    out.setdefault("state_transfer", {})

    node_ops = []
    for raw in _as_list(out.get("node_ops")):
        if not isinstance(raw, dict):
            continue
        op = dict(raw)
        if not op.get("node_id"):
            for alias in ("node", "name", "id"):
                if op.get(alias):
                    op["node_id"] = op[alias]
                    break
        if op.get("node_id"):
            op["node_id"] = _normalize_node_id(op["node_id"], infer_pod=True)
        for key in ("old_node", "new_node", "parent"):
            if op.get(key):
                op[key] = _normalize_node_id(op[key], infer_pod=key != "parent")
        if isinstance(op.get("children"), list):
            op["children"] = [_normalize_node_id(child) for child in op["children"]]
        if str(op.get("op", "")).upper() in {"", "KEEP_EDGE"}:
            continue
        node_ops.append(_repair_node_op(op, evidence))

    edge_ops = []
    for raw in _as_list(out.get("edge_ops")):
        if not isinstance(raw, dict):
            continue
        op = dict(raw)
        _copy_nested_edge_fields(op)
        op_name = str(op.get("op", "")).upper()
        if op_name in {"KEEP_EDGE", "KEEP_NODE", ""}:
            continue
        if op_name == "ADD_BLACKHOLE_NODE":
            node_ops.append({
                "op": "ADD_BLACKHOLE_NODE",
                "service": op.get("service") or op.get("source") or target_service,
                "attrs": op.get("attrs", {}),
            })
            continue
        for key in ("source", "old_source", "new_source"):
            if op.get(key):
                op[key] = _normalize_node_id(op[key])
        for key in ("target", "old_target", "new_target"):
            if op.get(key):
                op[key] = _normalize_node_id(op[key], infer_pod=_looks_like_pod_name(op[key]))
        edge_ops.append(_repair_edge_op(op, evidence))

    weight_ops = []
    for raw in _as_list(out.get("weight_ops")):
        if not isinstance(raw, dict):
            continue
        op = dict(raw)
        attrs = dict(op.get("attrs", {})) if isinstance(op.get("attrs"), dict) else {}
        for key in ("dependency_weight", "anomaly_weight", "topology_confidence", "change_weight", "final_weight"):
            if key in op:
                attrs.setdefault(key, op.pop(key))
        if "weight" not in op and "final_weight" in attrs:
            op["weight"] = attrs["final_weight"]
        if "source" not in op or "target" not in op:
            _copy_nested_edge_fields(op)
        if "source" not in op or "target" not in op:
            source, target = _source_target_from_edge_id(str(op.get("edge_id", "")), target_service)
            op.setdefault("source", source)
            op.setdefault("target", target)
        if op.get("source"):
            op["source"] = _normalize_node_id(op["source"])
        if op.get("target"):
            op["target"] = _normalize_node_id(op["target"], infer_pod=_looks_like_pod_name(op["target"]))
        op["attrs"] = attrs
        weight_ops.append(_repair_weight_op(op, evidence, len(weight_ops)))

    out["node_ops"] = node_ops
    out["edge_ops"] = edge_ops
    out["weight_ops"] = weight_ops
    return _ensure_minimal_ops(out, evidence)


def _repair_node_op(op: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    op = dict(op)
    op_name = str(op.get("op", "")).upper()
    target_service = str(evidence.get("target_service", "") or "")
    if op_name == "ADD_BLACKHOLE_NODE" and not op.get("service"):
        op["service"] = target_service
    if op_name == "ALIGN_NODE":
        old_node, new_node = _drift_pair_from_evidence(evidence)
        op.setdefault("old_node", old_node)
        op.setdefault("new_node", new_node)
    if op_name == "SPLIT_NODE":
        op.setdefault("parent", target_service)
        if not op.get("children") and target_service:
            op["children"] = _version_children_from_evidence(evidence)
    return op


def _repair_edge_op(op: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    op = dict(op)
    op_name = str(op.get("op", "")).upper()
    target_service = str(evidence.get("target_service", "") or "")
    if op_name in {"ADD_EDGE", "REMOVE_EDGE", "ADD_TRACE_MISMATCH_EDGE"}:
        if op_name == "ADD_TRACE_MISMATCH_EDGE":
            mismatch = _first_trace_mismatch(evidence)
            op.setdefault("source", mismatch.get("source", ""))
            op.setdefault("target", mismatch.get("observed", ""))
        if not op.get("source") and target_service:
            op["source"] = target_service
        if not op.get("target"):
            op["target"] = _first_relevant_pod_node(evidence, prefer_added=True)
    elif op_name == "REDIRECT_EDGE":
        old_target, new_target = _redirect_targets_from_evidence(evidence)
        op.setdefault("old_source", target_service)
        op.setdefault("old_target", old_target)
        op.setdefault("new_target", new_target)
    elif op_name == "REPLACE_ENDPOINT_EDGE":
        old_node, new_node = _drift_pair_from_evidence(evidence)
        op.setdefault("old_source", target_service)
        op.setdefault("old_target", old_node)
        op.setdefault("new_target", new_node)
    elif op_name == "SPLIT_EDGE":
        op.setdefault("source", "frontend")
        op.setdefault("target", target_service)
        if not op.get("new_target"):
            children = _version_children_from_evidence(evidence)
            op["new_target"] = children[-1] if children else f"{target_service}:v2"
    return op


def _repair_weight_op(op: Dict[str, Any], evidence: Dict[str, Any], idx: int) -> Dict[str, Any]:
    op = dict(op)
    target_service = str(evidence.get("target_service", "") or "")
    attrs = dict(op.get("attrs", {})) if isinstance(op.get("attrs"), dict) else {}
    op.setdefault("attrs", attrs)
    if "weight" not in op:
        for key in ("final_weight", "dependency_weight", "route_weight", "weight"):
            if key in attrs:
                op["weight"] = attrs[key]
                break
    if not op.get("source"):
        op["source"] = target_service or "frontend"
    if not op.get("target"):
        targets = _weight_targets_from_evidence(evidence)
        if idx < len(targets):
            op["target"] = targets[idx]
        elif targets:
            op["target"] = targets[-1]
    return op


def _ensure_minimal_ops(out: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Add missing intent-preserving fields/ops from evidence summaries."""
    target_service = str(evidence.get("target_service", "") or "")
    scenario = str(evidence.get("scenario_id", "") or "").upper()
    scenario_name = str(evidence.get("scenario_name", "") or "").lower()
    change_type = str(out.get("change_type", "") or evidence.get("topology_diff_summary", {}).get("topology_change_type", "")).lower()

    node_ops = list(out.get("node_ops", []))
    edge_ops = list(out.get("edge_ops", []))
    weight_ops = list(out.get("weight_ops", []))

    if (scenario == "T1" or "scale" in scenario_name or "scale" in change_type) and target_service:
        existing_adds = {op.get("node_id") for op in node_ops if str(op.get("op", "")).upper() == "ADD_NODE"}
        added = _added_pod_nodes(evidence)
        for node_id in added:
            if node_id not in existing_adds:
                node_ops.append({"op": "ADD_NODE", "node_id": node_id})
        existing_edges = {(op.get("source"), op.get("target")) for op in edge_ops if str(op.get("op", "")).upper() == "ADD_EDGE"}
        weight = 1.0 / max(len(_all_target_pod_nodes(evidence)), 1)
        for node_id in added:
            if (target_service, node_id) not in existing_edges:
                edge_ops.append({"op": "ADD_EDGE", "source": target_service, "target": node_id, "edge_type": "endpoint", "weight": weight})
        existing_weights = {(op.get("source"), op.get("target")) for op in weight_ops}
        for node_id in _all_target_pod_nodes(evidence):
            if (target_service, node_id) not in existing_weights:
                weight_ops.append({"op": "REWEIGHT_EDGE", "source": target_service, "target": node_id, "weight": weight})

    if (scenario == "T2" or "drift" in scenario_name or "drift" in change_type) and target_service:
        old_node, new_node = _drift_pair_from_evidence(evidence)
        if old_node and new_node:
            node_ops = [
                op for op in node_ops
                if not (
                    isinstance(op, dict)
                    and str(op.get("op", "")).upper() == "ADD_NODE"
                    and op.get("node_id") in {old_node, new_node}
                )
            ]
            if not any(str(op.get("op", "")).upper() == "ALIGN_NODE" for op in node_ops):
                node_ops.append({"op": "ALIGN_NODE", "old_node": old_node, "new_node": new_node})
            if not any(str(op.get("op", "")).upper() == "REPLACE_ENDPOINT_EDGE" for op in edge_ops):
                edge_ops.append({"op": "REPLACE_ENDPOINT_EDGE", "old_source": target_service, "old_target": old_node, "new_target": new_node})

    if (scenario == "T3" or "selector" in scenario_name or "redirect" in change_type) and target_service:
        node_ops = [
            op for op in node_ops
            if not (isinstance(op, dict) and str(op.get("op", "")).upper() == "ADD_NODE")
        ]
        if "empty" in change_type or _has_empty_endpoint(evidence):
            if not any(str(op.get("op", "")).upper() == "ADD_BLACKHOLE_NODE" for op in node_ops):
                node_ops.append({"op": "ADD_BLACKHOLE_NODE", "service": target_service})
        elif "redirect" in change_type or _redirect_targets_from_evidence(evidence)[1]:
            old_target, new_target = _redirect_targets_from_evidence(evidence)
            if old_target and new_target and not any(str(op.get("op", "")).upper() == "REDIRECT_EDGE" for op in edge_ops):
                edge_ops.append({"op": "REDIRECT_EDGE", "old_source": target_service, "old_target": old_target, "new_target": new_target})
        mismatch = _first_trace_mismatch(evidence)
        if mismatch and not any(str(op.get("op", "")).upper() == "ADD_TRACE_MISMATCH_EDGE" for op in edge_ops):
            edge_ops.append({"op": "ADD_TRACE_MISMATCH_EDGE", "source": mismatch.get("source", ""), "target": mismatch.get("observed", ""), "edge_type": "trace_mismatch"})

    if (scenario == "T4" or "canary" in scenario_name or "split" in change_type) and target_service:
        children = _version_children_from_evidence(evidence)
        child_set = set(children)
        added_pods = set(_added_pod_nodes(evidence))
        known_nodes = _known_node_ids_from_evidence(evidence) | child_set | added_pods
        upstreams = _t4_upstream_services_from_evidence(evidence)
        primary_upstream = upstreams[0] if upstreams else "frontend"
        for op in node_ops:
            if not isinstance(op, dict):
                continue
            if op.get("node_id"):
                op["node_id"] = _canonical_t4_pod_id(
                    _canonical_t4_node_id(op["node_id"], target_service),
                    target_service,
                    added_pods,
                )
            for key in ("old_node", "new_node", "parent"):
                if op.get(key):
                    op[key] = _canonical_t4_pod_id(
                        _canonical_t4_node_id(op[key], target_service),
                        target_service,
                        added_pods,
                    )
            if str(op.get("op", "")).upper() == "SPLIT_NODE":
                op["parent"] = target_service
                op["children"] = children
        node_ops = [
            op for op in node_ops
            if not (
                isinstance(op, dict)
                and str(op.get("op", "")).upper() == "ADD_NODE"
                and op.get("node_id")
                and op.get("node_id") not in added_pods
            )
        ]
        existing_adds = {
            op.get("node_id") for op in node_ops
            if isinstance(op, dict) and str(op.get("op", "")).upper() == "ADD_NODE"
        }
        for node_id in added_pods:
            if node_id not in existing_adds:
                node_ops.append({"op": "ADD_NODE", "node_id": node_id})
        for op in edge_ops:
            if not isinstance(op, dict):
                continue
            for key in ("source", "target", "old_source", "old_target", "new_source", "new_target"):
                if op.get(key):
                    op[key] = _canonical_t4_pod_id(
                        _canonical_t4_node_id(op[key], target_service),
                        target_service,
                        added_pods,
                    )
        for op in weight_ops:
            if not isinstance(op, dict):
                continue
            for key in ("source", "target"):
                if op.get(key):
                    op[key] = _canonical_t4_pod_id(
                        _canonical_t4_node_id(op[key], target_service),
                        target_service,
                        added_pods,
                    )
        if upstreams:
            edge_ops = [
                op for op in edge_ops
                if not (
                    isinstance(op, dict)
                    and str(op.get("op", "")).upper() == "SPLIT_EDGE"
                    and op.get("source")
                    and op.get("target") == target_service
                    and op.get("source") not in upstreams
                )
            ]
            weight_ops = [
                op for op in weight_ops
                if not (
                    isinstance(op, dict)
                    and str(op.get("op", "")).upper() == "REWEIGHT_EDGE"
                    and op.get("target") in child_set
                    and op.get("source")
                    and op.get("source") not in upstreams
                )
            ]
        edge_ops = [
            op for op in edge_ops
            if not (
                isinstance(op, dict)
                and str(op.get("op", "")).upper() == "SPLIT_EDGE"
                and op.get("source") == target_service
                and op.get("target") == target_service
            )
        ]
        edge_ops = [
            op for op in edge_ops
            if not (
                isinstance(op, dict)
                and str(op.get("op", "")).upper() == "REMOVE_EDGE"
                and op.get("target") in added_pods
            )
        ]
        node_ops = [
            op for op in node_ops
            if not (
                isinstance(op, dict)
                and str(op.get("op", "")).upper() == "KEEP_NODE"
                and op.get("node_id")
                and op.get("node_id") not in known_nodes
            )
        ]
        edge_ops = [
            op for op in edge_ops
            if not (
                isinstance(op, dict)
                and str(op.get("op", "")).upper() == "SPLIT_EDGE"
                and op.get("new_target")
                and str(op.get("new_target")).startswith(f"{target_service}:")
                and op.get("new_target") not in child_set
            )
        ]
        weight_ops = [
            op for op in weight_ops
            if not (
                isinstance(op, dict)
                and str(op.get("target", "")).startswith(f"{target_service}:")
                and op.get("target") not in child_set
            )
        ]
        if children and not any(str(op.get("op", "")).upper() == "SPLIT_NODE" for op in node_ops):
            node_ops.append({"op": "SPLIT_NODE", "parent": target_service, "children": children})
        existing_splits = {
            (op.get("source"), op.get("new_target")) for op in edge_ops
            if isinstance(op, dict) and str(op.get("op", "")).upper() == "SPLIT_EDGE"
        }
        if children:
            for upstream in (upstreams or [primary_upstream]):
                for child in children:
                    if (upstream, child) not in existing_splits:
                        edge_ops.append({"op": "SPLIT_EDGE", "source": upstream, "target": target_service, "new_target": child})
        route_weights = _route_weights_from_evidence(evidence)
        existing_weights = {(op.get("source"), op.get("target")) for op in weight_ops}
        for upstream in (upstreams or [primary_upstream]):
            for child in children:
                version = child.rsplit(":", 1)[-1]
                if (upstream, child) not in existing_weights and version in route_weights:
                    weight_ops.append({"op": "REWEIGHT_EDGE", "source": upstream, "target": child, "weight": route_weights[version]})
        existing_edges = {
            (op.get("source"), op.get("target")) for op in edge_ops
            if isinstance(op, dict) and str(op.get("op", "")).upper() == "ADD_EDGE"
        }
        existing_edges.update(
            (op.get("source"), op.get("new_target")) for op in edge_ops
            if isinstance(op, dict) and str(op.get("op", "")).upper() == "SPLIT_EDGE"
        )
        for op in weight_ops:
            if not isinstance(op, dict) or str(op.get("op", "")).upper() != "REWEIGHT_EDGE":
                continue
            source = op.get("source")
            target = op.get("target")
            if not source or not target:
                continue
            if known_nodes and (source not in known_nodes or target not in known_nodes):
                continue
            if (source, target) not in existing_edges:
                edge_ops.append({
                    "op": "ADD_EDGE",
                    "source": source,
                    "target": target,
                    "edge_type": "dependency",
                    "weight": op.get("weight"),
                    "attrs": op.get("attrs", {}),
                })
                existing_edges.add((source, target))

    out["node_ops"] = [_repair_node_op(op, evidence) for op in node_ops if isinstance(op, dict)]
    out["edge_ops"] = [_repair_edge_op(op, evidence) for op in edge_ops if isinstance(op, dict)]
    out["weight_ops"] = [_repair_weight_op(op, evidence, idx) for idx, op in enumerate(weight_ops) if isinstance(op, dict)]
    return out


def _topology_diff_summary(topology_diff: Dict[str, Any]) -> Dict[str, Any]:
    annotation = topology_diff.get("annotation", {}) if isinstance(topology_diff.get("annotation"), dict) else {}
    comparisons = topology_diff.get("comparisons", {}) if isinstance(topology_diff.get("comparisons"), dict) else {}
    return {
        "schema": topology_diff.get("schema", ""),
        "verification_result": annotation.get("verification_result", ""),
        "topology_change_type": annotation.get("topology_change_type", ""),
        "detected_types": annotation.get("detected_types", []),
        "evidence": annotation.get("evidence", [])[:5] if isinstance(annotation.get("evidence", []), list) else [],
        "comparisons": sorted(comparisons.keys()),
    }


def _dependency_edge_summary(snapshot: Dict[str, Any], target_service: str) -> Dict[str, Any]:
    upstreams = []
    downstreams = []
    services = {
        str(dep.get("name", ""))
        for dep in snapshot.get("deployments", []) or []
        if isinstance(dep, dict) and dep.get("name")
    }
    services.update(
        str(svc.get("name", ""))
        for svc in snapshot.get("services", []) or []
        if isinstance(svc, dict) and svc.get("name")
    )
    for source, target in KNOWN_EDGES:
        if source not in services or target not in services:
            continue
        if target == target_service and source != target_service:
            upstreams.append({"source": source, "target": target, "edge_type": "dependency"})
        if source == target_service and target != target_service:
            downstreams.append({"source": source, "target": target, "edge_type": "dependency"})
    return {
        "target_service": target_service,
        "upstream_dependencies": upstreams[:20],
        "downstream_dependencies": downstreams[:20],
    }


def _endpoint_diff_summary(pre: Dict[str, Any], during: Dict[str, Any],
                           target_service: str) -> Dict[str, Any]:
    services = sorted({
        str(svc.get("name", ""))
        for snap in (pre, during)
        for svc in (snap.get("services", []) or [])
        if isinstance(svc, dict) and svc.get("name")
    })
    changed = []
    for service in services:
        before = service_endpoint_ips(pre, service)
        after = service_endpoint_ips(during, service)
        if before != after:
            changed.append({
                "service": service,
                "before": before,
                "after": after,
                "before_backend_services": [
                    {"service": owner, "pod": pod.get("name", ""), "ip": pod.get("ip", "")}
                    for owner, pod in endpoint_backend_service(pre, service)
                ],
                "after_backend_services": [
                    {"service": owner, "pod": pod.get("name", ""), "ip": pod.get("ip", "")}
                    for owner, pod in endpoint_backend_service(during, service)
                ],
            })
    return {
        "target_service": target_service,
        "target_before": service_endpoint_ips(pre, target_service),
        "target_after": service_endpoint_ips(during, target_service),
        "changed_services": changed[:20],
    }


def _pod_diff_summary(pre: Dict[str, Any], during: Dict[str, Any],
                      target_service: str) -> Dict[str, Any]:
    pre_by_service = pods_by_service(pre)
    during_by_service = pods_by_service(during)
    services = sorted(set(pre_by_service) | set(during_by_service))
    changed = []
    for service in services:
        before = {str(p.get("name", "")): p for p in pre_by_service.get(service, [])}
        after = {str(p.get("name", "")): p for p in during_by_service.get(service, [])}
        added = sorted(after.keys() - before.keys())
        removed = sorted(before.keys() - after.keys())
        if added or removed or len(before) != len(after):
            changed.append({
                "service": service,
                "added": added,
                "removed": removed,
                "count_before": len(before),
                "count_after": len(after),
            })
    return {
        "target_service": target_service,
        "changed_services": changed[:20],
    }


def _deployment_diff_summary(pre: Dict[str, Any], during: Dict[str, Any],
                             target_service: str) -> Dict[str, Any]:
    before = {str(d.get("name", "")): d for d in pre.get("deployments", []) or [] if isinstance(d, dict)}
    after = {str(d.get("name", "")): d for d in during.get("deployments", []) or [] if isinstance(d, dict)}
    changed = []
    for name in sorted(set(before) | set(after)):
        b = before.get(name, {})
        a = after.get(name, {})
        if b.get("replicas") != a.get("replicas") or b.get("version") != a.get("version"):
            changed.append({
                "deployment": name,
                "replicas_before": b.get("replicas", 0),
                "replicas_after": a.get("replicas", 0),
                "version_before": b.get("version", ""),
                "version_after": a.get("version", ""),
            })
    return {
        "target_service": target_service,
        "changed_deployments": changed[:20],
    }


def _trace_route_summary(path: Path) -> Dict[str, Any]:
    mismatches = trace_mismatch_edges(path)
    summary = {
        "exists": path.exists(),
        "declared_observed_mismatches": mismatches[:20],
        "rows_scanned": 0,
        "services": [],
    }
    if not path.exists():
        return summary
    services = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= 200:
                break
            summary["rows_scanned"] += 1
            if row.get("service"):
                services.add(str(row["service"]))
            if row.get("destination_service"):
                services.add(str(row["destination_service"]))
    summary["services"] = sorted(services)[:30]
    return summary


def _metric_summary(metrics_dir: Path) -> Dict[str, Any]:
    if not metrics_dir.is_dir():
        return {"exists": False, "files": []}
    files = sorted(path.name for path in metrics_dir.iterdir() if path.is_file())
    return {
        "exists": True,
        "files": files[:30],
        "metric_names": sorted(path.rsplit(".", 1)[0] for path in files if path.endswith(".csv"))[:30],
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an LLM-guided graph update plan.")
    parser.add_argument("--experiment-dir", default=None,
                        help="Experiment directory containing metadata/topology files.")
    parser.add_argument("--out", default=None,
                        help="Output GraphUpdatePlan JSON path.")
    parser.add_argument("--index", default=None,
                        help="Optional index artifact for batch LLM plan generation.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for batch LLM plan generation.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional max index rows for batch mode.")
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("DYRCA_LLM_CONCURRENCY", "1")),
                        help="Concurrent LLM API calls for batch mode.")
    parser.add_argument("--fallback", default="rule", choices=["rule", "none"],
                        help="Fallback strategy when LLM output is unavailable or invalid.")
    parser.add_argument("--cache-dir", default="artifacts/llm_plans",
                        help="Directory for cached LLM plans.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached LLM plans.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    fallback = "rule" if args.fallback == "rule" else "none"
    if args.index:
        if not args.out_dir:
            raise SystemExit("--out-dir is required with --index")
        outputs = write_llm_plans_from_index(
            index_path=args.index,
            out_dir=args.out_dir,
            fallback=fallback,
            cache_dir=args.cache_dir,
            limit=args.limit,
            use_cache=not args.no_cache,
            concurrency=args.concurrency,
        )
        print(f"wrote {len(outputs)} LLM graph plans -> {args.out_dir}")
        return 0
    if not args.experiment_dir or not args.out:
        raise SystemExit("--experiment-dir and --out are required outside batch mode")
    plan = plan_experiment_with_llm(
        args.experiment_dir,
        llm_client=None,
        fallback=fallback,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )
    write_plan(plan, args.out)
    print(f"wrote LLM graph plan for {plan.experiment_id} -> {args.out}")
    return 0


def _extract_response_text(data: Dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"])
    output = data.get("output")
    if isinstance(output, list):
        parts = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("text"):
                        parts.append(str(block["text"]))
                    elif block.get("type") in {"output_text", "text"} and block.get("content"):
                        parts.append(str(block["content"]))
            elif isinstance(content, str):
                parts.append(content)
        if parts:
            return "\n".join(parts)
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message", {})
            if isinstance(message, dict) and message.get("content"):
                return str(message["content"])
            if first.get("text"):
                return str(first["text"])
    return ""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_node_id(value: Any, infer_pod: bool = False) -> str:
    text = str(value)
    if text.startswith("pod_"):
        return "pod:" + text[4:]
    if infer_pod and _looks_like_pod_name(text):
        return "pod:" + text
    return text


def _source_target_from_edge_id(edge_id: str, default_source: str) -> tuple:
    if "->" not in edge_id:
        return default_source, ""
    left, right = edge_id.split("->", 1)
    source = default_source if left in {"endpoint", "service", ""} else left
    return source, _normalize_node_id(right, infer_pod=_looks_like_pod_name(right))


def _copy_nested_edge_fields(op: Dict[str, Any]) -> None:
    edge = op.get("edge")
    if not isinstance(edge, dict):
        return
    if not op.get("source"):
        op["source"] = edge.get("source") or edge.get("src")
    if not op.get("target"):
        op["target"] = edge.get("target") or edge.get("dst")
    if not op.get("old_source"):
        op["old_source"] = edge.get("old_source") or edge.get("old_src")
    if not op.get("old_target"):
        op["old_target"] = edge.get("old_target") or edge.get("old_dst")
    if not op.get("new_source"):
        op["new_source"] = edge.get("new_source") or edge.get("new_src")
    if not op.get("new_target"):
        op["new_target"] = edge.get("new_target") or edge.get("new_dst")


def _looks_like_pod_name(value: Any) -> bool:
    text = str(value)
    if text.startswith("pod:") or text.startswith("blackhole:") or ":" in text:
        return False
    parts = text.split("-")
    return len(parts) >= 3 and any(char.isdigit() for char in text)


def _target_pod_change(evidence: Dict[str, Any]) -> Dict[str, Any]:
    target = str(evidence.get("target_service", "") or "")
    for item in evidence.get("pod_diff_summary", {}).get("changed_services", []) or []:
        if isinstance(item, dict) and item.get("service") == target:
            return item
    return {}


def _added_pod_nodes(evidence: Dict[str, Any]) -> List[str]:
    item = _target_pod_change(evidence)
    return [_normalize_node_id(name, infer_pod=True) for name in item.get("added", []) or []]


def _removed_pod_nodes(evidence: Dict[str, Any]) -> List[str]:
    item = _target_pod_change(evidence)
    return [_normalize_node_id(name, infer_pod=True) for name in item.get("removed", []) or []]


def _all_target_pod_nodes(evidence: Dict[str, Any]) -> List[str]:
    target = str(evidence.get("target_service", "") or "")
    pods = []
    for item in evidence.get("endpoint_diff_summary", {}).get("changed_services", []) or []:
        if not isinstance(item, dict) or item.get("service") != target:
            continue
        for backend in item.get("after_backend_services", []) or []:
            if isinstance(backend, dict) and backend.get("pod"):
                pods.append(_normalize_node_id(backend["pod"], infer_pod=True))
    for node_id in _added_pod_nodes(evidence):
        pods.append(node_id)
    return sorted(dict.fromkeys(pods))


def _first_relevant_pod_node(evidence: Dict[str, Any], prefer_added: bool = False) -> str:
    if prefer_added:
        added = _added_pod_nodes(evidence)
        if added:
            return added[0]
    all_nodes = _all_target_pod_nodes(evidence)
    return all_nodes[0] if all_nodes else ""


def _drift_pair_from_evidence(evidence: Dict[str, Any]) -> tuple:
    removed = _removed_pod_nodes(evidence)
    added = _added_pod_nodes(evidence)
    return (removed[0] if removed else "", added[0] if added else "")


def _endpoint_change(evidence: Dict[str, Any]) -> Dict[str, Any]:
    target = str(evidence.get("target_service", "") or "")
    for item in evidence.get("endpoint_diff_summary", {}).get("changed_services", []) or []:
        if isinstance(item, dict) and item.get("service") == target:
            return item
    return {}


def _redirect_targets_from_evidence(evidence: Dict[str, Any]) -> tuple:
    target = str(evidence.get("target_service", "") or "")
    change = _endpoint_change(evidence)
    old_target = ""
    new_target = ""
    before = change.get("before_backend_services", []) or []
    for backend in before:
        if isinstance(backend, dict) and backend.get("pod"):
            old_target = _normalize_node_id(backend["pod"], infer_pod=True)
            break
    after = change.get("after_backend_services", []) or []
    for backend in after:
        if not isinstance(backend, dict) or not backend.get("pod"):
            continue
        owner = str(backend.get("service", "") or "")
        if owner and owner != target:
            new_target = _normalize_node_id(backend["pod"], infer_pod=True)
            break
    if not old_target:
        old_target = _first_relevant_pod_node(evidence)
    return old_target, new_target


def _has_empty_endpoint(evidence: Dict[str, Any]) -> bool:
    change = _endpoint_change(evidence)
    if not change:
        return False
    after = change.get("after", [])
    backends = change.get("after_backend_services", [])
    return len(after or []) == 0 or len(backends or []) == 0


def _first_trace_mismatch(evidence: Dict[str, Any]) -> Dict[str, Any]:
    mismatches = evidence.get("trace_route_summary", {}).get("declared_observed_mismatches", []) or []
    for item in mismatches:
        if isinstance(item, dict):
            return item
    return {}


def _route_weights_from_evidence(evidence: Dict[str, Any]) -> Dict[str, float]:
    params = evidence.get("metadata_params", {})
    if not isinstance(params, dict):
        return {}
    weights = {}
    if "v1_weight" in params:
        weights["v1"] = _ratio(params.get("v1_weight"))
    if "v2_weight" in params:
        weights["v2"] = _ratio(params.get("v2_weight"))
    if "v2" in weights and "v1" not in weights:
        weights["v1"] = max(0.0, 1.0 - weights["v2"])
    if "v1" in weights and "v2" not in weights:
        weights["v2"] = max(0.0, 1.0 - weights["v1"])
    return weights


def _version_children_from_evidence(evidence: Dict[str, Any]) -> List[str]:
    target = str(evidence.get("target_service", "") or "")
    if not target:
        return []
    return [f"{target}:v1", f"{target}:v2"]


def _canonical_t4_node_id(value: Any, target_service: str) -> str:
    text = str(value)
    prefix = f"{target_service}:"
    if not target_service or not text.startswith(prefix):
        return text
    suffix = text[len(prefix):].lower()
    if suffix.startswith("v1") or "v1" in suffix:
        return f"{target_service}:v1"
    if suffix.startswith("v2") or "v2" in suffix:
        return f"{target_service}:v2"
    return text


def _canonical_t4_pod_id(value: Any, target_service: str, added_pods: set) -> str:
    text = str(value)
    if not target_service or not added_pods:
        return text
    pod_prefix = f"pod:{target_service}-v2-"
    if text.startswith(pod_prefix) and text not in added_pods:
        return sorted(added_pods)[0]
    return text


def _t4_upstream_services_from_evidence(evidence: Dict[str, Any]) -> List[str]:
    target = str(evidence.get("target_service", "") or "")
    summary = evidence.get("dependency_edge_summary", {})
    upstreams = []
    if isinstance(summary, dict):
        for item in summary.get("upstream_dependencies", []) or []:
            if isinstance(item, dict) and item.get("source"):
                upstreams.append(str(item["source"]))
    if not upstreams:
        fallback = {
            "currencyservice": ["checkoutservice", "frontend"],
            "productcatalogservice": ["frontend", "recommendationservice", "checkoutservice"],
            "recommendationservice": ["frontend"],
            "shippingservice": ["checkoutservice"],
        }
        upstreams = list(fallback.get(target, []))
    return sorted(dict.fromkeys(upstreams))


def _known_node_ids_from_evidence(evidence: Dict[str, Any]) -> set:
    nodes = set()
    target = str(evidence.get("target_service", "") or "")
    if target:
        nodes.add(target)
    trace_summary = evidence.get("trace_route_summary", {})
    for service in trace_summary.get("services", []) if isinstance(trace_summary, dict) else []:
        if service:
            nodes.add(str(service))
    for item in evidence.get("pod_diff_summary", {}).get("changed_services", []) or []:
        if not isinstance(item, dict):
            continue
        service = item.get("service")
        if service:
            nodes.add(str(service))
        for pod_name in list(item.get("added", []) or []) + list(item.get("removed", []) or []):
            nodes.add(_normalize_node_id(pod_name, infer_pod=True))
    endpoint_summary = evidence.get("endpoint_diff_summary", {})
    for item in endpoint_summary.get("changed_services", []) if isinstance(endpoint_summary, dict) else []:
        if not isinstance(item, dict):
            continue
        service = item.get("service")
        if service:
            nodes.add(str(service))
        for key in ("before_backend_services", "after_backend_services"):
            for backend in item.get(key, []) or []:
                if not isinstance(backend, dict):
                    continue
                if backend.get("service"):
                    nodes.add(str(backend["service"]))
                if backend.get("pod"):
                    nodes.add(_normalize_node_id(backend["pod"], infer_pod=True))
    for upstream in _t4_upstream_services_from_evidence(evidence):
        nodes.add(upstream)
    return nodes


def _weight_targets_from_evidence(evidence: Dict[str, Any]) -> List[str]:
    scenario = str(evidence.get("scenario_id", "") or "").upper()
    if scenario == "T4":
        return _version_children_from_evidence(evidence)
    targets = _all_target_pod_nodes(evidence)
    if not targets:
        targets = _added_pod_nodes(evidence)
    return targets


def _ratio(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric / 100.0 if numeric > 1.0 else numeric


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

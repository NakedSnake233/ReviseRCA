"""Paired, label-blind simulated-operator pilot on counterfactual controls.

This is an agent simulation, not the E8 human study. A controlled endpoint
representation error is seeded without labels. All three conditions see the
same observation and the same probabilistically available collector reference;
only H2 can edit through the production workbench.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import uuid
from collections import Counter
from copy import deepcopy
from pathlib import Path

import torch
from jsonschema import Draft4Validator

from dyrca_rca.experiments.inference import load_checkpoint
from dyrca_rca.interaction.observation import anonymous_id, canonical, digest, evidence, load_observation
from dyrca_rca.interaction.state import effective_state
from dyrca_rca.interaction.workbench import Workbench
from dyrca_rca.planner.llm_planner import ResponsesAPIClient
from dyrca_rca.planner import llm_planner as llm_client_module


DIAGNOSIS_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["service", "mechanism", "evidence_ids", "confidence"],
    "properties": {
        "service": {"type": "string"},
        "mechanism": {"type": "string", "minLength": 1},
        "evidence_ids": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}
REPAIR_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["action", "service", "endpoint_ips", "reason"],
    "properties": {
        "action": {"enum": ["repair", "none"]},
        "service": {"type": "string"},
        "endpoint_ips": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
}
DECISION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["decision", "reason"],
    "properties": {"decision": {"enum": ["accept", "reject"]}, "reason": {"type": "string"}},
}


def _read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _latest(view):
    return max((item for item in view["evidence"] if item["kind"] == "topology"),
               key=lambda item: item["data"]["timestamp"])


def seed_endpoint_defect(view, seed, inject=True):
    """Return a copied observation, collector proxy, and frozen defect record."""
    original = _latest(view)
    eligible = [svc for svc in original["data"]["services"]
                if isinstance(svc.get("endpoint_ips"), list) and svc["endpoint_ips"]]
    if not eligible:
        raise ValueError("no nonempty observed endpoint set to perturb")
    chosen = sorted(eligible, key=lambda svc: svc["name"])
    selected = chosen[random.Random(seed ^ int(digest(view["incident_id"])[:12], 16)).randrange(len(chosen))]
    proxy = {"service": selected["name"], "endpoint_ips": list(selected["endpoint_ips"]),
             "source": "collector_during_snapshot_proxy"}
    altered = deepcopy(view)
    old_id = original["id"]
    if inject:
        for index, item in enumerate(altered["evidence"]):
            if item["id"] == old_id:
                item["data"]["services"] = [
                    dict(svc, endpoint_ips=[]) if svc["name"] == selected["name"] else svc
                    for svc in item["data"]["services"]
                ]
                altered["evidence"][index] = evidence("topology", item["data"])
                break
        altered["evidence"].sort(key=lambda item: item["id"])
        altered["observation_version"] = digest({k: v for k, v in altered.items()
                                                   if k != "observation_version"})
    return altered, proxy, {
        "service": selected["name"], "field": "endpoint_ips",
        "before": [] if inject else list(selected["endpoint_ips"]),
        "collector_reference_hash": digest(proxy), "seeded_defect": inject,
        "source_snapshot_id": old_id,
    }


def _visible_packet(view, prediction=None, oracle=None):
    snapshots = sorted((item for item in view["evidence"] if item["kind"] == "topology"),
                       key=lambda item: item["data"]["timestamp"])
    latest = snapshots[-1]
    traces = [item for item in view["evidence"] if item["kind"] == "trace_service"]
    packet = {
        "application": view["system_id"],
        "as_of": view["as_of"],
        "service_topology": [
            {"name": svc["name"], "endpoint_ips": svc.get("endpoint_ips")}
            for svc in latest["data"]["services"]
        ],
        "pre_service_topology": [
            {"name": svc["name"], "endpoint_ips": svc.get("endpoint_ips")}
            for svc in snapshots[0]["data"]["services"]
        ] if len(snapshots) > 1 else [],
        "trace_service_evidence": [
            dict(id=item["id"], **item["data"]) for item in traces
        ],
        "oracle_check": oracle,
        "eligible_evidence_ids": sorted(item["id"] for item in view["evidence"]),
    }
    if prediction is not None:
        packet["ai_recommendation"] = {
            "ranked_services": prediction["ranked_node_ids"][:12],
            "top_score": prediction.get("scores", {}).get(prediction["ranked_node_ids"][0])
            if prediction["ranked_node_ids"] else None,
        }
    return packet


def _fallback_diagnosis(packet):
    services = [svc["name"] for svc in packet["service_topology"]]
    ai = packet.get("ai_recommendation")
    if ai and ai["ranked_services"]:
        service = ai["ranked_services"][0]
    else:
        stats = {item["service"]: item for item in packet["trace_service_evidence"]}
        service = sorted(services, key=lambda name: (
            -(stats.get(name, {}).get("error_count", 0) /
              max(1, stats.get(name, {}).get("status_count", 0))), name))[0]
    refs = [item["id"] for item in packet["trace_service_evidence"]
            if item["service"] == service]
    return {"service": service, "mechanism": "undetermined_from_eligible_evidence",
            "evidence_ids": refs[:2], "confidence": 0.0}


def _externalize_packet(packet, kind):
    """Replace local identifiers with case-local aliases before any API call."""
    services, endpoints, evidence_ids = set(), set(), set()
    for key in ("service_topology", "pre_service_topology"):
        for item in packet.get(key, []):
            services.add(item["name"])
            endpoints.update(item.get("endpoint_ips") or [])
    for item in packet.get("trace_service_evidence", []):
        services.add(item["service"])
        evidence_ids.add(item["id"])
    evidence_ids.update(packet.get("eligible_evidence_ids", []))
    oracle = packet.get("oracle_check")
    if oracle:
        services.add(oracle["service"])
        endpoints.update(oracle["endpoint_ips"])
    ai = packet.get("ai_recommendation")
    if ai:
        services.update(ai["ranked_services"])
    repair = packet.get("repair")
    if repair:
        services.add(repair["service"])
        endpoints.update(repair["endpoint_ips"])
    preview = packet.get("preview", {})
    services.update(preview.get(key) for key in ("before_top1", "after_top1")
                    if isinstance(preview.get(key), str))
    endpoints.update(preview.get("before_endpoint_ips") or [])
    endpoints.update(preview.get("after_endpoint_ips") or [])
    services.discard("")
    smap = {value: "S%03d" % index for index, value in enumerate(sorted(services), 1)}
    emap = {value: "E%03d" % index for index, value in enumerate(sorted(endpoints), 1)}
    vmap = {value: "V%03d" % index for index, value in enumerate(sorted(evidence_ids), 1)}

    def alias(value, field=None):
        if field in ("application", "as_of"):
            return "redacted"
        if field in ("name", "service", "before_top1", "after_top1"):
            return smap.get(value, value)
        if field in ("ranked_services",):
            return [smap.get(item, item) for item in value]
        if field in ("endpoint_ips", "before_endpoint_ips", "after_endpoint_ips"):
            return [emap[item] for item in value] if value is not None else None
        if field in ("id",):
            return vmap.get(value, value)
        if field == "eligible_evidence_ids":
            return [vmap[item] for item in value]
        if isinstance(value, dict):
            return {key: alias(item, key) for key, item in value.items()}
        if isinstance(value, list):
            return [alias(item) for item in value]
        return value

    remote = alias(packet)
    reverse_services = {value: key for key, value in smap.items()}
    reverse_endpoints = {value: key for key, value in emap.items()}
    reverse_evidence = {value: key for key, value in vmap.items()}

    def restore(output):
        local = dict(output)
        if kind == "diagnose":
            local["service"] = reverse_services.get(output["service"], output["service"])
            local["evidence_ids"] = [reverse_evidence.get(item, item)
                                     for item in output["evidence_ids"]]
        elif kind == "repair":
            local["service"] = reverse_services.get(output["service"], output["service"])
            local["endpoint_ips"] = [reverse_endpoints.get(item, item)
                                     for item in output["endpoint_ips"]]
        return local

    return remote, restore


def _call_agent(kind, packet, schema, fallback, args, case_id, condition):
    if args.planner == "rule":
        return fallback, "rule", "selected_rule_planner"
    instruction = {
        "diagnose": "Select the most likely faulty logical service. Give a cautious mechanism and cite only eligible evidence IDs. Do not infer an injection label. Return one JSON object.",
        "repair": "Decide whether the oracle topology check justifies repairing one endpoint_ips field. If no oracle check exists, return action none. Use exactly the checked service and IPs; do not guess. Return one JSON object.",
        "decide": "Review the proposed endpoint correction and preview. Topology fidelity is the criterion: a model top-1 ranking may remain unchanged after a valid graph correction. Accept if the patch exactly matches the collector check and the preview shows the corrected endpoint object; otherwise reject. Return one JSON object.",
    }[kind]
    try:
        remote_packet, restore = _externalize_packet(packet, kind)
        prompt = instruction + "\nSchema: " + canonical(schema) + "\nPacket: " + canonical(remote_packet)
        client = ResponsesAPIClient.from_env()
        # Each stage is a self-contained one-turn request. Use a distinct,
        # stable routing session so the provider cannot mix stage histories.
        client.session_id = str(uuid.UUID(
            hex=digest([args.seed, case_id, condition, kind])[:32]))
        response = client(prompt)
        output = json.loads(response)
        Draft4Validator(schema).validate(output)
        output = restore(output)
        if kind == "diagnose":
            candidates = {svc["name"] for svc in packet["service_topology"]}
            if output["service"] not in candidates:
                raise ValueError("diagnosis outside service inventory")
            if not set(output["evidence_ids"]) <= set(packet["eligible_evidence_ids"]):
                raise ValueError("diagnosis cites unavailable evidence")
        return output, "llm", None
    except Exception as exc:
        return fallback, "rule_fallback", type(exc).__name__


def _diagnose(view, prediction, oracle, args, case_id, condition):
    packet = _visible_packet(view, prediction, oracle)
    fallback = _fallback_diagnosis(packet)
    result, source, error = _call_agent("diagnose", packet, DIAGNOSIS_SCHEMA, fallback,
                                        args, case_id, condition)
    return result, source, error


def _repair_plan(view, prediction, oracle, args, case_id):
    packet = _visible_packet(view, prediction, oracle)
    current_endpoints = None
    if oracle:
        current_endpoints = next(svc.get("endpoint_ips") for svc in
                                 _latest(view)["data"]["services"]
                                 if svc["name"] == oracle["service"])
    needs_repair = oracle is not None and current_endpoints != oracle["endpoint_ips"]
    fallback = ({"action": "repair", "service": oracle["service"],
                 "endpoint_ips": oracle["endpoint_ips"], "reason": "collector_reference_match"}
                if needs_repair else {"action": "none", "service": "", "endpoint_ips": [],
                                      "reason": "no_repair_needed_or_no_verified_check"})
    result, source, error = _call_agent("repair", packet, REPAIR_SCHEMA, fallback,
                                        args, case_id, "H2")
    raw_repair_proposed = result["action"] == "repair"
    if raw_repair_proposed and (
        not needs_repair or result["service"] != oracle["service"]
        or result["endpoint_ips"] != oracle["endpoint_ips"]
    ):
        # A valid but wrong model proposal must not be silently replaced with
        # the correct oracle value; doing so would inflate correction accuracy.
        rejected = {"action": "none", "service": "", "endpoint_ips": [],
                    "reason": "unsupported_repair_rejected_by_guard"}
        return rejected, "llm_guard_rejected", "unsupported_repair", True, True
    return result, source, error, raw_repair_proposed, False


def _decision(oracle, repair, preview, before_top1, before_endpoint_ips, args, case_id):
    # The previous stage's free-text reason may contain aliases from a larger
    # inventory. Re-alias only structured fields for this standalone request.
    checked_repair = {key: repair[key] for key in ("action", "service", "endpoint_ips")}
    comparison = preview["comparison"]
    packet = {"oracle_check": oracle, "repair": checked_repair,
              "preview": {"before_top1": before_top1,
                          "after_top1": preview["preview"]["prediction"]["ranked_node_ids"][0],
                          "before_endpoint_ips": before_endpoint_ips,
                          "after_endpoint_ips": repair["endpoint_ips"],
                          "nodes_changed_count": len(comparison["nodes"]["changed"])
                          + len(comparison["nodes"]["added"]) + len(comparison["nodes"]["removed"]),
                          "edges_changed_count": len(comparison["edges"]["changed"])
                          + len(comparison["edges"]["added"]) + len(comparison["edges"]["removed"]),
                          "evidence_changed_count": len(comparison["evidence"]["changed"])
                          + len(comparison["evidence"]["added"]) + len(comparison["evidence"]["removed"])}}
    fallback = {"decision": "accept", "reason": "exact_collector_reference"}
    return _call_agent("decide", packet, DECISION_SCHEMA, fallback, args, case_id, "H2")


def run(args):
    raw, out, pilot = Path(args.data_root).resolve(), Path(args.out_dir).resolve(), Path(args.pilot_dir).resolve()
    if (not 0 <= args.oracle_prob <= 1 or not 0 <= args.defect_prob <= 1
            or args.limit < 1 or args.max_rows < 1):
        raise ValueError("invalid probability or limit")
    if args.planner == "llm" and not args.allow_external_evidence:
        raise ValueError("LLM mode requires explicit --allow-external-evidence")
    if args.planner == "llm":
        configured = ResponsesAPIClient.from_env()
        if (configured.model != "gpt-5.6-luna"
                or configured.base_url != "https://opencode.ai/zen/go/v1"):
            raise ValueError("LLM pilot requires the frozen OpenCode Go gpt-5.6-luna endpoint")
    if out == raw or raw in out.parents or out.exists() and any(out.iterdir()):
        raise ValueError("output must be a fresh directory outside raw data")
    protocol = json.loads((pilot / "protocol.json").read_text())
    if protocol["system"] != args.system or Path(protocol["data_root"]).resolve() != raw:
        raise ValueError("pilot protocol does not match selected application/data root")
    checkpoint = Path(protocol["checkpoint"])
    model, model_name = load_checkpoint(str(checkpoint), "auto")
    model.eval()
    torch.set_num_threads(1)
    frozen_weights = {key: value.clone() for key, value in model.state_dict().items()}
    inventory = {anonymous_id(protocol["dataset_id"], path.name): path
                 for path in raw.iterdir() if (path / "metadata.json").exists()}
    manifest = _read_jsonl(pilot / "collection" / "observation_manifest.jsonl")
    ids = sorted(row["case_id"] for row in manifest)
    random.Random(args.seed).shuffle(ids)
    ids = ids[:args.limit]
    if len(ids) < args.limit or any(case_id not in inventory for case_id in ids):
        raise ValueError("selected cases are unavailable")
    out.mkdir(parents=True)
    config = {"status": "simulated_agent_pilot_not_human_study", "data_root": str(raw),
              "pilot_dir": str(pilot), "pilot_protocol_sha256": _sha256(pilot / "protocol.json"),
              "checkpoint_sha256": _sha256(checkpoint), "seed": args.seed,
              "runner_sha256": _sha256(Path(__file__)),
              "llm_client_sha256": _sha256(Path(llm_client_module.__file__)),
              "source_revision": None, "source_revision_status": "git_metadata_unavailable",
              "planner": args.planner, "model": "gpt-5.6-luna" if args.planner == "llm" else None,
              "external_payload": "case_local_aliases_and_aggregate_counts_only"
              if args.planner == "llm" else "none",
              "limit": args.limit, "max_rows": args.max_rows,
              "oracle_prob": args.oracle_prob, "case_ids": ids,
              "defect_prob": args.defect_prob,
              "defect_policy": "label_blind_bernoulli_seeded_empty_endpoint_or_no_repair",
              "oracle_source": "same_collector_during_snapshot_proxy_not_independent_review",
              "cutoff_limitation": protocol["cutoff_rule"]}
    (out / "protocol.json").write_text(canonical(config) + "\n")
    rows, events = [], []
    for index, case_id in enumerate(ids, 1):
        path = inventory[case_id]
        view = load_observation(path, protocol["dataset_id"], args.system, max_rows=args.max_rows)
        if view["incident_id"] != case_id:
            raise ValueError("case mapping changed")
        inject_defect = (
            random.Random(args.seed ^ int(digest([case_id, "defect"])[:12], 16)).random()
            < args.defect_prob
        )
        corrupted, reference, defect = seed_endpoint_defect(view, args.seed, inject_defect)
        oracle_access = random.Random(args.seed ^ int(digest([case_id, "oracle"])[:12], 16)).random() < args.oracle_prob
        oracle = reference if oracle_access else None
        wb = Workbench(out / "workbench" / case_id, [corrupted], model, model_name,
                       config["checkpoint_sha256"])
        baseline = wb.snapshot(case_id)["baseline"]
        for condition in ("H0", "H1"):
            prediction = None if condition == "H0" else baseline["prediction"]
            diag, source, error = _diagnose(corrupted, prediction, oracle, args, case_id, condition)
            rows.append({"case_id": case_id, "condition": condition, "agent_output": diag,
                         "agent_source": source, "fallback_error": error,
                         "oracle_access": oracle_access, "oracle_reference_hash": digest(reference),
                         "observation_version": corrupted["observation_version"],
                         "initial_model_top1": baseline["prediction"]["ranked_node_ids"][0],
                         "final_model_top1": None if condition == "H0" else baseline["prediction"]["ranked_node_ids"][0],
                         "seeded_defect": defect, "repair_proposed": False,
                         "repair_accepted": False,
                         "topology_object_correct_after": not inject_defect})
        repair, repair_source, repair_error, raw_repair_proposed, guard_rejected = _repair_plan(
            corrupted, baseline["prediction"], oracle, args, case_id)
        accepted = False
        decision_source, decision_error, decision_reason = None, None, None
        if repair["action"] == "repair":
            fid = "agent-repair-" + case_id
            base = {"incident_id": case_id, "base_revision": 0, "actor": "simulated_sre_agent",
                    "reason": "SIMULATION: collector topology reference; no human review"}
            proposal = dict(base, action="propose", event_id=fid, task="T2",
                            service=repair["service"], values={"endpoint_ips": repair["endpoint_ips"]})
            preview_command = dict(base, action="preview", event_id=fid + "-preview", target_id=fid)
            wb.command(proposal)
            preview = wb.command(preview_command)
            before_endpoint_ips = next(
                svc.get("endpoint_ips") for svc in _latest(corrupted)["data"]["services"]
                if svc["name"] == repair["service"])
            decision, decision_source, decision_error = _decision(
                oracle, repair, preview,
                baseline["prediction"]["ranked_node_ids"][0],
                before_endpoint_ips, args, case_id)
            decision_reason = decision["reason"]
            action = "accept_feedback" if decision["decision"] == "accept" else "reject_feedback"
            wb.command(dict(base, action=action, event_id=fid + "-decision", target_id=fid))
            accepted = action == "accept_feedback"
            events.append({"case_id": case_id, "proposal": proposal, "preview": preview_command,
                           "decision": action, "decision_source": decision_source,
                           "decision_reason": decision_reason,
                           "decision_fallback_error": decision_error})
        current = wb.snapshot(case_id)
        effective, _ = effective_state(current["session"])
        latest = _latest(effective)
        final_object = next(svc.get("endpoint_ips") for svc in latest["data"]["services"]
                            if svc["name"] == reference["service"])
        diag, source, error = _diagnose(effective, current["current"]["prediction"],
                                        oracle, args, case_id, "H2")
        rows.append({"case_id": case_id, "condition": "H2", "agent_output": diag,
                     "agent_source": source, "fallback_error": error,
                     "oracle_access": oracle_access, "oracle_reference_hash": digest(reference),
                     "observation_version": corrupted["observation_version"],
                     "initial_model_top1": baseline["prediction"]["ranked_node_ids"][0],
                     "final_model_top1": current["current"]["prediction"]["ranked_node_ids"][0],
                     "seeded_defect": defect, "repair_proposed": repair["action"] == "repair",
                     "repair_raw_proposal": raw_repair_proposed,
                     "repair_guard_rejected": guard_rejected,
                     "repair_source": repair_source, "repair_fallback_error": repair_error,
                     "decision_source": decision_source,
                     "decision_reason": decision_reason,
                     "repair_accepted": accepted,
                     "topology_object_correct_after": final_object == reference["endpoint_ips"]})
        print("simulated %s/%s %s" % (index, len(ids), case_id), flush=True)
    if any(not torch.equal(frozen_weights[key], value)
           for key, value in model.state_dict().items()):
        raise RuntimeError("frozen model weights changed during agent simulation")
    pred_path = out / "agent_predictions.jsonl"
    pred_path.write_text("".join(canonical(row) + "\n" for row in rows))
    (out / "workbench_events.jsonl").write_text("".join(canonical(row) + "\n" for row in events))
    prediction_hash = _sha256(pred_path)
    # No metadata or labels are opened above this line.
    case_meta = {row["case_id"]: row for row in
                 _read_jsonl(pilot / "collection" / "cases.jsonl")}
    labels = {row["case_id"]: row["root_cause_service"]
              for row in _read_jsonl(pilot / "collection" / "labels.jsonl")}
    scored = [dict(row, root_service=labels.get(row["case_id"])
                   if not case_meta[row["case_id"]]["is_benign"] else None,
                   is_benign=case_meta[row["case_id"]]["is_benign"],
                   service_correct=(row["agent_output"]["service"] == labels[row["case_id"]])
                   if row["case_id"] in labels and not case_meta[row["case_id"]]["is_benign"]
                   else None) for row in rows]
    (out / "scored.jsonl").write_text("".join(canonical(row) + "\n" for row in scored))
    fault_rows = [row for row in scored if row["service_correct"] is not None]
    condition_source_counts = {
        condition: dict(Counter(row["agent_source"] for row in rows
                                if row["condition"] == condition))
        for condition in ("H0", "H1", "H2")}
    condition_fault_hit1_by_source = {}
    for condition in ("H0", "H1", "H2"):
        sources = sorted({row["agent_source"] for row in fault_rows
                          if row["condition"] == condition})
        condition_fault_hit1_by_source[condition] = {}
        for source in sources:
            subset = [row for row in fault_rows if row["condition"] == condition
                      and row["agent_source"] == source]
            condition_fault_hit1_by_source[condition][source] = {
                "n": len(subset),
                "hit1": sum(row["service_correct"] for row in subset) / len(subset)}
    by_case = {case_id: {row["condition"]: row for row in scored
                        if row["case_id"] == case_id} for case_id in ids}
    llm_only_fault_ids = [
        case_id for case_id in ids if not case_meta[case_id]["is_benign"]
        and by_case[case_id]["H1"]["agent_source"] == "llm"
        and by_case[case_id]["H2"]["agent_source"] == "llm"
        and by_case[case_id]["H2"]["repair_source"] == "llm"
        and (not by_case[case_id]["H2"]["repair_proposed"]
             or by_case[case_id]["H2"]["decision_source"] == "llm")
    ]
    summary = {"status": "simulated_agent_pilot_not_human_study",
               "prediction_sha256_before_label_read": prediction_hash,
               "case_count": len(ids),
               "fault_case_count": sum(not case_meta[case_id]["is_benign"] for case_id in ids),
               "benign_case_count": sum(case_meta[case_id]["is_benign"] for case_id in ids),
               "seeded_defect_case_count": sum(row["seeded_defect"]["seeded_defect"]
                                               for row in rows if row["condition"] == "H2"),
               "no_repair_case_count": sum(not row["seeded_defect"]["seeded_defect"]
                                           for row in rows if row["condition"] == "H2"),
               "oracle_access_case_count": sum(row["oracle_access"] for row in rows if row["condition"] == "H0"),
               "agent_source_counts": dict(Counter(row["agent_source"] for row in rows)),
               "condition_agent_source_counts": condition_source_counts,
               "condition_fault_hit1_by_source": condition_fault_hit1_by_source,
               "llm_only_fault_paired": {
                   "n": len(llm_only_fault_ids),
                   "h1_hit1": (sum(by_case[case_id]["H1"]["service_correct"]
                                   for case_id in llm_only_fault_ids) / len(llm_only_fault_ids))
                   if llm_only_fault_ids else None,
                   "h2_hit1": (sum(by_case[case_id]["H2"]["service_correct"]
                                   for case_id in llm_only_fault_ids) / len(llm_only_fault_ids))
                   if llm_only_fault_ids else None},
               "repair_proposed": sum(row["repair_proposed"] for row in rows if row["condition"] == "H2"),
               "repair_accepted": sum(row["repair_accepted"] for row in rows if row["condition"] == "H2"),
               "llm_accepted_repairs": sum(row["repair_accepted"] and row["repair_source"] == "llm"
                                           for row in rows if row["condition"] == "H2"),
               "rule_fallback_accepted_repairs": sum(
                   row["repair_accepted"] and row["repair_source"] == "rule_fallback"
                   for row in rows if row["condition"] == "H2"),
               "repair_source_counts": dict(Counter(row.get("repair_source")
                                                     for row in rows if row["condition"] == "H2")),
               "decision_source_counts": dict(Counter(row.get("decision_source")
                                                       for row in events)),
               "seeded_defects_corrected": sum(row["topology_object_correct_after"]
                                               for row in rows if row["condition"] == "H2"
                                               and row["seeded_defect"]["seeded_defect"]),
               "no_repair_raw_repair_proposals": sum(row["repair_raw_proposal"]
                                                     for row in rows if row["condition"] == "H2"
                                                     and not row["seeded_defect"]["seeded_defect"]),
               "accepted_unnecessary_edits_after_guard": sum(
                   row["repair_accepted"] for row in rows if row["condition"] == "H2"
                   and not row["seeded_defect"]["seeded_defect"]),
               "repair_guard_rejections": sum(row["repair_guard_rejected"]
                                              for row in rows if row["condition"] == "H2"),
               "paired_fault_service_outcomes": {
                   state: sum(
                       (next(row["service_correct"] for row in scored
                             if row["case_id"] == case_id and row["condition"] == "H1"),
                        next(row["service_correct"] for row in scored
                             if row["case_id"] == case_id and row["condition"] == "H2")) == pair
                       for case_id in ids if not case_meta[case_id]["is_benign"])
                   for state, pair in (
                       ("both_wrong", (False, False)), ("h1_only", (True, False)),
                       ("h2_only", (False, True)), ("both_correct", (True, True)))},
               "condition_service_hit1": {
                   condition: (
                       sum(row["service_correct"] for row in scored
                           if row["condition"] == condition and row["service_correct"] is not None)
                       / sum(row["condition"] == condition and row["service_correct"] is not None
                             for row in scored)
                   ) if any(row["condition"] == condition and row["service_correct"] is not None
                            for row in scored) else None
                   for condition in ("H0", "H1", "H2")},
               "limitations": ["simulated agent, no human participants or 8-minute human task timing",
                               "collector snapshot is an oracle proxy, not independent topology review",
                               "seeded defect incidence is not natural builder error prevalence",
                               "service-only outcome; mechanism/evidence correctness lacks independent rubric",
                               "pilot cohort lacks independent two-reviewer verification"]}
    (out / "summary.json").write_text(canonical(summary) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pilot-dir", required=True)
    parser.add_argument("--system", required=True, choices=("boutique", "trainticket"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", choices=("llm", "rule"), default="rule")
    parser.add_argument("--allow-external-evidence", action="store_true",
                        help="Explicitly authorize aliased topology and aggregate trace evidence sent to the LLM provider")
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--oracle-prob", type=float, default=0.5)
    parser.add_argument("--defect-prob", type=float, default=0.5)
    print(canonical(run(parser.parse_args())), flush=True)


if __name__ == "__main__":
    main()

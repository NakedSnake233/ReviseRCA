"""Measured, bounded R2 follow-up: source consistency and paired API revisions.

Collector snapshots are an explicitly limited reference, not physical causal
truth. Two endpoint defects are seeded without labels, predictions are frozen
before scoring, and neither model parameters nor raw collection files change.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from dyrca_rca.experiments.backfill_revised_source import frozen_stage, sha

PROJECT = Path(__file__).resolve().parents[3]
CONDITIONS = ("none", "correct", "redundant", "nomination", "partial", "incorrect")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def json_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percent(values):
    return 100 * statistics.mean(values) if values else None


def normalize_service(name):
    return re.sub(r"-v[0-9]+$", "", name)


def version_groups(snapshot):
    groups = defaultdict(set)
    for pod in snapshot.get("pods", []):
        if pod.get("service") and pod.get("version") not in (None, "", "unknown"):
            groups[normalize_service(pod["service"])].add(pod["version"])
    return {key: sorted(value) for key, value in sorted(groups.items())}


def raw_objects(case, cutoff):
    """A separate raw-JSON parser, independent of graph/model predictions."""
    from dyrca_rca.interaction.observation import epoch
    values, availability, sources, unknown = {}, {}, {}, 0
    for phase in ("pre", "during"):
        file = Path(case) / "topology" / (phase + ".json")
        if not file.exists():
            continue
        snapshot = read_json(file)
        if epoch(snapshot["timestamp"]) > epoch(cutoff):
            continue
        sources[phase] = {"path": str(file.resolve()), "sha256": sha(file),
                          "timestamp": snapshot["timestamp"]}
        for collection in ("services", "pods", "deployments"):
            availability[(phase, collection)] = isinstance(snapshot.get(collection), list)
        for service in snapshot.get("services", []):
            value = service.get("endpoint_ips")
            if isinstance(value, list):
                values[(phase, "endpoints", service["name"])] = tuple(sorted(set(value)))
            else:
                unknown += 1
        for pod in snapshot.get("pods", []):
            version = pod.get("version")
            if pod.get("service") and version not in (None, "", "unknown"):
                values[(phase, "pod_version_group", pod["name"])] = (pod["service"], version)
            else:
                unknown += 1
    return values, availability, sources, unknown


def view_objects(view):
    values, availability = {}, {}
    for fact in view["evidence"]:
        if fact["kind"] != "topology":
            continue
        data, phase = fact["data"], fact["data"]["phase"]
        for collection in ("services", "pods", "deployments"):
            availability[(phase, collection)] = collection in data.get("observed_collections", [])
        for service in data["services"]:
            value = service.get("endpoint_ips")
            if isinstance(value, list):
                values[(phase, "endpoints", service["name"])] = tuple(sorted(set(value)))
        for pod in data["pods"]:
            if pod.get("service") and pod.get("version") not in (None, "", "unknown"):
                values[(phase, "pod_version_group", pod["name"])] = (pod["service"], pod["version"])
    return values, availability


def fidelity(reference, actual, defect_keys=()):
    keys = set(reference) | set(actual)
    defects = set(defect_keys) & set(reference)
    unaffected = set(reference) - defects
    return {"fidelity": percent([key in reference and key in actual and actual[key] == reference[key] for key in sorted(keys)]),
            "repair": percent([actual.get(key) == reference[key] for key in sorted(defects)]),
            "damage": percent([actual.get(key) != reference[key] for key in sorted(unaffected)]),
            "objects": len(keys), "defects": len(defects), "unaffected": len(unaffected),
            "actual_only_objects": len(set(actual) - set(reference))}


def source_manifest(cases):
    """Stream hashes of consumed source files; targets are bound separately after inference."""
    files = set()
    for case in cases:
        for relative in ('topology/pre.json', 'topology/during.json', 'k8s_events/post_pods.json',
                         'traces/trace_summary.csv'):
            path = case / relative
            if path.is_file():
                files.add(path.resolve())
        files.update(p.resolve() for p in (case / 'metrics').glob('*.csv') if p.is_file())
    return {str(path): sha(path) for path in sorted(files)}


def bind_cases_to_root(cases, root):
    root = Path(root).resolve()
    if any(root not in Path(case).resolve().parents for case in cases):
        raise ValueError('Frozen case lies outside the declared --data-root')


def target_rank(prediction, target):
    ranked = prediction['ranked_node_ids']
    return ranked.index(target) + 1 if target in ranked else len(ranked) + 1


def target_reciprocal_rank(prediction, target):
    return 1 / target_rank(prediction, target) if target in prediction['ranked_node_ids'] else 0.0


def read_frozen_target(path, expected_sha256):
    contents=Path(path).read_bytes()
    checksum=hashlib.sha256(contents).hexdigest()
    if checksum!=expected_sha256:
        raise ValueError('Target label differs from receipt-bound collection audit: '+str(path))
    return json.loads(contents)['root_cause_service'],checksum


def corrupt_endpoints(view, seed):
    """Two phase-specific objects at one randomly selected logical service."""
    from dyrca_rca.interaction.observation import digest, evidence
    snapshots = sorted((fact for fact in view["evidence"] if fact["kind"] == "topology"),
                       key=lambda fact: fact["data"]["timestamp"])
    by_phase = {fact["data"]["phase"]: fact for fact in snapshots}
    if set(by_phase) != {"pre", "during"}:
        raise ValueError("Paired endpoint defects require two eligible phases")
    common = None
    for fact in snapshots:
        eligible = {svc["name"] for svc in fact["data"]["services"] if svc.get("endpoint_ips")}
        common = eligible if common is None else common & eligible
    if not common:
        raise ValueError("No nonempty service endpoint pair to perturb")
    rng = random.Random(seed ^ int(digest([view["incident_id"], "paired-endpoints-v1"])[:12], 16))
    chosen = sorted(common)[rng.randrange(len(common))]
    altered = deepcopy(view)
    objects = []
    for i, fact in enumerate(altered["evidence"]):
        if fact["kind"] != "topology":
            continue
        service = next(svc for svc in fact["data"]["services"] if svc["name"] == chosen)
        old = list(service["endpoint_ips"])
        service["endpoint_ips"] = []
        altered["evidence"][i] = evidence("topology", fact["data"])
        objects.append({"phase": fact["data"]["phase"], "service": chosen,
                        "source_id": fact["id"], "reference": old, "seeded_value": []})
    altered["evidence"].sort(key=lambda fact: fact["id"])
    altered["observation_version"] = digest({k: v for k, v in altered.items() if k != "observation_version"})
    return altered, chosen, sorted(objects, key=lambda obj: obj["phase"])


def tensor_signature(item):
    """Bind every consumed tensor and candidate order, not only graph JSON."""
    h = hashlib.sha256()
    h.update(json.dumps(item["node_ids"], sort_keys=True).encode())
    for name in ("x", "edge_index", "edge_attr"):
        value = item[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def event(session, eid, operation, target, payload=None, **extra):
    view = session["observation"]
    return dict(event_id=eid, incident_id=session["incident_id"], base_revision=session["state_revision"],
                operation=operation, target_id=target, payload=payload or {}, reason="source-reference follow-up",
                actor="controlled_content_policy", scope={"system_id": view["system_id"], "as_of": view["as_of"]}, **extra)


def patch(session, eid, phase, service, value, model, model_name, log):
    from dyrca_rca.interaction.state import effective_state, apply_event
    from dyrca_rca.interaction.pipeline import replay
    current = effective_state(session)[0]
    fact = next(f for f in current["evidence"] if f["kind"] == "topology" and f["data"]["phase"] == phase)
    row = next(s for s in fact["data"]["services"] if s["name"] == service)
    proposal = event(session, eid, "patch_topology", fact["id"],
                     {"collection":"services", "entity":service, "field":"endpoint_ips",
                      "expected":row["endpoint_ips"], "value":value}, event_kind="feedback")
    proposed = apply_event(session, proposal, lambda _: None)
    preview = replay(proposed, model, model_name, mode="preview", feedback_id=eid)[0]
    decision = event(proposed, eid + "-accept", "accept_feedback", eid, event_kind="decision")
    accepted = apply_event(proposed, decision, lambda state: replay(state, model, model_name))
    log.append({"proposal":proposal, "decision":decision,
                "execution_status":accepted["outcomes"][decision["event_id"]]["execution_status"],
                "preview_prediction":preview["prediction"], "preview_graph_hash":hash_json(preview["graph"])})
    return accepted


def hash_json(value):
    from dyrca_rca.interaction.observation import digest
    return digest(value)


def masked_prediction(result, model, model_name):
    """Remove all operation-derived channels and recompute composite weights."""
    from dyrca_rca.dataset.materialize import graph_to_model_tensors
    from dyrca_rca.experiments.inference import predict_observed
    from dyrca_rca.graph.weights import WEIGHT_COEFFICIENTS
    graph = deepcopy(result["graph"])
    for edge in graph["edges"]:
        attrs = edge["attrs"]
        for name in ("op", "change_type", "reweighted", "trace_mismatch", "redirected_from"):
            attrs.pop(name, None)
        attrs["change_weight"] = 0.0
        available = [key for key in WEIGHT_COEFFICIENTS if attrs.get(key) is not None]
        attrs["final_weight"] = sum(WEIGHT_COEFFICIENTS[key] * attrs[key] for key in available) / sum(WEIGHT_COEFFICIENTS[key] for key in available)
        edge["weight"] = attrs["final_weight"]
    item = graph_to_model_tensors(graph, {"experiment_id":result["incident_id"]})
    item.pop("label", None)
    item['metadata']['feature_protocol'] = graph['attrs']['feature_protocol']
    names = item["metadata"]["node_feature_names"]
    for name in ("max_incident_change_weight", "incident_update_count"):
        item["x"][:, names.index(name)] = 0
    item["metadata"]["experiment_id"] = result["incident_id"]
    return predict_observed(model, item, model_name), tensor_signature(item)


def paired_revisions(case, view, reference, seed, model, model_name):
    from dyrca_rca.interaction.state import new_session, effective_state, apply_event
    from dyrca_rca.interaction.pipeline import replay
    broken, chosen, defect_records = corrupt_endpoints(view, seed)
    defect_keys = {(obj["phase"], "endpoints", chosen) for obj in defect_records}
    records, histories, audits = [], [], []
    for stratum, initial in (("seeded", broken), ("unperturbed", view)):
        base_session = new_session(initial)
        baseline, base_item = replay(base_session, model, model_name)
        universe = sorted(baseline["prediction"]["ranked_node_ids"])
        baseline_signature = tensor_signature(base_item)
        for condition in CONDITIONS:
            session, log = deepcopy(base_session), []
            if condition not in ("none", "nomination"):
                for obj in defect_records:
                    if condition == "partial" and obj["phase"] != "during":
                        continue
                    if condition in ("correct", "partial"):
                        value = obj["reference"]
                    elif condition == "redundant":
                        value = [] if stratum == "seeded" else obj["reference"]
                    else:
                        # Valid string identity, explicitly false content; no ranking-based choice.
                        value = ["source-audit-unsupported-endpoint"]
                    session = patch(session, f"{stratum}-{condition}-{obj['phase']}", obj["phase"], chosen,
                                    value, model, model_name, log)
            result, item = replay(session, model, model_name)
            prediction = deepcopy(result["prediction"])
            recommendation_order = list(prediction['ranked_node_ids'])
            if condition == "nomination":
                recommendation_order = [chosen] + [name for name in recommendation_order if name != chosen]
            if sorted(prediction["ranked_node_ids"]) != universe:
                raise ValueError("Revision changed the frozen service candidate universe")
            actual = view_objects(effective_state(session)[0])[0]
            content = fidelity(reference, actual, defect_keys if stratum == "seeded" else ())
            record = {"experiment":case.name, "case_id":view["incident_id"], "stratum":stratum,
                      "condition":condition, "nominated_service":chosen, "prediction":prediction,
                      "recommendation_order":recommendation_order,
                      "recommendation_policy":'forced post-processing; unchanged model scores' if condition == 'nomination' else 'model ranking',
                      "baseline_prediction":baseline["prediction"], "input_hash":tensor_signature(item),
                      "baseline_input_hash":baseline_signature, "content":content,
                      "attempted_commands":len(log), "accepted_commands":sum(row["execution_status"] == "succeeded" for row in log)}
            if stratum == "seeded" and condition in ("none", "correct"):
                record["tag_masked_prediction"], record["tag_masked_input_hash"] = masked_prediction(result, model, model_name)
            records.append(record)
            histories.append({"case_id":view["incident_id"], "stratum":stratum, "condition":condition,
                              "defects":defect_records if stratum == "seeded" else [], "events":log})
            if condition == "redundant":
                duplicate = log[-1]["decision"]
                repeated = apply_event(session, duplicate, lambda _: (_ for _ in ()).throw(AssertionError("Retry must not replay")))
                plan_proposal = event(session, "equivalent-plan", "replace_plan", "graph_plan", deepcopy(result["plan"]), event_kind="feedback")
                trial = apply_event(session, plan_proposal, lambda _: None)
                trial = apply_event(trial, event(trial,"equivalent-plan-accept","accept_feedback","equivalent-plan",event_kind="decision"),
                                    lambda state: replay(state,model,model_name))
                equivalent, equivalent_item = replay(trial, model, model_name)
                audits.append({"experiment":case.name,"stratum":stratum,
                               "duplicate_state_equal":repeated == session,
                               "redundant_input_changed":tensor_signature(item) != baseline_signature,
                               "redundant_prediction":prediction,"baseline_prediction":baseline["prediction"],
                               "equivalent_plan_input_changed":tensor_signature(equivalent_item) != tensor_signature(item),
                               "equivalent_plan_prediction":equivalent["prediction"],
                               "channels":{key:sum(float(edge['attrs'].get(key,0) or 0) for edge in result['graph']['edges'])
                                           for key in ('change_weight','final_weight')}})
    return records, histories, audits


def score_revisions(records, labels):
    grouped = defaultdict(list)
    for row in records:
        target = labels[row["experiment"]]
        recommendation = {'ranked_node_ids': row.get('recommendation_order', row['prediction']['ranked_node_ids'])}
        row["target_rank"] = target_rank(recommendation, target)
        row["baseline_target_rank"] = target_rank(row["baseline_prediction"], target)
        row["hit1"] = row["target_rank"] == 1
        row["mrr"] = target_reciprocal_rank(recommendation, target)
        if "tag_masked_prediction" in row:
            row["tag_masked_target_rank"] = target_rank(row["tag_masked_prediction"], target)
            row['tag_masked_mrr'] = target_reciprocal_rank(row['tag_masked_prediction'], target)
        grouped[(row["stratum"],row["condition"])].append(row)
    summaries = {}
    for (stratum,condition), group in grouped.items():
        proposals = sum(row["attempted_commands"] for row in group)
        summaries.setdefault(stratum,{})[condition] = {
            "n":len(group),"accepted":100*sum(row["accepted_commands"] for row in group)/proposals if proposals else None,
            "commands":proposals,"h1":percent([row["hit1"] for row in group]),"mrr":statistics.mean(row["mrr"] for row in group),
            "better":sum(row["target_rank"] < row["baseline_target_rank"] for row in group),
            "same":sum(row["target_rank"] == row["baseline_target_rank"] for row in group),
            "worse":sum(row["target_rank"] > row["baseline_target_rank"] for row in group),
            **{key:statistics.mean(values) if values else None for key in ("fidelity","repair","damage")
               for values in [[row["content"][key] for row in group if row["content"][key] is not None]]}}
    return summaries


def source_audit(cases, config):
    from dyrca_rca.interaction.observation import load_observation
    records = []
    for case in cases:
        view = load_observation(case, config["dataset_id"], config["system"], max_rows=config["max_rows"])
        reference, avail, sources, unknown = raw_objects(case, view["as_of"])
        actual, actual_avail = view_objects(view)
        keys = set(reference) | set(actual)
        endpoints = [key for key in keys if key[1] == "endpoints"]
        versions = [key for key in keys if key[1] == "pod_version_group"]
        precision, recall = [], []
        for key in endpoints:
            observed, ref = set(actual.get(key, ())), set(reference.get(key, ()))
            precision.append(len(observed & ref)/len(observed) if observed else float(not ref))
            recall.append(len(observed & ref)/len(ref) if ref else float(not observed))
        different = [key for key in keys if key not in reference or key not in actual or actual[key] != reference[key]]
        during = next(f['data'] for f in view['evidence'] if f['kind']=='topology' and f['data']['phase']=='during')
        multiversion = {key:value for key,value in version_groups(during).items() if len(value)>1}
        provenance_supported = {}
        for fact in view['evidence']:
            if fact['kind'] != 'topology':
                continue
            phase = fact['data']['phase']
            provenance_supported[phase] = (phase in sources and fact['data']['timestamp'] == sources[phase]['timestamp']
                                           and fact['id'] == 'topology:' + hash_json(fact['data'])[:24])
        # No numeric route reference is present in the frozen topology/CSV schema.
        records.append({"experiment":case.name,"case_id":view['incident_id'],"objects":len(keys),"reference_objects":len(reference),
                        "endpoint_objects":len(endpoints),"version_objects":len(versions),
                        "source_paths":sources,"source_reference_type":"collector snapshot, independently parsed; not external truth",
                        "unknown_values":unknown,"route_groups_without_value_reference":len(multiversion),
                        "endpoint_precision":statistics.mean(precision) if precision else None,
                        "endpoint_recall":statistics.mean(recall) if recall else None,
                        "version_agreement":percent([key in reference and key in actual and actual[key]==reference[key] for key in versions]),
                        "availability_correct":sum(actual_avail.get(key)==value for key,value in avail.items()),
                        "availability_objects":len(avail),"source_status_correct":sum(provenance_supported.values()),
                        "source_status_objects":len(provenance_supported),"fidelity":fidelity(reference,actual)['fidelity'],
                        "actual_only_objects":len(set(actual) - set(reference)),
                        "natural_discrepancies":len(different),"packet_discoverable":bool(different),
                        "endpoint_form_actionable":any(key[0]=='during' and key[1]=='endpoints' for key in different),
                        "multiversion_groups":multiversion})
    return {"scope":"collector-source local-field consistency, not independent causal/global physical fidelity",
            "aggregation":"Endpoint and Pod agreement: case-macro of per-case object-macro; fidelity: case-macro over union of reference and actual keys",
            "source_status_scope":"One timestamp/content-address check per topology snapshot, not independent object-level provenance",
            "cases":len(records),"records":records,"objects":sum(r['objects'] for r in records),
            "reference_objects":sum(r['reference_objects'] for r in records),
            "actual_only_objects":sum(r['actual_only_objects'] for r in records),
            "objects_without_reference":sum(r['unknown_values']+r['route_groups_without_value_reference']+r['actual_only_objects'] for r in records),
            "endpoint_precision":percent([r['endpoint_precision'] for r in records if r['endpoint_precision'] is not None]),
            "endpoint_recall":percent([r['endpoint_recall'] for r in records if r['endpoint_recall'] is not None]),
            "version_agreement":statistics.mean(r['version_agreement'] for r in records if r['version_agreement'] is not None),
            "availability_correct":sum(r['availability_correct'] for r in records),
            "availability_objects":sum(r['availability_objects'] for r in records),
            "source_status_correct":sum(r['source_status_correct'] for r in records),
            "source_status_objects":sum(r['source_status_objects'] for r in records),
            "fidelity":statistics.mean(r['fidelity'] for r in records),
            "natural_discrepancy_cases":sum(bool(r['natural_discrepancies']) for r in records),
            "packet_discoverable_cases":sum(r['packet_discoverable'] for r in records),
            "endpoint_form_actionable_cases":sum(r['endpoint_form_actionable'] for r in records),
            "supported_route_groups":0,"route_l1_error":None,
            "t4_cases":sum(r['experiment'].startswith('t4-') for r in records),
            "t4_numeric_repair_cases":0,
            "t4_restraint_cases":sum(r['experiment'].startswith('t4-') for r in records)}


def operation_latencies(cases, config, out, model, model_name):
    from dyrca_rca.interaction.observation import load_observation
    from dyrca_rca.interaction.workbench import Workbench
    samples, checks = defaultdict(list), []
    for case in cases[:10]:
        view = load_observation(case, config['dataset_id'],config['system'],max_rows=config['max_rows'])
        wb = Workbench(out/'latency_workbench'/case.name,[view],model,model_name,config['checkpoint_hash'])
        incident = view['incident_id']
        path, original = wb.cases[incident]
        latest = next(f for f in view['evidence'] if f['kind']=='topology' and f['data']['phase']=='during')
        service = next(s for s in latest['data']['services'] if s.get('endpoint_ips'))
        for index in range(25):
            wb.cases[incident] = (path,deepcopy(original))
            wb.persist(path,original)
            base={'incident_id':incident,'base_revision':0,'actor':'latency_fixture','reason':'fixed operation boundary'}
            wb.command(dict(base,action='propose',event_id='a',task='T2',service=service['name'],values={'endpoint_ips':[]}))
            start=time.perf_counter_ns()
            wb.command(dict(base,action='preview',target_id='a'))
            preview=(time.perf_counter_ns()-start)/1e6
            start=time.perf_counter_ns()
            wb.command(dict(base,action='accept_feedback',event_id='accept-a',target_id='a'))
            accept=(time.perf_counter_ns()-start)/1e6
            base['base_revision']=1
            wb.command(dict(base,action='propose',event_id='b',task='T2',service=service['name'],values={'endpoint_ips':service['endpoint_ips']}))
            wb.command(dict(base,action='accept_feedback',event_id='accept-b',target_id='b'))
            before=wb.snapshot(incident)
            base['base_revision']=2
            start=time.perf_counter_ns()
            failed=wb.command(dict(base,action='retract_feedback',event_id='retract-a',target_id='a'))
            conflict=(time.perf_counter_ns()-start)/1e6
            retained=(failed['session']['state_revision']==before['session']['state_revision']
                      and failed['current']==before['current']
                      and failed['session']['outcomes']['retract-a']['execution_status']=='failed')
            if not retained:
                raise ValueError('Conflict replay did not preserve formal evidence/plan/graph/score')
            if index>=5:
                for op,value in [('preview',preview),('accept',accept),('conflict',conflict)]:
                    samples[op].append({'experiment':case.name,'repeat':index-5,'ms':value})
                checks.append(retained)
    def quantile(values,q):
        values=sorted(values)
        return values[round((len(values)-1)*q)]
    summary={op:{'n':len(rows),'p50_ms':quantile([r['ms'] for r in rows],.5),'p95_ms':quantile([r['ms'] for r in rows],.95)}
             for op,rows in samples.items()}
    write_json(out/'operation_latency_samples.json',dict(samples))
    return {'cases':min(10,len(cases)),'warmups':5,'repeats':20,'scope':'Workbench API including persistence for accept/conflict; warm OS file cache',
            'retained_formal_state':sum(checks),'conflict_attempts':len(checks),'summaries':summary}


def weight_predictions(cases, config, model, model_name):
    from dyrca_rca.interaction.observation import load_observation
    from dyrca_rca.interaction.state import new_session
    from dyrca_rca.interaction.pipeline import replay, WEIGHT_COEFFICIENTS
    original=dict(WEIGHT_COEFFICIENTS)
    rows=[]
    try:
        for case in cases:
            view=load_observation(case,config['dataset_id'],config['system'],max_rows=config['max_rows'])
            for coefficient in (0.,.10,.30):
                WEIGHT_COEFFICIENTS.update(original)
                factor=(1-coefficient)/(1-original['change_weight'])
                WEIGHT_COEFFICIENTS.update({k:v*factor for k,v in original.items() if k!='change_weight'})
                WEIGHT_COEFFICIENTS['change_weight']=coefficient
                result,item=replay(new_session(view),model,model_name)
                shares=[]
                for edge in result['graph']['edges']:
                    attrs=edge['attrs']
                    available=[key for key in WEIGHT_COEFFICIENTS if attrs.get(key) is not None]
                    denominator=sum(WEIGHT_COEFFICIENTS[key] for key in available)
                    shares.append(WEIGHT_COEFFICIENTS['change_weight']/denominator if denominator and 'change_weight' in available else 0.)
                rows.append({'experiment':case.name,'change_coefficient':coefficient,
                             'coefficients':dict(WEIGHT_COEFFICIENTS),'input_hash':tensor_signature(item),'prediction':result['prediction'],
                             'effective_change_coefficient_min':min(shares) if shares else None,
                             'effective_change_coefficient_max':max(shares) if shares else None})
    finally:
        WEIGHT_COEFFICIENTS.clear();WEIGHT_COEFFICIENTS.update(original)
    return rows


def agent_history_audit(agent):
    from dyrca_rca.interaction.state import effective_state, feedback_statuses
    from dyrca_rca.interaction.observation import anonymous_id
    scored=json_rows(agent/'scored.jsonl')
    events=json_rows(agent/'workbench_events.jsonl')
    protocol=read_json(agent/'protocol.json')
    pilot=read_json(Path(protocol['pilot_dir'])/'protocol.json')
    if sha(Path(protocol['pilot_dir'])/'protocol.json')!=protocol['pilot_protocol_sha256']:
        raise ValueError('Agent setup changed after the simulation')
    by_case=defaultdict(dict)
    for row in scored: by_case[row['case_id']][row['condition']]=row
    inventory={anonymous_id(pilot['dataset_id'],p.name):p for p in Path(protocol['data_root']).iterdir() if (p/'topology/during.json').is_file()}
    classifications=Counter()
    movements=[]
    checks=[]
    for case_id,conditions in sorted(by_case.items()):
        files=list((agent/'workbench'/case_id).glob('*.json'))
        if len(files)!=1:
            classifications['unassessable']+=1; continue
        wb=read_json(files[0]); current=effective_state(wb['session'])[0]
        reference=read_json(inventory[case_id]/'topology/during.json')
        reference={s['name']:s.get('endpoint_ips') for s in reference.get('services',[])}
        statuses=feedback_statuses(wb['session'])
        accepted=[e for e in wb['session']['events'] if e['operation']=='patch_topology'
                  and statuses[e['event_id']]=='accepted']
        if not accepted:
            stratum='no_edits'
        else:
            supported=True;redundant=True
            for e in accepted:
                p=e['payload']; supported &= p['value']==reference.get(p['entity'])
                redundant &= p['value']==p['expected']
                before=wb['baseline']['prediction']['ranked_node_ids']
                after=wb['current']['prediction']['ranked_node_ids']
                movements.append({'case_id':case_id,'service':p['entity'],'source_supported':p['value']==reference.get(p['entity']),
                                  'before_rank':before.index(p['entity'])+1 if p['entity'] in before else None,
                                  'after_rank':after.index(p['entity'])+1 if p['entity'] in after else None})
            stratum='redundant_edits' if redundant else 'supported_edits' if supported else 'other_edits'
        classifications[stratum]+=1
        checks.append({'case_id':case_id,'stratum':stratum,'initial_wrong':conditions['H2']['initial_model_top1']!=conditions['H2']['root_service'],
                       'final_correct':bool(conditions['H2']['service_correct']),'source_file_sha256':sha(files[0])})
    rescue={}
    for stratum in ('supported_edits','status_only','redundant_edits','other_edits','no_edits','unassessable'):
        eligible=[r for r in checks if r['stratum']==stratum and r['initial_wrong']]
        rescue[stratum]={'correct':sum(r['final_correct'] for r in eligible),'eligible':len(eligible),'all_tasks':classifications[stratum]}
    return {'scope':'Application-specific collector-reference audit of executed Luna API simulation, not GUI human logs',
            'cases':len(by_case),'preview_commands':len(events),'unaccepted_previews':sum(r['decision']!='accept_feedback' for r in events),
            'classifications':dict(classifications),'rescue':rescue,'edited_service_movement':movements,'case_checks':checks,
            'log_schema':'workbench-v1 / granular-feedback-v1','analysis_timing':'post-hoc after service outcomes',
            'source_hashes':{p.name:sha(p) for p in (agent/'protocol.json',agent/'scored.jsonl',agent/'workbench_events.jsonl')},
            'workbench_source_hash':sha(PROJECT/'src/dyrca_rca/interaction/workbench.py')}


def t4_walkthrough(cases, config, model, model_name, planner):
    from dyrca_rca.interaction.observation import load_observation
    from dyrca_rca.interaction.state import new_session
    from dyrca_rca.interaction.pipeline import replay
    candidates=[p for p in sorted(cases) if p.name.startswith('t4-')]
    if not candidates:
        return {'status':'bounded_no_t4','case':None}
    case=candidates[0]
    view=load_observation(case,config['dataset_id'],config['system'],max_rows=config['max_rows'])
    session=new_session(view)
    result,_=replay(session,model,model_name)
    during=next(f for f in view['evidence'] if f['kind']=='topology' and f['data']['phase']=='during')
    groups={k:v for k,v in version_groups(during['data']).items() if len(v)>1}
    safe_packet={'observed_version_groups':[{'service':f'S{i:03d}','versions':[f'V{j:03d}' for j,_ in enumerate(v,1)]}
                    for i,(_,v) in enumerate(groups.items(),1)],'numeric_route_weight_reference':None,
                 'configured_route_metadata_in_packet':False,'rule':'Require source-backed weights to submit a full numeric routing plan.'}
    schema={'type':'object','additionalProperties':False,'required':['action','rationale'],
            'properties':{'action':{'enum':['retain_unknown','submit_numeric_plan']},'rationale':{'type':'string','minLength':1}}}
    fallback={'action':'retain_unknown','rationale':'Eligible packet has version membership but no configured per-version routing reference.'}
    prompt='Act as an SRE. Decide whether the shared evidence supports an exact numeric per-version routing plan. Do not use injection metadata. Return JSON. Schema: '+json.dumps(schema)+' Packet: '+json.dumps(safe_packet)
    answer,source,error=fallback,'rule',None
    response_text=None
    request_attempts=0
    request_seconds=None
    if planner=='llm':
        try:
            import uuid
            from jsonschema import Draft4Validator
            from dyrca_rca.planner.llm_planner import ResponsesAPIClient
            client=ResponsesAPIClient.from_env()
            if client.model!='gpt-5.6-luna' or client.base_url!='https://opencode.ai/zen/go/v1':
                raise ValueError('Unexpected SRE simulation provider/model')
            client.session_id=str(uuid.UUID(hex=hash_json([view['incident_id'],'t4-restraint-v1'])[:32]))
            request_attempts=1
            request_start=time.perf_counter()
            response_text=client(prompt)
            request_seconds=time.perf_counter()-request_start
            answer=json.loads(response_text)
            Draft4Validator(schema).validate(answer)
            source='llm'
        except Exception as exc:
            answer,source,error=fallback,'rule_fallback',type(exc).__name__
    return {'status':'recorded_restraint','case':case.name,'case_id':view['incident_id'],
            'selection':'First lexicographic T4 in frozen held-out split, without inspecting prediction',
            'natural_or_seeded':'unperturbed observation-only unsplit state; no seeded defect',
            'source_ids':[f['id'] for f in view['evidence'] if f['kind']=='topology'],
            'version_groups':groups,'eligible_numeric_route_groups':0,'proposal':None,'preview':None,'decision':None,
            'agent_action':answer['action'],'agent_source':source,'fallback_error':error,
            'agent_answer':answer,'agent_response_text':response_text,'anonymous_request_packet':safe_packet,
            'anonymous_request_schema':schema,'anonymous_request_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'request_attempts':request_attempts,'request_seconds':request_seconds,
            'request_endpoint':'https://opencode.ai/zen/go/v1/responses' if request_attempts else None,
            'request_model':'gpt-5.6-luna' if request_attempts else None,
            'effective_action':'retain_unknown_routing; no unsupported numeric plan committed',
            'state_revision_before':0,'state_revision_after':0,'graph_hash_before':hash_json(result['graph']),
            'graph_hash_after':hash_json(result['graph']),'prediction':result['prediction'],
            'version_membership_reference':'observed raw Pod labels; -vN service suffix maps to base logical service',
            'route_fidelity':None,'route_fidelity_denominator':0,'session_unchanged':True,
            'privacy':'Only local aliases and source-availability flags sent to model'}


def deployment_inventory(cases):
    images, resolved, total, rates = set(), 0, 0, []
    source_hashes={}
    for case in cases:
        snapshot=read_json(case/'topology/during.json')
        for dep in snapshot.get('deployments',[]):
            total+=1; resolved+=dep.get('version') not in (None,'','unknown')
        # Post-export image metadata is inventory only, never a diagnosis/reference input.
        path=case/'k8s_events/post_pods.json'
        if path.is_file():
            data=read_json(path)
            source_hashes[str(path.resolve())]=sha(path)
            for pod in (data.get('items',[]) if isinstance(data,dict) else data):
                for container in pod.get('spec',{}).get('containers',pod.get('containers',[])):
                    if container.get('image'):images.add(container['image'])
        start=snapshot['timestamp']
        pre=read_json(case/'topology/pre.json')['timestamp']
        path=case/'metrics/request_rate.csv'
        if path.is_file():
            by_time=defaultdict(float)
            with path.open(newline='') as stream:
                for row in csv.DictReader(stream):
                    if pre<=row['timestamp']<=start:
                        labels=json.loads(row['labels_json'])
                        if labels.get('destination_workload')=='frontend':
                            by_time[row['timestamp']]+=float(row['value'])
            if by_time:rates.append(statistics.mean(by_time.values()))
    return {'image_strings':sorted(images),'post_image_source_hashes':source_hashes,
            'deployment_labels_resolved':resolved,'deployment_labels_total':total,
            'request_rate_cases':len(rates),'frontend_inbound_rps_median':statistics.median(rates) if rates else None,
            'request_rate_scope':'pre/during topology interval, inbound frontend; not total cross-service traffic',
            'historical_application_git_revision':None,'image_inventory_scope':'post-export inventory only; not time-eligible model/reference evidence'}


def validate_fresh_output(raw, out, module):
    """Allow only the launcher's pre-created task and live log in a fresh attempt."""
    if out == raw or raw in out.parents:
        raise ValueError('Output must be fresh and outside raw data')
    if not out.exists():
        return
    if not out.is_dir():
        raise ValueError('Output must be a fresh directory')
    entries = list(out.iterdir())
    if not entries:
        return
    if any(p.is_symlink() or not p.is_file() or p.name not in ('stage.json', 'run.log') for p in entries):
        raise ValueError('Output already contains experiment artifacts')
    if not (out/'stage.json').is_file():
        raise ValueError('Existing launch files require a matching stage specification')
    spec = read_json(out/'stage.json')
    if spec.get('module') != module or not isinstance(spec.get('args'), list):
        raise ValueError('Stage specification does not match this runner')
    for flag, expected in (('--out-dir', out), ('--data-root', raw)):
        argv = spec['args']
        if argv.count(flag) != 1 or argv.index(flag)+1 >= len(argv) or Path(argv[argv.index(flag)+1]).resolve() != expected:
            raise ValueError('Stage specification has mismatched output/data roots')


def run(args):
    import torch
    from dyrca_rca.experiments.inference import load_checkpoint
    from dyrca_rca.interaction.observation import load_observation
    from dyrca_rca.collection_safety import read_memory_state
    raw,out=args.data_root.resolve(),args.out_dir.resolve()
    validate_fresh_output(raw,out,'dyrca_rca.experiments.complete_revised_experiments')
    if args.seed!=42 or args.limit is not None and args.limit<1:
        raise ValueError('Prespecified seed42 and positive bounded limit required')
    if read_memory_state()['available_gib']<10+args.memory_cap_gib:
        raise RuntimeError('Memory admission deferred; preserve collection reserve')
    app=args.system
    release=frozen_stage(args.experiment_root,app,'release_s42_t42')
    prepared=frozen_stage(args.experiment_root,app,'03_prepare')
    if not release or not prepared: raise ValueError('Completed frozen release required')
    config=read_json(release/'split.json')
    for name in ('interaction/observation.py','interaction/state.py','interaction/pipeline.py','interaction/tasks.py',
                 'dataset/materialize.py','models/config.py','experiments/inference.py','graph/weights.py'):
        if sha(PROJECT/'src/dyrca_rca'/name)!=config['source_hashes'][name]:
            raise ValueError('Frozen execution source changed: '+name)
    checkpoint=release/'temporal_v3.pt'
    config['checkpoint_hash']=sha(checkpoint)
    torch.set_num_threads(1)
    model,model_name=load_checkpoint(checkpoint,'auto')
    frozen={key:value.clone() for key,value in model.state_dict().items()}
    data=Path(config['data_root'])
    test=[data/name for name in sorted(config['split']['test'])]
    audit_cases=[data/name for name in sorted(config['split']['train']+config['split']['val']+config['split']['test'])]
    if args.limit is not None:test=test[:args.limit];audit_cases=test
    bind_cases_to_root(audit_cases,raw)
    initial_source_hashes=source_manifest(audit_cases)
    collection_audit=frozen_stage(args.experiment_root,app,'02_audit')
    if not collection_audit:raise ValueError('Frozen collection label hashes required')
    out.mkdir(parents=True,exist_ok=True)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    protocol={'schema_version':'r2-completion-v2','seed':args.seed,'planner':args.planner,
              'scorer_planner':'rule','t4_agent':args.planner,'limit':args.limit,'data_root':str(raw),
              'system':app,'checkpoint':str(checkpoint),'checkpoint_sha256':config['checkpoint_hash'],
              'split_sha256':sha(release/'split.json'),'runner_sha256':sha(Path(__file__)),
              'test_experiments':[p.name for p in test],'source_audit_experiments':[p.name for p in audit_cases],
              'conditions':CONDITIONS,'defect_policy':'two endpoint fields, pre/during, at a label-blind random common nonempty service',
              'candidate_policy':'retain identical logical-service universe in every paired condition',
              'reference':'independently parsed collector raw snapshots, not model output or independently adjudicated causal truth',
              'known_outcome_timing':'exploratory follow-up after original main ranking was observed',
              'nomination_policy':'forced recommendation post-processing; model prediction/scores preserved separately',
              'target_hash_binding':'ground_truth.json checked against the receipt-bound collection audit after predictions close; labels parsed only then',
              'source_manifest_sha256':hash_json(initial_source_hashes),
              'weight_coefficients':[0,.10,.30],'latency_warmups':5,'latency_repeats':20,'cpu_threads':1,
              'memory_cap_gib':args.memory_cap_gib,'host_reserve_gib':10,'host_admit_gib':10+args.memory_cap_gib}
    write_json(out/'protocol.json',protocol)
    prediction_path=out/'revision_predictions.jsonl'
    history_path=out/'revision_histories.jsonl'
    records,audits=[],[]
    with prediction_path.open('w') as pred,history_path.open('w') as history:
        for i,case in enumerate(test,1):
            if read_memory_state()['available_gib']<10:
                raise RuntimeError('Host memory fell below collection reserve; partial run not publishable')
            view=load_observation(case,config['dataset_id'],config['system'],max_rows=config['max_rows'])
            if view['observation_version']!=read_json(release/'report.json')['observation_hashes'][case.name]:
                raise ValueError('Case packet differs from frozen release')
            reference=raw_objects(case,view['as_of'])[0]
            rows,hist,checks=paired_revisions(case,view,reference,args.seed,model,model_name)
            records.extend(rows);audits.extend(checks)
            for row in rows:pred.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            for row in hist:history.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            pred.flush();history.flush()
            print(f'paired revisions {i}/{len(test)}',flush=True)
    prediction_hash=sha(prediction_path)
    weight_rows=weight_predictions(test,config,model,model_name)
    write_json(out/'weight_predictions.json',weight_rows)
    weight_hash=sha(out/'weight_predictions.json')
    # The first opening of evaluation targets happens after prediction artifacts close/hash.
    audit_rows=json_rows(collection_audit/'collection_audit.jsonl')
    frozen_target_hashes={row['case_id']:row['hashes']['ground_truth.json'] for row in audit_rows}
    frozen_sources={row['case_id']:row['hashes'] for row in audit_rows}
    for case in audit_cases:
        for relative in ('topology/pre.json','topology/during.json'):
            if sha(case/relative)!=frozen_sources[case.name][relative]:
                raise ValueError('Source-audit snapshot differs from frozen collection hash: '+case.name+'/'+relative)
    labels,target_hashes={},{}
    for case in test:
        target,checksum=read_frozen_target(case/'ground_truth.json',frozen_target_hashes[case.name])
        target_hashes[str((case/'ground_truth.json').resolve())]=checksum
        labels[case.name]=target
    summaries=score_revisions(records,labels)
    write_json(out/'revision_summary.json',{'scope':'source-reference oracle-content API experiment; not actual SRE behavior',
        'cases':len(test),'summaries':summaries,'prediction_sha256_before_label_read':prediction_hash,'strata':['seeded','unperturbed'],
        'classification':'seeded representation-error simulation on measured incidents', 'checkpoint_sha256':config['checkpoint_hash']})
    write_json(out/'revision_scored.json',records)
    for row in audits:
        target=labels[row['experiment']]
        row['redundant_target_rank_changed']=target_rank(row['redundant_prediction'],target)!=target_rank(row['baseline_prediction'],target)
        row['equivalent_target_rank_changed']=target_rank(row['equivalent_plan_prediction'],target)!=target_rank(row['redundant_prediction'],target)
    masked={}
    for condition in ('none','correct'):
        group=[r for r in records if r['stratum']=='seeded' and r['condition']==condition]
        masked[condition]={'h1':percent([r['tag_masked_target_rank']==1 for r in group]),
                           'mrr':statistics.mean(r['tag_masked_mrr'] for r in group),'input_identity_count':sum(r['input_hash']==r['tag_masked_input_hash'] for r in group)}
    write_json(out/'edit_audit.json',{'scope':'fixed-checkpoint no-op and complete tag-derived-input masking sensitivity; primary seeded stratum',
        'records':audits,'seeded_cases':len(test),'masked':masked,
        'channels_accounted':['edge change_weight','edge final_weight','node max_incident_change_weight','node incident_update_count','node max_incident_final_weight recomputed'],
        'duplicate_equal':sum(r['duplicate_state_equal'] for r in audits),'duplicate_attempts':len(audits)})
    weight_summary={}
    for coefficient in (0.,.10,.30):
        group=[r for r in weight_rows if r['change_coefficient']==coefficient]
        ranks=[target_rank(r['prediction'],labels[r['experiment']]) for r in group]
        shares=[r[key] for r in group for key in ('effective_change_coefficient_min','effective_change_coefficient_max') if r[key] is not None]
        weight_summary[str(coefficient)]={'n':len(group),'h1':percent([rank==1 for rank in ranks]),
                'mrr':statistics.mean(target_reciprocal_rank(r['prediction'],labels[r['experiment']]) for r in group),
                'effective_change_share_min':min(shares) if shares else None,'effective_change_share_max':max(shares) if shares else None}
    write_json(out/'weight_sensitivity.json',{'scope':'three fixed research-variable settings, inference sensitivity only; no selection or retraining','settings':weight_summary,'prediction_sha256_before_label_read':weight_hash})
    print('source consistency audit',flush=True)
    write_json(out/'source_audit.json',source_audit(audit_cases,config))
    write_json(out/'deployment_inventory.json',deployment_inventory(audit_cases))
    print('operation boundaries',flush=True)
    write_json(out/'operation_latency.json',operation_latencies(test,config,out,model,model_name))
    agent_stage=frozen_stage(args.experiment_root,app,'08_luna')
    if agent_stage:write_json(out/'agent_history_audit.json',agent_history_audit(agent_stage/'agent'))
    print('T4 session',flush=True)
    walkthrough=t4_walkthrough(test,config,model,model_name,args.planner)
    if walkthrough['case']:walkthrough['target_rank']=target_rank(walkthrough['prediction'],labels[walkthrough['case']])
    write_json(out/'t4_walkthrough.json',walkthrough)
    if any(not torch.equal(frozen[key],value) for key,value in model.state_dict().items()):
        raise ValueError('Model parameters changed during revisions')
    final_source_hashes=source_manifest(audit_cases)
    if initial_source_hashes!=final_source_hashes or any(sha(Path(path))!=checksum for path,checksum in target_hashes.items()):
        raise ValueError('Consumed source/target files changed during follow-up')
    write_json(out/'source_integrity.json',{'status':'verified','consumed_files':len(initial_source_hashes),
               'consumed_files_sha256_before':initial_source_hashes,'consumed_files_sha256_after':final_source_hashes,
               'target_files':len(target_hashes),'target_sha256':target_hashes,
               'frozen_target_hash_source':str(collection_audit/'collection_audit.jsonl'),
               'frozen_target_hash_source_sha256':sha(collection_audit/'collection_audit.jsonl'),
               'scope':'Named consumed files unchanged, targets also match frozen collection hashes; not a whole-repository raw-data scan'})
    # stdout appends the final summary after run() returns, so the live log
    # cannot be part of the immutable measurement-file manifest.
    files=[p for p in sorted(out.iterdir()) if p.is_file() and p.name != 'run.log']
    summary={'schema_version':'r2-completion-v2','status':'bounded_smoke' if args.limit is not None else 'completed','system':app,
             'seed':args.seed,'planner':args.planner,'limit':args.limit,'test_cases':len(test),'audit_cases':len(audit_cases),
             'scorer_planner':'rule','t4_agent':args.planner,
             'paired_conditions':6,'paired_strata':2,'prediction_records':len(records),'checkpoint_unchanged':True,
             'consumed_source_files_unchanged':True,'frozen_targets_verified':len(target_hashes),
             'protocol_sha256':sha(out/'protocol.json'),'artifact_hashes':{p.name:sha(p) for p in files}}
    write_json(out/'completion_summary.json',summary)
    return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=PROJECT/'data/experiments_tosem_four_copies')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--experiment-root',type=Path,default=PROJECT/'artifacts/tosem_four_copies/experiments_20260930')
    p.add_argument('--system',choices=('boutique','trainticket'),default='boutique')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--planner',choices=('rule','llm'),default='rule',help='Scorer graph remains rule-based; optional Luna SRE in T4 only')
    p.add_argument('--limit',type=int)
    p.add_argument('--memory-cap-gib',type=float,default=8)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(int(args.memory_cap_gib*1024**3),int(args.memory_cap_gib*1024**3)))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    print(json.dumps(run(args),sort_keys=True))

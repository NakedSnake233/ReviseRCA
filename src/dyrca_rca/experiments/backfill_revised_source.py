"""Bind completed four-copy measurements to the uploaded R2 result registry.

No collection, injection, training, or LLM calls are performed. Partial shards
cannot supply application results. Copied evidence, immutable receipts, and
field-level denominators accompany the updated, compilable manuscript.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
METHOD_KEYS = {"NSigma-Dyn": "nsigma", "BARO-Dyn": "baro", "CIRCA-Dyn": "circa",
               "CloudRanger-Dyn": "cloudranger", "Eadro-Dyn": "eadro", "HolisticRCA-Dyn": "holistic",
               "Change-only": "change", "Node-only": "node", "DY-RCA": "dyrca"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def configuration(name):
    return name.rsplit("-run", 1)[0]


def escape(value):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_\allowbreak{}", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(char, char) for char in str(value))


def number(value, percent=False, digits=2):
    return f"{float(value) * (100 if percent else 1):.{digits}f}"


def replace_registry(text, bindings, command="SetResult"):
    """Replace complete single-line registry entries, preserving TeX braces."""
    pattern = re.compile(r"^(\\" + command + r"\{([a-zA-Z0-9_.-]+)\}\{)(.*)(\}\s*)$", re.M)
    declared = {match.group(2) for match in pattern.finditer(text)}
    def substitute(match):
        value = bindings.get(match.group(2))
        return match.group(1) + value + match.group(4) if value is not None else match.group(0)
    updated = pattern.sub(substitute, text)
    for key in sorted(set(bindings) - declared):
        updated += f"\\{command}{{{key}}}{{{bindings[key]}}}\n"
    return updated


def cluster_interval(values, seed=42, replicates=2000):
    """Resample configuration clusters, retaining all incident rows per draw."""
    groups = [values[key] for key in sorted(values)]
    if not groups:
        raise ValueError("Cannot estimate an interval without eligible clusters")
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sample = [value for _ in groups for value in groups[rng.randrange(len(groups))]]
        draws.append(statistics.mean(sample))
    draws.sort()
    return {"estimate": statistics.mean(value for group in groups for value in group),
            "low": draws[int(.025 * replicates)], "high": draws[int(.975 * replicates)],
            "groups": len(groups), "replicates": replicates}


def frozen_stage(root, app, stage):
    path = Path(root) / app / stage / "receipt.json"
    if not path.is_file():
        return None
    receipt = load(path)
    output = Path(receipt["out"]).resolve()
    if (receipt.get("seed") != 42 or receipt.get("app") != app or receipt.get("stage") != stage
            or output.parent != path.parent.resolve()):
        raise ValueError("Unexpected stage receipt identity")
    for name, checksum in receipt["outputs"].items():
        file = output / name
        if not file.is_file() or sha(file) != checksum:
            raise ValueError("Frozen result does not match its completed receipt")
    return output


class Integrator:
    def __init__(self, args):
        self.args = args
        self.paper = args.paper_dir.resolve()
        self.out = args.out_dir.resolve()
        self.fields = {}
        self.sources = {}
        self.applications = {}
        self.findings = {}
        self.derived = {}

    def capture(self, path, app="shared"):
        path = Path(path).resolve()
        checksum = sha(path)
        relative = Path("backfill/evidence") / app / (checksum[:12] + "_" + path.name)
        target = self.paper / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        if sha(target) != checksum:
            raise ValueError("Copied evidence checksum mismatch")
        self.sources[str(relative)] = {"original": str(path), "sha256": checksum}
        return str(relative)

    def bind(self, key, value, source, denominator=None, note="", tex=False):
        if value is None:
            return
        if isinstance(source, (str, Path)):
            source = [source]
        evidence = [self.capture(path, key.split(".", 1)[0]) for path in source]
        self.fields[key] = {"value": value, "latex": str(value) if tex else escape(value),
                            "sources": evidence, "denominator": denominator, "note": note}

    def derived_file(self, name, content):
        path = self.out / name
        path.write_text(json.dumps(content, sort_keys=True, indent=2, allow_nan=False) + "\n")
        self.derived[name] = str(path)
        return path

    def packet_statistics(self, release, report, names):
        # Import only after the CLI establishes its address-space/thread budget.
        import torch
        from dyrca_rca.interaction.observation import load_observation
        from dyrca_rca.interaction.temporal import temporal_features
        torch.set_num_threads(1)
        config = load(release / "split.json")
        observation_source = PROJECT / "src/dyrca_rca/interaction/observation.py"
        if config["source_hashes"]["interaction/observation.py"] != sha(observation_source):
            raise ValueError("Observation reader changed after the frozen release")
        receipts = {load(path)["sha256"]: path.with_suffix(".pt") for path in (release / "cache").glob("*.json")}
        families = {
            "endpoint_change": ("endpoint_pair_observed", ("endpoint_added", "endpoint_removed")),
            "endpoint_empty": ("endpoint_pair_observed", ("endpoint_became_empty",)),
            "pod_change": ("pod_pair_observed", ("pod_added", "pod_removed")),
            "version_change": ("version_pair_observed", ("version_added", "version_removed")),
            "readiness": ("ready_pair_observed", ("ready_drop", "ready_recovery")),
            "backend": (None, ("endpoint_backend_mismatch", "endpoint_backend_unresolved")),
        }
        records = []
        for name in names:
            view = load_observation(Path(config["data_root"]) / name, config["dataset_id"],
                                    config["system"], max_rows=config["max_rows"])
            if view["observation_version"] != report["observation_hashes"][name]:
                raise ValueError("Reconstructed packet does not match frozen observation hash")
            checksum = report["cache_hashes"]["interactive-topology-trace-v3-temporal"][name]
            path = receipts[checksum]
            if sha(path) != checksum:
                raise ValueError("Tensor cache changed after release")
            item = torch.load(path, map_location="cpu", weights_only=True)
            features, _ = temporal_features(view)
            snapshots = sorted([e["data"] for e in view["evidence"] if e["kind"] == "topology"],
                               key=lambda snapshot: snapshot["timestamp"])
            current = snapshots[-1]
            backend_eligible = {s["name"] for s in current["services"]
                                if isinstance(s.get("endpoint_ips"), list) and
                                "pods" in current.get("observed_collections", [])}
            family_counts = {}
            for family, (mask, signals) in families.items():
                eligible = {service for service, values in features.items()
                            if (values[mask] > 0 if mask else service in backend_eligible)}
                active = any(features[service][field] > 0 for service in eligible for field in signals)
                family_counts[family] = {"eligible": len(eligible), "active": active}
            traces = view["modalities"]["traces"]
            records.append({"experiment": name, "configuration": configuration(name),
                            "observation_hash": view["observation_version"], "cache_hash": checksum,
                            "service_nodes": int(item["x"][:, 0].sum()), "graph_nodes": len(item["node_ids"]),
                            "eligible_consumed_rows": traces["eligible_rows"],
                            "scan_truncated": traces["status"] == "truncated", "families": family_counts})
        return {"cases": len(records), "cohort": "screened T1-T4 ranking release, all splits",
                "trace_order": "CSV file order; scan at most 10000 rows before completion-cutoff filtering; no additional start bound",
                "records": records}

    def application(self, app):
        prefix = "ob" if app == "boutique" else "tt"
        root = self.args.experiment_root
        stages = {name: frozen_stage(root, app, name) for name in
                  ("01_index", "02_audit", "03_prepare", "release_s42_t42", "ranking_s42_t42",
                   "ablation_t42", "10_stability", "10_contract_tests", "07_latency", "08_luna", "11_downstream",
                   "16_revision_completion", "17_coverage_completion")}
        index = stages["01_index"]
        if not index:
            self.applications[app] = {"status": "pending_complete_collection"}
            return
        inventory = load(index / "collection_index.json")
        if not inventory.get("formal_ready") or inventory.get("limit") is not None:
            raise ValueError("Partial collection cannot populate a formal result column")
        audit = stages["02_audit"]
        if not audit:
            self.applications[app] = {"status": "pending_audit"}
            return
        audit_path = audit / "collection_audit.jsonl"
        audits = rows(audit_path)
        screened = [row for row in audits if row["verified"]]
        summary_path = audit / "e0_e1_summary.json"
        summary = next(row for row in load(summary_path)["applications"] if row["application"] == app)
        self.bind(prefix + ".intended_n", summary["intended_n"], summary_path)
        for key, source_key in (("exported_n", "scheduler_success_n"), ("screened_n", "verified_n"),
                                ("topology_excluded_n", "unverified_n")):
            self.bind(prefix + "." + key, summary[source_key], summary_path,
                      note="Topology/evidence screening, not independent causal adjudication")
        self.bind(prefix + ".scenario_fault_change", sum(r["scenario_id"] in ("T1","T2","T3","T4") for r in screened),
                  audit_path, note="Screened T1-T4 family count; causal relevance not independently adjudicated")
        self.bind(prefix + ".scenario_fault_nochange", sum(r["scenario_flags"]["no_major_topology_change"] for r in screened),
                  audit_path, note="No observed service endpoint/pod/version change; source-defined proxy")
        self.bind(prefix + ".scenario_fault_benign", sum(r["scenario_flags"]["fault_plus_healthy_change"] for r in screened),
                  audit_path, note="Co-occurring-change proxy; decoy health has not been independently adjudicated")
        self.bind(prefix + ".scenario_multi_change", sum(r["scenario_flags"]["multiple_changed_services"] and not r["scenario_flags"]["benign_dynamic_change"] for r in screened),
                  audit_path, note="More than one observed changed service; categories may overlap")
        prepared, release, ranking = (stages[name] for name in ("03_prepare", "release_s42_t42", "ranking_s42_t42"))
        if not (prepared and release and ranking):
            self.applications[app] = {"status": "pending_ranking"}
            return
        split_path = release / "split.json"
        split = load(split_path)
        for part, key in (("train","train_n_groups"), ("val","val_n_groups"), ("test","test_n_groups")):
            names = split["split"][part]
            self.bind(prefix + "." + key, f"{len(names)} / {len({configuration(n) for n in names})}", split_path,
                      note="Frozen grouping key is experiment-name prefix before the last -run")
        if not split["split"]["regression"]:
            self.bind(prefix + ".regression_overlap", "Not retained", split_path)
        self.bind(prefix + ".window_rule", "Pre/during snapshots; during cutoff; completed spans; no extra trace start bound; arrival times unavailable",
                  prepared / "protocol.json")
        self.bind(prefix + ".trace_budget", f"{split['max_rows']:,} scanned rows; CSV order, then window/completion filter", split_path)
        self.bind(prefix + ".runtime_manifest", "CPU, one thread; PyTorch " + load(release / "runtime.json")["torch"],
                  release / "runtime.json")
        candidate = load(prepared / "candidate_release.json")
        names = sorted(name for part in ("train","val","test") for name in split["split"][part])
        self.bind(prefix + ".target_configuration_coverage",
                  f"{len(set(candidate['target_services'].values()))} targets / {len({configuration(n) for n in names})} frozen keys",
                  prepared / "candidate_release.json")
        release_report = load(release / "report.json")
        packet = self.packet_statistics(release, release_report, names)
        packet_path = self.derived_file(app + "_packet_statistics.json", packet)
        values = packet["records"]
        self.bind(prefix + ".services_nodes", f"{statistics.median(v['service_nodes'] for v in values):g} / {statistics.median(v['graph_nodes'] for v in values):g}",
                  packet_path, len(values), "Builder service nodes / total graph nodes in the screened T1-T4 release")
        consumed = sorted(v["eligible_consumed_rows"] for v in values)
        q1, q3 = statistics.quantiles(consumed, n=4, method="inclusive")[0::2]
        self.bind(prefix + ".eligible_rows", f"{statistics.median(consumed):g} / [{q1:g}, {q3:g}]", packet_path, len(values),
                  "Consumed eligible rows after bounded scan, not all eligible rows before truncation")
        self.bind(prefix + ".truncation_rate", number(sum(v["scan_truncated"] for v in values) / len(values), True),
                  packet_path, len(values), "Proportion with a truncated CSV scan")
        for family in values[0]["families"]:
            eligible = sum(v["families"][family]["eligible"] for v in values)
            active = sum(v["families"][family]["active"] for v in values)
            self.bind(prefix + ".active_" + family, f"{eligible} / {active}", packet_path, len(values),
                      "Eligible service-object comparisons / unique active cases, screened T1-T4 release")
        source = ranking / "report.json"
        report = load(source)
        if {r["method"] for r in report["main_comparison"]} != set(METHOD_KEYS):
            raise ValueError("Main comparison does not contain exactly nine required methods")
        denominators = {r["test_n"] for r in report["main_comparison"]}
        if denominators != {len(split["split"]["test"])}:
            raise ValueError("Ranking denominator differs from frozen split")
        for row in report["main_comparison"]:
            for metric, suffix in (("hit1","h1"), ("hit3","h3"), ("mrr","mrr")):
                self.bind(f"{prefix}.rank_{METHOD_KEYS[row['method']]}_{suffix}", number(row["test_" + metric], metric != "mrr", 4 if metric == "mrr" else 2),
                          source, row["test_n"], "New within-application main split, not archival 45-case scores")
        self.bind(prefix + ".release_manifest", "split " + sha(split_path)[:12] + "; predictions " + report["prediction_hash"][:12],
                  [split_path, source])
        self.bind("audit.code_binding", "Recorded source hashes; checkpoint " + sha(release / "temporal_v3.pt")[:12] + "; temporal v3: 43 node fields, 8 edge fields; Git commit unavailable",
                  [release / "report.json", release / "split.json"], note="This binding does not certify unexecuted operation audits")
        self.shortcuts(prefix, stages, audits)
        if stages["ablation_t42"]:
            source = stages["ablation_t42"] / "ablation_summary.json"
            identifiers = {"current_state_v2":"v2", "full_temporal_v3":"v3", "without_change_values":"nochange",
                           "without_temporal_masks":"nomask", "constant_positive_edge_weights":"unitweight"}
            for row in load(source)["variants"]:
                for metric, suffix in (("hit1","h1"), ("mrr","mrr")):
                    self.bind(f"{prefix}.ablation_{identifiers[row['variant']]}_{suffix}", number(row[metric], metric=="hit1", 2 if metric=="hit1" else 4),
                              source, row["n"], "Retrained, seed 42; same main test")
        if stages["10_stability"]:
            self.stability(prefix, stages["10_stability"], ranking)
        if stages["07_latency"]:
            source = stages["07_latency"] / "latency_summary.json"
            for operation, key in (("cold_evidence_graph_ms","cold"), ("cached_scorer_inference_ms","infer")):
                for quantile in ("p50","p95"):
                    summary = load(source)["summaries"][operation]
                    self.bind(f"{prefix}.latency_{key}_{quantile}", number(summary[quantile + "_ms"], digits=3), source,
                              summary["n"], "Warm OS/file cache after five warmups; uncached graph reconstruction or cached tensor inference; CPU")
        if stages["10_contract_tests"]:
            source = stages["10_contract_tests"] / "state_tests.xml"
            suites = list(ET.parse(source).getroot().iter("testsuite"))
            total = sum(int(s.get("tests",0)) for s in suites)
            excluded = sum(int(s.get(k,0)) for s in suites for k in ("failures","errors","skipped"))
            self.bind("revision.tests_total", total, source, note="Two fixture/contract test files; not real incident replays")
            self.bind("revision.tests_passed", total-excluded, source, total)
        if stages["08_luna"]:
            self.agent(app, stages["08_luna"], prepared, release)
        self.applications.setdefault(app, {}).update(status="available_results_integrated",
            main_test_n=len(split["split"]["test"]), screened_n=len(screened), ranking_population_n=len(names),
            independent_causal_verification=False)
        if stages['16_revision_completion']:
            self.supplement(app, stages['16_revision_completion'], stages['17_coverage_completion'])

    def supplement(self, app, stage, coverage_stage):
        from jsonschema import validate
        prefix = 'ob' if app == 'boutique' else 'tt'
        complete = load(stage/'completion_summary.json')
        validate(complete,{'type':'object','required':['status','system','seed','limit','test_cases','audit_cases','paired_conditions','paired_strata','checkpoint_unchanged','artifact_hashes'],
            'properties':{'schema_version':{'enum':['r2-completion-v2']},'status':{'enum':['completed']},'system':{'enum':[app]},'seed':{'enum':[42]},'limit':{'type':'null'},
                          'test_cases':{'type':'integer','minimum':1},'audit_cases':{'type':'integer','minimum':1},'paired_conditions':{'enum':[6]},'paired_strata':{'enum':[2]},
                          'checkpoint_unchanged':{'enum':[True]},'artifact_hashes':{'type':'object','minProperties':1,'additionalProperties':{'type':'string','pattern':'^[0-9a-f]{64}$'}}}})
        if complete['status'] != 'completed' or complete['limit'] is not None or complete['system'] != app:
            raise ValueError('Bounded supplemental runs cannot populate a formal result column')
        for name, checksum in complete['artifact_hashes'].items():
            if sha(stage/name) != checksum:
                raise ValueError('Supplemental artifact no longer matches its completion hash')
        for name in ('protocol.json','source_integrity.json'):
            if (stage/name).is_file():self.capture(stage/name,prefix)
        protocol=load(stage/'protocol.json')
        runner=stage/'runner_source.py'
        if not runner.is_file() or sha(runner)!=protocol['runner_sha256']:
            raise ValueError('Supplemental execution source is not bound to its recorded runner')
        integrity=load(stage/'source_integrity.json')
        if integrity['status']!='verified' or integrity['target_files']!=complete['test_cases'] or integrity['consumed_files_sha256_before']!=integrity['consumed_files_sha256_after']:
            raise ValueError('Supplemental input/target integrity proof is incomplete')
        self.capture(runner,prefix)
        self.capture(stage.parent/'receipt.json',prefix)
        source=stage/'source_audit.json'; audit=load(source)
        n=complete['test_cases']
        self.bind(prefix+'.fidelity_audit_n_objects', f"{audit['cases']} / {audit['objects']}",source,audit['cases'],
                  'Independent parser of the same collector snapshots: source consistency, not external truth or causal fidelity')
        self.bind(prefix+'.fidelity_unverifiable_n',audit['objects_without_reference'],source,
                  note='Unknown fields plus multiversion groups with no eligible numeric route reference')
        self.bind(prefix+'.fidelity_endpoint_pr',f"{audit['endpoint_precision']:.2f} / {audit['endpoint_recall']:.2f}",source,
                  {'cases':audit['cases'],'objects':sum(r['endpoint_objects'] for r in audit['records'])},'Case-macro of per-case endpoint-object macro precision/recall')
        self.bind(prefix+'.fidelity_version_agreement',number(audit['version_agreement']),source,
                  {'cases':audit['cases'],'objects':sum(r['version_objects'] for r in audit['records'])},
                  'Case-macro of per-case raw Pod service/version field agreement; not proof of complete logical version/routing graphs')
        self.bind(prefix+'.fidelity_status_accuracy',number(audit['availability_correct']/audit['availability_objects'],True),source,audit['availability_objects'])
        self.bind(prefix+'.fidelity_provenance_accuracy',f"{audit['source_status_correct']} / {audit['source_status_objects']}",source,audit['source_status_objects'],
                  'One timestamp/content-address check per topology snapshot; not object-level provenance or a human status rubric')
        self.bind(prefix+'.fidelity_fidelity',number(audit['fidelity']),source)
        for key,field in [('natural_defect_n','natural_discrepancy_cases'),('packet_discoverable_n','packet_discoverable_cases'),('discoverable_n','endpoint_form_actionable_cases')]:
            self.bind(prefix+'.fidelity_'+key,f"{audit[field]} / {audit['cases']}",source,audit['cases'],
                      'Local snapshot-field discrepancies only; no GUI participant observation')
        self.bind(prefix+'.fidelity_seeded_noop_n',f'{n} / {n}',stage/'completion_summary.json',{'seeded':n,'unperturbed':n})
        self.bind(prefix+'.fidelity_route_error',f"{audit['supported_route_groups']} / n/a",source,0,
                  'No eligible independent numeric route weights; L1 error has an empty denominator')
        source=stage/'revision_summary.json'; revision=load(source)
        for condition,row in revision['summaries']['seeded'].items():
            for key in ('n','accepted','fidelity','repair','damage','h1','mrr','better','same','worse'):
                value=row[key]
                self.bind(f'{prefix}.feedback_{condition}_{key}', 'n/a' if value is None else
                          value if key in ('n','better','same','worse') else number(value,digits=4 if key=='mrr' else 2),source,row['n'],
                          'Primary seeded stratum; two phase endpoint objects at one label-blind service; collector-source oracle content; separate unperturbed results in artifact')
        scored=load(stage/'revision_scored.json')
        groups={}
        for label,auxiliary in (('remaining_services',False),('named_auxiliary_services',True)):
            rows=[r for r in scored if r['stratum']=='seeded' and
                  (r['nominated_service'] in ('locust-master','frontend-external'))==auxiliary]
            groups[label]={}
            for condition in ('none','correct'):
                condition_rows=[r for r in rows if r['condition']==condition]
                groups[label][condition]={'n':len(condition_rows),
                    'h1':100*statistics.mean(r['hit1'] for r in condition_rows) if condition_rows else None,
                    'mrr':statistics.mean(r['mrr'] for r in condition_rows) if condition_rows else None,
                    'fidelity':statistics.mean(r['content']['fidelity'] for r in condition_rows) if condition_rows else None}
        subgroup_source=self.derived_file(prefix+'_revision_subgroups.json',{
            'scope':'Post-hoc fixed-name subgroup description, not an independent incident cohort',
            'auxiliary_names':['locust-master','frontend-external'],'groups':groups,
            'scored_source_sha256':sha(stage/'revision_scored.json')})
        remaining,auxiliary=groups['remaining_services'],groups['named_auxiliary_services']
        parts=[]
        for title,group in (('other services',remaining),('named auxiliary services',auxiliary)):
            before,after=group['none'],group['correct']
            parts.append(f"{title} n={before['n']}, "+(f"H@1 {before['h1']:.2f}/{after['h1']:.2f}, source consistency {before['fidelity']:.2f}/{after['fidelity']:.2f}" if before['n'] else 'n/a'))
        self.bind(prefix+'.feedback_subgroups','; '.join(parts),[subgroup_source,stage/'revision_summary.json'],
                  {key:value['none']['n'] for key,value in groups.items()},'No-repair/full-repair comparison; exploratory same-incident subgroup description')
        source=stage/'operation_latency.json'; latency=load(source)
        for operation,row in latency['summaries'].items():
            for quantile in ('p50','p95'):
                self.bind(f'{prefix}.latency_{operation}_{quantile}',number(row[quantile+'_ms'],digits=3),source,row['n'],latency['scope'])
        source=stage/'weight_sensitivity.json'; weights=load(source)['settings']
        self.bind(prefix+'.stability_weight_sensitivity','change 0/.10/.30; H@1 '+
                  f"[{min(r['h1'] for r in weights.values()):.2f}, {max(r['h1'] for r in weights.values()):.2f}]; MRR "+
                  f"[{min(r['mrr'] for r in weights.values()):.4f}, {max(r['mrr'] for r in weights.values()):.4f}]",source,n,
                  'Three fixed nominal inference-time settings; available edge channels renormalize effective shares, recorded in evidence; no selection or retuning')
        source=stage/'deployment_inventory.json'; inventory=load(source)
        manifest=self.derived_file(prefix+'_image_manifest.json',{'images':inventory['image_strings'],
                                                               'source_hashes':inventory['post_image_source_hashes'],'scope':inventory['image_inventory_scope']})
        self.bind(prefix+'.application_revision',f"{len(inventory['image_strings'])} image strings; manifest {sha(manifest)[:12]}; Git revision unavailable",
                  [source,manifest],note='Post-export runtime inventory only; no use in model inputs or source-reference fidelity')
        self.bind(prefix+'.version_coverage',number(inventory['deployment_labels_resolved']/inventory['deployment_labels_total'],True),source,inventory['deployment_labels_total'])
        rate=inventory['frontend_inbound_rps_median']
        self.bind(prefix+'.workload_rate',f"frontend median {rate:.2f} req/s; {inventory['request_rate_cases']} eligible cases" if rate is not None else
                  f"n/a ({inventory['request_rate_cases']} eligible frontend-rate cases)",source,inventory['request_rate_cases'],inventory['request_rate_scope'])
        self.bind(prefix+'.adapter_effort','Shared frozen adapters; developer-hours not recorded',stage/'protocol.json',
                  note='No fabricated historical adaptation-time estimate')
        source=stage/'edit_audit.json'; edits=load(source)
        primary=[r for r in edits['records'] if r['stratum']=='seeded']
        self.bind('audit.'+prefix+'_redundant',f"{len(primary)} / {sum(r['redundant_input_changed'] for r in primary)} / {sum(r['redundant_target_rank_changed'] for r in primary)}",source,len(primary))
        if app=='boutique':
            self.bind('audit.duplicate_event',f"{edits['duplicate_equal']} / {edits['duplicate_attempts']} retained; state API retry",source,edits['duplicate_attempts'])
            self.bind('audit.endpoint_noop',f"{sum(r['redundant_input_changed'] for r in primary)} / {len(primary)} changed inputs",source,len(primary))
            self.bind('audit.plan_noop',f"{sum(r['equivalent_plan_input_changed'] for r in primary)} / {len(primary)} changed inputs",source,len(primary))
            masked=edits['masked']
            self.bind('audit.tag_masked',f"OB none/correct H@1 {masked['none']['h1']:.2f}/{masked['correct']['h1']:.2f}; MRR {masked['none']['mrr']:.4f}/{masked['correct']['mrr']:.4f}",source,n,
                      'Complete operation-derived-channel masking; sensitivity, not a retrained encoding')
            self.bind('audit.attribution','Same-value endpoint and equivalent-plan edits preserve tensors; masking accounts for final weights and node change/update summaries',source)
            agent_source=stage/'agent_history_audit.json'; agent=load(agent_source)
            self.bind('human.edit_audit_binding',f"OB {agent['cases']} logged API tasks; collector-reference assessment; post-hoc; workbench {agent['workbench_source_hash'][:12]}",agent_source,agent['cases'])
            self.bind('human.preview_unaccepted',f"{agent['unaccepted_previews']} / {agent['preview_commands']}",agent_source,agent['preview_commands'])
            moved=agent['edited_service_movement']; supported=[r for r in moved if r['source_supported']]
            self.bind('human.edited_rank_by_content',f"{len(supported)} supported edits; {sum(r['before_rank']!=r['after_rank'] for r in supported)} changed service ranks; unsupported edits {len(moved)-len(supported)}",agent_source,len(moved))
            for stratum,row in agent['rescue'].items():
                self.bind('human.rescue_'+stratum,f"n/a (eligible={row['eligible']}; stratum={row['all_tasks']})" if not row['eligible'] else
                          f"{row['correct']} / {row['eligible']} (stratum={row['all_tasks']})",agent_source,row['eligible'],
                          'Executed agent service-level rescue; not independently scored human joint diagnosis')
            source=stage/'t4_walkthrough.json'; walk=load(source)
            self.bind('walkthrough.case_sources',walk['case']+'; first held-out T4; unperturbed; '+', '.join(x[:22] for x in walk['source_ids']),source)
            self.bind('walkthrough.defect_source','Observation-only unsplit; Pod version labels present; numeric route references absent; retain unknown',source)
            self.bind('walkthrough.proposal','n/a: no source-supported numeric full plan proposed',source,0)
            self.bind('walkthrough.preview','n/a: no proposal; no relation/rank changes',source,0)
            self.bind('walkthrough.decision',f"No commit; revision {walk['state_revision_before']} to {walk['state_revision_after']}; policy {walk['agent_source']}",source)
            self.bind('walkthrough.fidelity','Pod membership unchanged; numeric route fidelity n/a (n=0)',source,0)
            self.bind('walkthrough.rank',f"{walk['target_rank']} / {walk['target_rank']}",source,1)
            self.bind('walkthrough.failure','No execution failure; unsupported weights withheld; state retained',source)
            self.bind('t4.packet_binding',f"OB {audit['t4_cases']} observation-only unsplit cases; collector Pod version labels; no eligible numeric route reference",stage/'source_audit.json',audit['t4_cases'])
            self.bind('t4.repair_restraint',f"Numeric repair {audit['t4_numeric_repair_cases']}; source-required restraint {audit['t4_restraint_cases']}",stage/'source_audit.json',audit['t4_cases'],
                      'Source-availability classification, not observed human restraint')
        if coverage_stage:
            self.capture(coverage_stage.parent/'receipt.json',prefix)
            self.capture(coverage_stage/'control_source_records.json',prefix)
            source=coverage_stage/'coverage_summary.json'; coverage=load(source)
            validate(coverage,{'type':'object','required':['status','system','seed','limit','runner_sha256','control_cases','matched_fault_cases','groups','causal_reference_coverage'],
                'properties':{'schema_version':{'enum':['r2-coverage-v2']},'status':{'enum':['completed']},'system':{'enum':[app]},'seed':{'enum':[42]},'limit':{'type':'null'},
                              'runner_sha256':{'type':'string','pattern':'^[0-9a-f]{64}$'},'control_cases':{'type':'integer','minimum':0},'matched_fault_cases':{'type':'integer','minimum':0},
                              'groups':{'type':'object','required':['fault_benign','multi_change','samechange','benign']},'causal_reference_coverage':{'type':'object'}}})
            runner=coverage_stage/'runner_source.py'
            if not runner.is_file() or sha(runner)!=coverage['runner_sha256']:
                raise ValueError('Coverage execution source is not bound to its archived runner')
            self.capture(runner,prefix)
            if coverage['limit'] is not None or coverage['status']!='completed':raise ValueError('Partial control coverage not publishable')
            causal=coverage['causal_reference_coverage']
            self.bind(prefix+'.verified_n','n/a (no external causal reference)',source,causal['independent_causal_references'])
            self.bind(prefix+'.unrealized_n',causal['unrealized_topology_changes'],source,note='Unrealized intended topology change, not causal diagnosis')
            self.bind(prefix+'.unresolved_n',f"{causal['causal_unresolved_screened_runs']} screened runs lack causal adjudication",source)
            self.bind(prefix+'.other_excluded_n',causal['other_topology_screen_exclusions'],source,note='Other topology-screen exclusions only')
            self.bind(prefix+'.verification_disagreement','n/a (0 independent adjudicators)',source,0)
            self.bind(prefix+'.scenario_samechange',coverage['matched_fault_cases'],source,
                      note='Exploratory exact aggregate observed-change match; benign matches reused; not an independently adjudicated matched causal cohort')
            for subset in ('fault_benign','multi_change','samechange'):
                group=coverage['groups'][subset]
                # One explicitly named protocol-decoy rate aggregates the three methods.
                values=[]
                for method in ('change','node','dy'):
                    row=group[method]
                    if subset=='samechange':
                        self.bind(f'{prefix}.shortcut_{subset}_{method}_h1',number(row['hit1'],True) if row['hit1'] is not None else 'n/a (n=0)',source,row['n'],coverage['scope'])
                    values.append(number(row['decoy_nomination_rate'],True) if row['decoy_nomination_rate'] is not None else 'n/a')
                count=next(iter(group.values()))['n']
                self.bind(f'{prefix}.shortcut_{subset}_decoy_rate','/'.join(values)+f' (n={count})',source,count,
                          'Change/Node/DY protocol-decoy nomination percentages; no independent physical-health claim')
            self.bind(prefix+'.shortcut_fault_nochange_decoy_rate','n/a (n=0)',source,0)
        self.applications[app]['supplemental']={'test_cases':n,'source_audit_cases':audit['cases'],
            'revision':revision['summaries']['seeded'],'source_consistency_only':True,'stage':str(stage),
            'artifact_schema':complete.get('schema_version','r2-completion-v1-legacy'),'scorer_planner':'rule','t4_agent':protocol.get('t4_agent',protocol['planner']),
            'checkpoint_unchanged':complete['checkpoint_unchanged'],
            'source_integrity':load(stage/'source_integrity.json') if (stage/'source_integrity.json').is_file() else None}

    def shortcuts(self, prefix, stages, audits):
        path = stages["11_downstream"]
        if not path:
            return
        source = path / "shortcut_summary.json"
        report = load(source)
        if any("scored_n" not in row for row in report["rows"]):
            # Legacy n included benign controls whose ranking outcome is null.
            # Reconstruct from the immutable scored predictions, not n or labels.
            from dyrca_rca.experiments.run_tosem_downstream import _shortcut_summary
            ranking_source = stages["ranking_s42_t42"] / "results.jsonl"
            report = _shortcut_summary(rows(ranking_source), {r["case_id"]: r for r in audits})
            report["provenance"] = {"ranking_path": str(ranking_source), "ranking_sha256": sha(ranking_source),
                                    "legacy_summary_sha256": sha(source)}
            source = self.derived_file(prefix + "_shortcut_scored_denominators.json", report)
        methods = {"Change-only":"change", "Node-only":"node", "DY-RCA":"dy"}
        for row in report["rows"]:
            if row["subset"] not in ("fault_plus_benign", "multiple_changed_services") or row["method"] not in methods:
                continue
            category = "fault_benign" if row["subset"] == "fault_plus_benign" else "multi_change"
            value = number(row["hit1"], True) if row["hit1"] is not None else "n/a (n=0)"
            self.bind(f"{prefix}.shortcut_{category}_{methods[row['method']]}_h1", value,
                      source, row["scored_n"], "Exploratory observed-change proxy; total rows=%s; scored denominator excludes null ranking outcomes; no independent healthy-decoy adjudication" % row["n"])
        if not any(r["verified"] and r["scenario_flags"]["no_major_topology_change"] for r in audits):
            for method in methods.values():
                self.bind(f"{prefix}.shortcut_fault_nochange_{method}_h1", "n/a (n=0)", stages["02_audit"] / "collection_audit.jsonl", 0)

    def stability(self, prefix, stage, ranking):
        source = stage / "stability_summary.json"
        report = load(source)
        self.bind(prefix + ".seeds_splits", "Training: 42, 123, 456, 789, 2024; split: 42, 137, 2026", source)
        self.bind(prefix + ".bootstrap_ci", "2,000 configuration draws; 95% percentile interval", source)
        for kind, key in (("training_seed","train_variation"), ("split_seed","split_variation")):
            row = next(r for r in report["summary"] if r["kind"]==kind and r["method"]=="DY-RCA")
            self.bind(prefix + ".stability_" + key, f"{row['runs']} / {number(row['hit1_mean'],True)} ({number(row['hit1_sd'],True)})", source,
                      row["runs"], "Configuration-macro Hit@1; repeats are not independent incidents")
        paired = report["paired"]
        self.bind(prefix + ".stability_baseline_delta_ci",
                  f"{paired['delta_mrr']:.4f} / [{paired['ci_low']:.4f}, {paired['ci_high']:.4f}]", source,
                  paired["configuration_groups"], "Configuration-macro MRR difference")
        self.bind(prefix + ".stability_comparator_rule", report["comparator"] + "; validation macro MRR, then Hit@1 and fixed order", source)
        from dyrca_rca.experiments.measure_tosem_stability import group_means
        means = group_means(rows(ranking / "results.jsonl"), "DY-RCA", "test", "hit1")
        interval = cluster_interval({key:[value] for key,value in means.items()}, self.args.seed)
        derived = self.derived_file(prefix + "_macro_hit1_interval.json", interval)
        self.bind(prefix + ".stability_macro_ci", f"{number(interval['estimate'],True)} / [{number(interval['low'],True)}, {number(interval['high'],True)}]", derived,
                  interval["groups"], "Post-hoc deterministic group bootstrap of existing frozen predictions")

    def agent(self, app, stage, prepared, release):
        agent = stage / "agent"
        source = agent / "summary.json"
        summary = load(source)
        protocol = load(agent / "protocol.json")
        if protocol["checkpoint_sha256"] != sha(release / "temporal_v3.pt"):
            raise ValueError("Agent simulation used a different checkpoint")
        scored = rows(agent / "scored.jsonl")
        manifest = rows(prepared / "collection/observation_manifest.jsonl")
        names = {r["case_id"]:r["experiment"] for r in manifest}
        test = set(load(release / "split.json")["split"]["test"])
        if any(names[r["case_id"]] not in test for r in scored):
            raise ValueError("Agent results contain non-test cases")
        self.capture(agent / "protocol.json", "human")
        by_case = defaultdict(dict)
        for row in scored:
            by_case[row["case_id"]][row["condition"]] = row
        if any(set(case) != {"H0","H1","H2"} for case in by_case.values()):
            raise ValueError("Incomplete paired agent conditions")
        prefix = "ob" if app == "boutique" else "tt"
        process = {"cases":len(by_case), "application":app, "conditions":{},
                   "seeded_cases":summary["seeded_defect_case_count"], "reference_access":summary["oracle_access_case_count"],
                   "repairs_proposed":summary["repair_proposed"], "repairs_accepted":summary["repair_accepted"],
                   "seeded_objects_corrected":summary["seeded_defects_corrected"], "guard_rejections":summary["repair_guard_rejections"],
                   "source_counts":summary["agent_source_counts"], "classification":"simulated-agent service-only, not human joint diagnosis"}
        paired = defaultdict(list)
        for identifier, case in sorted(by_case.items()):
            paired[configuration(names[identifier])].append(int(case["H2"]["service_correct"])-int(case["H1"]["service_correct"]))
        process["paired_service_difference"] = cluster_interval(paired, self.args.seed)
        for condition in ("H0","H1","H2"):
            selected = [case[condition] for case in by_case.values()]
            correct = sum(bool(r["service_correct"]) for r in selected)
            initial_wrong = [r for r in selected if r["initial_model_top1"] != r["root_service"]]
            initial_right = [r for r in selected if r["initial_model_top1"] == r["root_service"]]
            process["conditions"][condition] = {"n":len(selected), "correct":correct, "source_counts":dict(Counter(r["agent_source"] for r in selected)),
                "rescue_correct":sum(bool(r["service_correct"]) for r in initial_wrong), "rescue_n":len(initial_wrong),
                "damage_incorrect":sum(not r["service_correct"] for r in initial_right), "damage_n":len(initial_right)}
        derived = self.derived_file(prefix + "_agent_service_statistics.json", process)
        self.applications.setdefault(app,{})["agent_statistics"] = process
        # Stable legacy keys remain in the registry, but the active RQ4 text and
        # row labels explicitly define these as simulated service outcomes.
        for condition,row in process["conditions"].items():
            key = condition.lower()
            self.bind(f"{prefix}.agent_{key}_assigned", f"{row['n']} / {row['n']}", derived, row["n"])
            self.bind(f"{prefix}.agent_{key}_correct", f"{row['correct']} / {number(row['correct']/row['n'],True)}", derived, row["n"])
            if condition != "H0":
                self.bind(f"{prefix}.agent_{key}_rescue", f"{row['rescue_correct']} / {row['rescue_n']}" if row["rescue_n"] else "n/a (n=0)", derived, row["rescue_n"])
                self.bind(f"{prefix}.agent_{key}_damage", f"{row['damage_incorrect']} / {row['damage_n']}", derived, row["damage_n"])
        for key, value in (("cases",process["cases"]), ("seeded",process["seeded_cases"]),
                           ("reference",process["reference_access"]), ("accepted",process["repairs_accepted"]),
                           ("corrected",process["seeded_objects_corrected"]), ("fallback",process["source_counts"].get("rule_fallback",0))):
            self.bind(f"{prefix}.agent_{key}", value, derived)
        difference=process["paired_service_difference"]
        self.bind(f"{prefix}.agent_service_rd", f"{number(difference['estimate'],True)} [{number(difference['low'],True)}, {number(difference['high'],True)}]",
                  derived, process["cases"], "Exploratory configuration-cluster bootstrap; service correctness only; no participant model")
        # Export only analysis columns, excluding free-text API errors/responses.
        selected_columns = ("case_id", "condition", "agent_source", "root_service", "initial_model_top1",
                            "service_correct", "repair_proposed", "repair_accepted", "topology_object_correct_after")
        analysis_records = {"source_sha256":sha(agent / "scored.jsonl"),
                            "records":[dict({key:r[key] for key in selected_columns}, configuration=configuration(names[r["case_id"]])) for r in scored]}
        analysis_path = self.derived_file(prefix + "_agent_analysis_records.json", analysis_records)
        self.capture(analysis_path, "human")
        self.bind(f"{prefix}.agent_sources", f"{process['source_counts'].get('llm',0)} Luna / {process['source_counts'].get('rule_fallback',0)} fallback", source, len(scored))
        self.bind(f"{prefix}.agent_repair_flow", f"{process['repairs_proposed']} / {process['repairs_accepted']} / {process['seeded_objects_corrected']}", derived)
        self.bind(f"{prefix}.agent_guard_rejected", process["guard_rejections"], derived)
        self.bind(f"{prefix}.agent_repair_coverage", f"{process['seeded_objects_corrected']} / {process['seeded_cases']}", derived, process["seeded_cases"])
        self.bind(f"{prefix}.agent_initial_correct", sum(case["H1"]["initial_model_top1"] == case["H1"]["root_service"] for case in by_case.values()), derived, process["cases"])

    def finding_bindings(self):
        available = [app for app, status in self.applications.items() if status.get("status")=="available_results_integrated"]
        if not available:
            return {}
        labels = {"boutique":"Online Boutique", "trainticket":"Train Ticket"}
        ranking = []
        agent = []
        for app in available:
            prefix = "ob" if app=="boutique" else "tt"
            n = self.applications[app]["main_test_n"]
            ranking.append(labels[app]+f" ({n} test incidents: DY-RCA "+str(self.fields[prefix+".rank_dyrca_h1"]["value"])+
                           "%, Change-only "+str(self.fields[prefix+".rank_change_h1"]["value"])+"% Hit@1)")
            if "agent_statistics" in self.applications[app]:
                stats = self.applications[app]["agent_statistics"]
                h1,h2 = (stats["conditions"][condition]["correct"] for condition in ("H1","H2"))
                interval = stats["paired_service_difference"]
                agent.append(f"{labels[app]} ({h1}/{stats['cases']} read-only versus {h2}/{stats['cases']} revisable service diagnoses; "
                             f"difference {100*interval['estimate']:.2f} percentage points, exploratory 95% configuration-cluster interval "
                             f"[{100*interval['low']:.2f}, {100*interval['high']:.2f}])")
        ranking_text = "; ".join(ranking)
        agent_text = "; ".join(agent)
        zero_intervals = [labels[app] for app in available
                          if "agent_statistics" in self.applications[app]
                          and self.applications[app]["agent_statistics"]["paired_service_difference"]["low"] <= 0
                          <= self.applications[app]["agent_statistics"]["paired_service_difference"]["high"]]
        uncertainty = ("The interval includes zero for "+", ".join(zero_intervals)+". ") if zero_intervals else ""
        values = {
            "abstract_revision":"The completed ranking results include "+ranking_text+"; independent natural-fidelity and six-condition revision comparisons remain unmeasured.",
            "contribution_revision":"The completed contract tests support execution checks; they do not yet establish an independently verified content-repair benefit.",
            "rq1":"Feature activation is measured from frozen packets, but independent value/status fidelity and natural repair-opportunity incidence remain unmeasured.",
            "rq2":"The current results are "+ranking_text+". A ceiling shared with Change-only and Node-only does not establish an incremental graph-update benefit; the separate collector-labelled controls retain their exploratory scope.",
            "rq3":"The current run records "+str(self.fields.get("revision.tests_passed",{}).get("value","pending"))+" passed fixture/contract tests. A controlled-agent endpoint repair is not a substitute for the six-condition fidelity/ranking experiment.",
            "conclusion_revision":"The completed substrate comparisons retain a change-cue ceiling, while independent fidelity and the full paired revision study remain requirements for a content-specific benefit claim.",
        }
        if agent:
            values.update(abstract_human="The simulated-agent results are "+agent_text+". These measure service localization without human participants or mechanism/evidence adjudication.",
                contribution_human="The current evidence is a GPT-5.6-Luna simulation with explicit deterministic fallback; a participant study and its joint-diagnosis outcomes have not been conducted.",
                rq4="The simulated-agent results are "+agent_text+". "+uncertainty+"Service correctness does not establish an improvement in evidence-supported human diagnosis.",
                discussion_joint="Current agent results test the implemented simulation package. They provide no estimate of human workload, comprehension, or time-bounded joint diagnosis.",
                conclusion_human=uncertainty+"The simulated H2-minus-H1 contrast measures service correctness; real-operator benefit is an untested claim.")
        supplemented=[app for app in available if 'supplemental' in self.applications[app]]
        if supplemented:
            details=[]
            for app in supplemented:
                study=self.applications[app]['supplemental']; result=study['revision']
                details.append(labels[app]+f" ({study['test_cases']} held-out incidents: source-reference fidelity "
                               f"{result['none']['fidelity']:.2f}% without repair and {result['correct']['fidelity']:.2f}% with full repair; "
                               f"Hit@1 {result['none']['h1']:.2f}% and {result['correct']['h1']:.2f}%)")
            evidence='; '.join(details)
            values.update(
                abstract_revision='The completed paired source-reference revision study includes '+evidence+'. Independent causal and global physical-fidelity assessment remain outside this measured source-consistency scope.',
                contribution_revision='Four API edit arms, a no-edit control and a forced-recommendation control are compared on frozen held-out incidents. Source-backed object repair is measurable, while a service-ranking improvement is not established.',
                rq1='The independent parser measures consistency with collector snapshots, including source linkage and local-field availability. It does not provide an external reference for causal truth, global topology fidelity, or unsupported numeric route weights.',
                rq3='The revision study with four API edit arms and two controls includes '+evidence+'. Unperturbed controls are separate; no-op edits and tag-masked inputs are audited. These are controlled source-reference interventions, not estimates of SRE behavior.',
                conclusion_revision='Supported API repair restores the seeded local fields while preserving other source-backed objects. The tested ranking remains at a ceiling, so these repairs do not establish a diagnostic-accuracy gain.')
        return {key:escape(value) for key,value in values.items()}

    def finish(self):
        # Snapshot original uploaded files once; the source ZIP is immutable.
        original = self.paper / "backfill/originals"
        original.mkdir(parents=True, exist_ok=True)
        for name in ("pending_values.tex","findings.tex","main.pdf"):
            if (self.paper/name).is_file() and not (original/name).exists():
                shutil.copy2(self.paper/name, original/name)
        locations = defaultdict(list)
        for path in sorted(self.paper.rglob("*.tex")):
            if any(part in ("archive","build","backfill") for part in path.relative_to(self.paper).parts):
                continue
            for line, text in enumerate(path.read_text().splitlines(), 1):
                for key in re.findall(r"\\Result\{([^}]+)\}", text):
                    locations[key].append({"file":str(path.relative_to(self.paper)),"line":line})
        registry = self.paper / "pending_values.tex"
        existing = set(re.findall(r"\\SetResult\{([^}]+)\}", registry.read_text()))
        bindings = {key:"" for key in set(locations)-existing}
        bindings.update({key:row["latex"] for key,row in self.fields.items()})
        registry.write_text(replace_registry(registry.read_text(), bindings))
        declared = set(re.findall(r"\\SetResult\{([^}]+)\}", registry.read_text()))
        pending = set(re.findall(r"\\SetResult\{([^}]+)\}\{\s*\}", registry.read_text()))
        findings = self.finding_bindings()
        finding_file = self.paper / "findings.tex"
        finding_file.write_text(replace_registry(finding_file.read_text(), findings, command="SetFinding"))
        completed = [app for app,status in self.applications.items() if status.get("status")=="available_results_integrated"]
        status_text = r"\paragraph{Measured-result snapshot and scope.}"+"\n"
        status_text += "Completed columns refer to the new four-copy collection, with topology/evidence screening and collector-labelled service targets. Independent causal adjudication and external global-fidelity assessment are not available. "
        supplemental=[app for app in completed if 'supplemental' in self.applications[app]]
        if supplemental:
            status_text += "The completed supplemental columns report raw-source local-field consistency, four paired API edit arms and two controls on frozen test incidents. "
        else:
            status_text += "The six-condition API revision experiment remains pending. "
        status_text += "Available application results: "+", ".join("Online Boutique" if app=="boutique" else "Train Ticket" for app in completed)+". "
        if "trainticket" not in completed:
            status_text += "Train Ticket collection and its full experimental column are pending. "
        status_text += "RQ4 reports simulated SRE agents, not recruited human participants. Filled cells retain source hashes and denominators in the accompanying backfill records.\n"
        (self.paper / "backfill/integration_status.tex").write_text(status_text)
        runner=self.out/'backfill_runner.py'
        runner.write_bytes(Path(__file__).read_bytes())
        self.capture(runner,'code')
        arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(self.args).items()}
        mapping = {"schema_version":"r2-backfill-v2","backfill_runner_sha256":sha(runner),"arguments":arguments,
                   "source_archive":"TOSEM_R2_revised_source_20260930.zip", "seed":self.args.seed,
                   "planner":self.args.planner, "data_root":str(self.args.data_root.resolve()), "limit":self.args.limit,
                   "applications":self.applications, "filled_fields":self.fields, "findings":findings, "source_hashes":self.sources,
                   "pending_registry_keys":sorted(pending), "registered_fields":len(declared),
                   "revised_source_present":True, "raw_data_policy":"read-only; integrity proof scoped to recorded consumed files", "submission_ready":False}
        for target in (self.out / "result_mapping.json", self.paper / "backfill/result_mapping.json"):
            target.write_text(json.dumps(mapping,sort_keys=True,indent=2,allow_nan=False)+"\n")
        (self.paper / "table_field_map.json").write_text(json.dumps(dict(sorted(locations.items())),sort_keys=True,indent=2)+"\n")
        return mapping


def run(args):
    raw=args.data_root.resolve()
    for output in (args.out_dir.resolve(),args.paper_dir.resolve()):
        if output==raw or raw in output.parents:
            raise ValueError("Manuscript outputs must be outside raw data")
    if args.seed != 42 or args.planner != "rule":
        raise ValueError("Backfill uses the prespecified main seed 42 and rule graph planner")
    if args.limit is not None:
        return {"status":"bounded_inventory_only", "limit":args.limit,
                "completed_receipts":[str(path) for path in sorted(args.experiment_root.glob("*/ranking_s42_t42/receipt.json"))[:args.limit]],
                "manuscript_modified":False}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if not (args.paper_dir / "pending_values.tex").is_file():
        raise FileNotFoundError("Uploaded R2 result registry is missing")
    with (args.paper_dir / ".result_integration.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        integrator=Integrator(args)
        for app in ("boutique","trainticket"):
            integrator.application(app)
        return integrator.finish()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,default=PROJECT/"data/experiments_tosem_four_copies")
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--paper-dir",type=Path,default=PROJECT/"ref/TOSEM_R2_revised_source_20260930")
    p.add_argument("--experiment-root",type=Path,default=PROJECT/"artifacts/tosem_four_copies/experiments_20260930")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--planner",choices=("rule",),default="rule")
    p.add_argument("--limit",type=int)
    p.add_argument("--memory-cap-gib",type=float,default=8)
    return p


if __name__=="__main__":
    args=parser().parse_args()
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(int(args.memory_cap_gib*1024**3),int(args.memory_cap_gib*1024**3)))
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    report=run(args)
    print(json.dumps({"filled_fields":len(report.get("filled_fields",{})), "applications":{k:v.get("status") for k,v in report.get("applications",{}).items()},
                      "status":report.get("status","integrated_available_results")},sort_keys=True))

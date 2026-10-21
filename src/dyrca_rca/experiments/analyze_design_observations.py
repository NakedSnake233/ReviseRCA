"""Describe archived training observations without labels, scoring, or training."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from dyrca_rca.interaction.observation import epoch, load_observation


def _known_versions(pods):
    return all(p.get("version") not in (None, "", "unknown") for p in pods)


def _cross_names(snapshot):
    observed = snapshot.get("observed_collections", [])
    if not {"services", "pods"}.issubset(observed):
        return None
    if any(not isinstance(s.get("endpoint_ips"), list) for s in snapshot["services"]):
        return None
    owners = {p["ip"]: p.get("service") for p in snapshot["pods"] if p.get("ip")}
    return sorted({(s["name"], owners[ip]) for s in snapshot["services"]
                   for ip in s["endpoint_ips"]
                   if owners.get(ip) and owners[ip] != s["name"]})


def characterize(view):
    """One case is one unit; unknown comparisons remain None, never False."""
    snapshots = sorted(
        (e for e in view["evidence"] if e["kind"] == "topology"
         and epoch(e["data"]["timestamp"]) <= epoch(view["as_of"])),
        key=lambda e: (epoch(e["data"]["timestamp"]), e["id"]))
    if not snapshots:
        raise ValueError("no eligible topology observations")
    first, last = snapshots[0], snapshots[-1]
    old, new = first["data"], last["data"]
    pair = epoch(old["timestamp"]) < epoch(new["timestamp"])
    available = [set(s.get("observed_collections", [])) for s in (old, new)]
    services = [{s["name"]: s for s in snap["services"]} for snap in (old, new)]
    services_known = pair and all("services" in a for a in available)
    pods_known = pair and all("pods" in a for a in available)
    ids = sorted(set(services[0]) | set(services[1])) if services_known else []
    comparable_endpoints = [s for s in ids if all(
        isinstance(m.get(s, {}).get("endpoint_ips"), list) for m in services)]
    changed_endpoints = [s for s in comparable_endpoints if
                         set(services[0][s]["endpoint_ips"]) !=
                         set(services[1][s]["endpoint_ips"])]
    became_empty = [s for s in comparable_endpoints if services[0][s]["endpoint_ips"]
                    and not services[1][s]["endpoint_ips"]]
    stable_empty = [s for s in comparable_endpoints if not services[0][s]["endpoint_ips"]
                   and not services[1][s]["endpoint_ips"]]
    pod_groups = []
    for snapshot in (old, new):
        groups = defaultdict(list)
        for pod in snapshot["pods"]:
            if pod.get("service"):
                groups[pod["service"]].append(pod)
        pod_groups.append(groups)
    populated = sorted(set(pod_groups[0]) & set(pod_groups[1])) if pods_known else []
    version_known = [s for s in populated if all(_known_versions(g[s]) for g in pod_groups)]
    before, after = _cross_names(old), _cross_names(new)
    new_cross = sorted(set(after) - set(before)) if pair and before is not None and after is not None else None
    traces = view["modalities"]["traces"]
    return {
        "incident_id": view["incident_id"], "cutoff": view["as_of"],
        "observation_version": view["observation_version"],
        "topology_refs": [first["id"], last["id"]],
        "topology_pair_observed": pair,
        "service_collections_observed": services_known, "pod_collections_observed": pods_known,
        "service_set_stable": set(services[0]) == set(services[1]) if services_known else None,
        "endpoint_set_changed": bool(changed_endpoints) if comparable_endpoints else None,
        "pod_name_set_changed": ({p["name"] for p in old["pods"]} !=
                                 {p["name"] for p in new["pods"]}) if pods_known else None,
        "service_pairs": len(ids), "endpoint_pairs_observed": len(comparable_endpoints),
        "endpoint_changed_services": changed_endpoints,
        "endpoint_comparable_services": comparable_endpoints,
        "endpoint_became_empty_services": became_empty, "endpoint_stable_empty_services": stable_empty,
        "populated_pod_service_pairs": len(populated),
        "version_pairs_observed": len(version_known),
        "populated_pod_services": populated, "version_known_services": version_known,
        "cross_names_before": before, "cross_names_after": after,
        "new_cross_names": new_cross,
        "has_cross_names_before": bool(before) if before is not None else None,
        "has_cross_names_after": bool(after) if after is not None else None,
        "has_new_cross_names": bool(new_cross) if new_cross is not None else None,
        "trace_status": traces["status"], "trace_rows_scanned": traces["rows_scanned"],
        "trace_eligible_rows": traces.get("eligible_rows"),
        "trace_unknown_timestamp_rows": traces.get("unknown_timestamp_rows"),
    }


def _fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "percent": 100.0 * numerator / denominator if denominator else None}


def summarize(rows):
    metrics = {}
    for key in ("service_set_stable", "endpoint_set_changed", "pod_name_set_changed",
                "has_cross_names_before", "has_cross_names_after", "has_new_cross_names"):
        known = [r[key] for r in rows if r[key] is not None]
        metrics[key] = _fraction(sum(known), len(known))
    metrics["endpoint_pair_coverage"] = _fraction(
        sum(r["endpoint_pairs_observed"] for r in rows), sum(r["service_pairs"] for r in rows))
    metrics["version_pair_coverage"] = _fraction(
        sum(r["version_pairs_observed"] for r in rows), sum(r["populated_pod_service_pairs"] for r in rows))
    metrics["trace_scan_truncated"] = _fraction(sum(r["trace_status"] == "truncated" for r in rows), len(rows))
    eligible = [r["trace_eligible_rows"] for r in rows if r["trace_eligible_rows"] is not None]
    return {"n_cases": len(rows), "metrics": metrics,
            "excluded_cases": {key: sum(r[key] is None for r in rows) for key in (
                "service_set_stable", "endpoint_set_changed", "pod_name_set_changed", "has_new_cross_names")},
            "endpoint_became_empty_occurrences": sum(len(r["endpoint_became_empty_services"]) for r in rows),
            "endpoint_stable_empty_occurrences": sum(len(r["endpoint_stable_empty_services"]) for r in rows),
            "unique_cross_names_before": sorted({tuple(p) for r in rows for p in (r["cross_names_before"] or [])}),
            "unique_new_cross_names": sorted({tuple(p) for r in rows for p in (r["new_cross_names"] or [])}),
            "missing_trace_summaries": sum(r["trace_status"] == "missing_summary" for r in rows),
            "trace_unknown_timestamp_rows": sum(r["trace_unknown_timestamp_rows"] or 0 for r in rows),
            "eligible_trace_rows": {"n_cases": len(eligible),
                                    "minimum": min(eligible) if eligible else None,
                                    "median": median(eligible) if eligible else None,
                                    "maximum": max(eligible) if eligible else None}}


def run(data_root, release_root, out_dir, limit, seed, planner,
        system="boutique", dataset_id="dyrca-v1"):
    data_root, release_root, out_dir = map(Path, (data_root, release_root, out_dir))
    if limit < 1:
        raise ValueError("limit must be positive")
    manifest = json.loads((release_root / "split.json").read_text())
    if seed != manifest["seed"] or planner != manifest["planner"]:
        raise ValueError("seed and planner must match the archived release")
    members = sorted(manifest["split"]["train"])[:limit]
    other_members = {x for k, v in manifest["split"].items() if k != "train" for x in v}
    if len(set(members)) != len(members) or set(members) & other_members:
        raise ValueError("partition membership is duplicated or overlapping")
    report = json.loads((release_root / "report.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "observations").mkdir(exist_ok=True)
    rows = []
    for name in members:
        case = data_root / name
        view = load_observation(case, dataset_id, system, max_rows=manifest["max_rows"])
        row = characterize(view)
        row.update({"experiment": name, "partition": "train", "task": name.split("-")[0].upper(),
                    "released_observation_matches": view["observation_version"] == report["observation_hashes"][name],
                    "source_paths": [str(case / "topology" / (phase + ".json"))
                                     for phase in ("pre", "during") if (case / "topology" / (phase + ".json")).exists()]
                                    + [str(case / "traces/trace_summary.csv")],
                    "observation_file": "observations/" + name + ".json"})
        row["ignored_topology_nonobject_rows"] = sum(
            sum(not isinstance(x, dict) for x in raw.get(collection, []))
            for phase in ("pre", "during") if (case / "topology" / (phase + ".json")).exists()
            for raw in [json.loads((case / "topology" / (phase + ".json")).read_text())]
            for collection in ("services", "pods", "deployments"))
        (out_dir / row["observation_file"]).write_text(json.dumps(view, sort_keys=True, indent=2) + "\n")
        rows.append(row)
    result = summarize(rows)
    result["config"] = {"data_root": str(data_root), "release_root": str(release_root),
                        "seed": seed, "planner": planner, "limit": limit,
                        "max_rows": manifest["max_rows"], "partition": "train",
                        "start_at": None,
                        "experiment_group": "retrospective_design_observations",
                        "graph_version": "not_materialized; observation-v1",
                        "system": system, "dataset_id": dataset_id,
                        "model_execution": False}
    result["members"] = members
    result["task_counts"] = {task: sum(r["task"] == task for r in rows) for task in sorted({r["task"] for r in rows})}
    result["released_observation_matches"] = sum(r["released_observation_matches"] for r in rows)
    result["ignored_topology_nonobject_rows"] = sum(r["ignored_topology_nonobject_rows"] for r in rows)
    result["units"] = {"structural_change": "case with a comparable pre/during pair",
                       "endpoint_coverage": "Service-name-by-case pair occurrence",
                       "version_coverage": "populated Pod-service-group-by-case pair occurrence",
                       "cross_name_incidence": "case; relation identities are separately enumerated",
                       "trace_distribution": "eligible rows per bounded case scan; no lower time bound"}
    (out_dir / "summary.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    with (out_dir / "cases.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    fields = [key for key in rows[0] if not isinstance(rows[0][key], list)] if rows else []
    with (out_dir / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: r[key] for key in fields} for r in rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=Path("artifacts/harness_temporal_v3_release"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner", choices=["rule"], default="rule")
    parser.add_argument("--system", choices=["boutique", "trainticket"], default="boutique")
    parser.add_argument("--dataset-id", default="dyrca-v1")
    args = parser.parse_args()
    result = run(args.data_root, args.release_root, args.out_dir, args.limit, args.seed,
                 args.planner, args.system, args.dataset_id)
    print(json.dumps({k: result[k] for k in ("n_cases", "task_counts", "metrics", "eligible_trace_rows", "released_observation_matches")}, indent=2))


if __name__ == "__main__":
    main()

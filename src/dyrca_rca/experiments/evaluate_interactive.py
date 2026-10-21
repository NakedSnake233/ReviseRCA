"""Paired topology evaluation with one frozen checkpoint and isolated labels."""
import csv
import json
from copy import deepcopy
from pathlib import Path

import torch

from dyrca_rca.experiments.serve_interactive import parser, selected_observations
from dyrca_rca.experiments.run_interactive import _file_hash, PROTOCOL
from dyrca_rca.experiments.inference import load_checkpoint
from dyrca_rca.interaction.observation import canonical, digest
from dyrca_rca.interaction.pipeline import replay
from dyrca_rca.interaction.state import new_session, apply_event
from dyrca_rca.interaction.tasks import task_options, task_feedback
from dyrca_rca.interaction.workbench import Workbench
from dyrca_rca.interaction.explain import compare_results
from dyrca_rca.models.config import set_seed


def rank_metrics(prediction, truth):
    if not isinstance(truth, str) or not truth:
        return {"label_status": "missing", "hit1": None, "hit3": None, "mrr": None}
    ranked = prediction["ranked_node_ids"]
    rank = ranked.index(truth) + 1 if truth in ranked else None
    return {"label_status": "in_candidates" if rank else "outside_candidates",
            "hit1": float(rank == 1), "hit3": float(rank is not None and rank <= 3),
            "mrr": 1.0 / rank if rank else 0.0}


def evaluate(args):
    if not args.checkpoint or not args.training_manifest:
        raise ValueError("checkpoint and training-manifest are required")
    out, raw = Path(args.out_dir).resolve(), Path(args.data_root).resolve()
    if out == raw or raw in out.parents:
        raise ValueError("out-dir must be outside raw data")
    selected = selected_observations(args)
    training = json.loads(Path(args.training_manifest).read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    manifest = json.loads(checkpoint["manifest_json"])
    split = training["split"]
    if manifest["training_metadata"].get("split_hash") != digest(split):
        raise ValueError("training manifest does not match checkpoint")
    if set(name for name, _ in selected) & set(split["train"] + split["val"]):
        raise ValueError("evaluation overlaps train/validation")
    set_seed(args.seed)
    model, model_name = load_checkpoint(args.checkpoint, "auto")
    active_protocol = model.feature_protocol
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    out.mkdir(parents=True, exist_ok=True)
    model_hash = _file_hash(args.checkpoint)
    workbench = Workbench(args.workbench_dir or out / "sessions", [v for _, v in selected],
                          model, model_name, model_hash)
    records, paired = [], []
    for name, view in selected:
        session = new_session(view)
        baseline = replay(session, model, model_name)[0]
        pre = deepcopy(view)
        snapshots = [e for e in pre["evidence"] if e["kind"] == "topology" and e["data"]["phase"] == "pre"]
        if len(snapshots) != 1:
            raise ValueError("static comparison requires one pre snapshot")
        pre["evidence"] = [e for e in pre["evidence"] if e["kind"] != "topology"] + snapshots
        pre["observation_version"] = digest(pre)
        static_session = new_session(pre)
        plan = {"experiment_id": view["incident_id"], "change_type": "static_observed_pre",
                "confidence": 1.0, "causal_role": "unknown", "node_ops": [], "edge_ops": [],
                "weight_ops": [], "state_transfer": {}, "evidence": [snapshots[0]["id"]]}
        scope = {"system_id": view["system_id"], "as_of": view["as_of"]}
        static_session = apply_event(static_session, {"event_id": "static", "incident_id": view["incident_id"],
                                    "base_revision": 0, "operation": "replace_plan", "target_id": "graph_plan",
                                    "payload": plan, "reason": "static structural control", "actor": "evaluation",
                                    "scope": scope, "verification_status": "verified"}, lambda s: replay(s, model, model_name))
        groups = {"static_pre": replay(static_session, model, model_name)[0], "automatic_rule": baseline}
        case = workbench.snapshot(view["incident_id"])
        feedback_status = "awaiting_operator"
        if args.simulate_feedback:
            if args.workbench_dir:
                raise ValueError("simulation must use a separate output session directory")
            options = task_options(session)
            task = name[:2].upper()
            services = sorted(options["services"], key=lambda s: s["name"])
            if task == "T4":
                candidates = [(s, sorted({p.get("version") for p in options["pods"]
                                         if p.get("service") == s["name"] and p.get("version")})) for s in services]
                candidates = [(s, vs) for s, vs in candidates if len(vs) >= 2]
                if not candidates:
                    feedback_status = "no_observed_version_split"
                else:
                    service, versions = candidates[0]
                    values = {"versions": versions, "weights": [1.0 / len(versions)] * len(versions)}
            else:
                old_services = {s["name"]: s for s in snapshots[0]["data"]["services"]}
                changed = [s for s in services if s.get("endpoint_ips") != old_services.get(s["name"], {}).get("endpoint_ips")]
                changed = sorted(changed, key=lambda s: (s.get("endpoint_ips") != [], s["name"]))
                service = (changed or services)[0]
                values = {"endpoint_ips": service.get("endpoint_ips", [])}
            if feedback_status != "no_observed_version_split":
                feedback = task_feedback(session, baseline, task, service["name"], values,
                                         "simulation", "simulated_operator",
                                         "Synthetic confirmation of observed endpoints; equal version proportions are a simulation assumption.")
                proposed = apply_event(session, feedback, lambda _: None)
                decision = {"event_id": "simulation-accept", "incident_id": view["incident_id"], "base_revision": 0,
                            "event_kind": "decision", "operation": "accept_feedback", "target_id": "simulation", "payload": {},
                            "reason": "scripted pilot, not human feedback", "actor": "simulated_operator", "scope": scope}
                accepted = apply_event(proposed, decision, lambda s: replay(s, model, model_name))
                feedback_status = accepted["outcomes"]["simulation-accept"]["execution_status"]
                if feedback_status == "succeeded":
                    groups["simulated_feedback"] = replay(accepted, model, model_name)[0]
                (out / (view["incident_id"] + "-simulation.json")).write_text(canonical(accepted) + "\n")
        elif case["session"]["state_revision"] > 0:
            simulated = any(e.get("actor", "").startswith("simulated_") for e in case["session"]["events"])
            source = getattr(args, "feedback_source", "operator")
            if simulated and source != "simulation":
                raise ValueError("simulated workbench sessions require --feedback-source simulation")
            group = "simulated_feedback" if source == "simulation" else "operator_feedback"
            groups[group] = case["current"]
            feedback_status = "recorded_" + source + "_feedback"
        # All model outputs and feedback are computed before reading labels.
        for group, result in groups.items():
            (out / (view["incident_id"] + "-" + group + ".json")).write_text(canonical(result) + "\n")
        truth_path = raw / name / "ground_truth.json"
        truth = json.loads(truth_path.read_text()).get("root_cause_service") if truth_path.exists() else None
        for group, result in groups.items():
            records.append(dict(rank_metrics(result["prediction"], truth), incident_id=view["incident_id"], task=name[:2].upper(),
                                group=group, seed=args.seed, planner=args.planner, limit=args.limit, max_rows=args.max_rows,
                                data_root=str(raw), model_hash=model_hash, feature_protocol=active_protocol,
                                graph_version=digest(result["graph"]), feedback_status=feedback_status))
        baseline_metrics = rank_metrics(baseline["prediction"], truth)
        target = groups.get("operator_feedback", groups.get("simulated_feedback", baseline))
        target_metrics = rank_metrics(target["prediction"], truth)
        paired.append({"incident_id": view["incident_id"], "feedback_status": feedback_status,
                       "metric_delta": {k: target_metrics[k] - baseline_metrics[k]
                                        if baseline_metrics[k] is not None else None for k in ("hit1", "hit3", "mrr")},
                       "comparison": compare_results(baseline, target)})
    if any(not torch.equal(frozen[k], v) for k, v in model.state_dict().items()):
        raise RuntimeError("model weights changed during evaluation")
    summaries = []
    for group in sorted({r["group"] for r in records}):
        for task in ("ALL", "T2", "T3", "T4"):
            rows = [r for r in records if r["group"] == group and (task == "ALL" or r["task"] == task)]
            labeled = [r for r in rows if r["label_status"] != "missing"]
            summaries.append({"group": group, "task": task, "n": len(rows), "labeled_n": len(labeled),
                              **{k: sum(r[k] for r in labeled) / len(labeled) if labeled else None for k in ("hit1", "hit3", "mrr")}})
    (out / "results.jsonl").write_text("".join(canonical(r) + "\n" for r in records))
    (out / "comparisons.json").write_text(canonical(paired) + "\n")
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    report = {"model_hash": model_hash, "weights_unchanged": True, "selection": [n for n, _ in selected],
              "training_split_hash": digest(split), "feedback_source": "simulation" if args.simulate_feedback else getattr(args, "feedback_source", "operator"),
              "checkpoint_training": manifest["training_metadata"], "summary": summaries,
              "limitation": "Bounded pilot; simulated feedback is not human evaluation. Group coverage is explicit; no causal improvement claim."}
    (out / "evaluation.json").write_text(canonical(report) + "\n")
    lines = ["# 固定模型拓扑交互小规模评估", "", "使用同一检查点；评估期间权重未改变。",
             "训练与验证共 %s 个样本，评估 %s 个样本；这是流程验证，不是充分的质量实验。" %
             (len(split["train"]) + len(split["val"]), len(selected)),
             "反馈来源：" + report["feedback_source"], "",
             "| 组别 | 样本数 | Hit@1 | Hit@3 | MRR |", "|---|---:|---:|---:|---:|"]
    for row in summaries:
        if row["task"] == "ALL":
            lines.append("| {group} | {n} | {hit1} | {hit3} | {mrr} |".format(**row))
    lines.extend(["", "模拟反馈只确认观察到的端点，并为观察到的版本组假设等比例流量；不使用根因标签。",
                  "unknown 版本保留为未知组，不被推断为 v1。版本节点复用服务级 trace，不能作为版本级异常定位证据。",
                  "标签仅在预测完成后用于评分。无反馈的案例不计作人工评估；缺失标签和候选覆盖显式记录。",
                  "真实人工效果需在工作台收集反馈后，使用 --workbench-dir 重新评估。"])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    p = parser()
    p.description = __doc__
    p.add_argument("--training-manifest", required=True)
    p.add_argument("--workbench-dir")
    p.add_argument("--simulate-feedback", action="store_true")
    p.add_argument("--feedback-source", choices=["operator", "simulation"], default="operator")
    print(canonical(evaluate(p.parse_args())))


if __name__ == "__main__":
    main()

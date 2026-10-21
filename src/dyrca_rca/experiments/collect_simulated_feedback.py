"""Collect bounded, label-blind simulated workbench feedback and evaluate it."""
import json
import random
from collections import Counter
from pathlib import Path
import torch

from dyrca_rca.experiments.serve_interactive import parser, selected_observations
from dyrca_rca.experiments.run_interactive import PROTOCOL, _file_hash
from dyrca_rca.experiments.inference import load_checkpoint
from dyrca_rca.experiments.evaluate_interactive import evaluate
from dyrca_rca.interaction.observation import canonical, digest
from dyrca_rca.interaction.state import new_session
from dyrca_rca.interaction.tasks import task_options
from dyrca_rca.interaction.workbench import Workbench
from dyrca_rca.models.config import set_seed


def simulation_values(view, task):
    """Never receives labels, experiment targets, scores, or injection metadata."""
    options = task_options(new_session(view))
    services = sorted(options["services"], key=lambda s: s["name"])
    if task == "T4":
        for service in services:
            versions = sorted({p.get("version") for p in options["pods"]
                               if p.get("service") == service["name"] and p.get("version")})
            if len(versions) >= 2:
                return task, service["name"], {"versions": versions, "weights": [1.0 / len(versions)] * len(versions)}, "observed_version_groups_equal_weight_assumption"
    snapshots = [e for e in view["evidence"] if e["kind"] == "topology" and e["data"]["phase"] == "pre"]
    old = {s["name"]: s for s in snapshots[0]["data"]["services"]} if snapshots else {}
    # Do not convert unknown endpoints into an explicitly empty list.
    known = [s for s in services if isinstance(s.get("endpoint_ips"), list)]
    changed = [s for s in known if s["endpoint_ips"] != old.get(s["name"], {}).get("endpoint_ips")]
    choices = sorted(changed or known, key=lambda s: (s["endpoint_ips"] != [], s["name"]))
    if not choices:
        raise ValueError("no supported observable feedback target")
    service = choices[0]
    return (task if task != "T4" else "T2"), service["name"], {"endpoint_ips": service["endpoint_ips"]}, (
        "endpoint_confirmation" if task != "T4" else "no_version_split_endpoint_confirmation_fallback")


def collect(args):
    out, raw = Path(args.out_dir).resolve(), Path(args.data_root).resolve()
    if out == raw or raw in out.parents or args.limit < 1:
        raise ValueError("positive limit and output outside raw data required")
    if not args.checkpoint:
        raise ValueError("compatible checkpoint required")
    training = json.loads(Path(args.training_manifest).read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    manifest = json.loads(checkpoint["manifest_json"])
    if manifest["training_metadata"].get("split_hash") != digest(training["split"]):
        raise ValueError("training manifest does not match checkpoint")
    excluded = set(training["split"]["train"] + training["split"]["val"])
    rng = random.Random(args.seed)
    groups = []
    for task in ("t2", "t3", "t4"):
        names = sorted(p.name for p in raw.glob(task + "*") if (p / "metadata.json").is_file() and p.name not in excluded
                       and json.loads((p / "metadata.json").read_text()).get("environment", {}).get("namespace") == args.system)
        rng.shuffle(names)
        groups.append(names)
    names = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)][:args.limit]
    if args.selection:
        names = json.loads(Path(args.selection).read_text())[:args.limit]
        if set(names) & excluded:
            raise ValueError("selection overlaps train/validation")
    if len(names) != args.limit:
        raise ValueError("not enough distinct eligible incidents")
    out.mkdir(parents=True, exist_ok=True)
    config = {"seed": args.seed, "limit": args.limit, "max_rows": args.max_rows,
              "checkpoint_hash": _file_hash(args.checkpoint), "data_root": str(raw),
              "system": args.system, "dataset_id": args.dataset_id,
              "selection": names, "simulation_protocol": "workbench-observable-confirmation-v1"}
    config_path = out / "collection_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("output already belongs to a different collection configuration")
    config_path.write_text(canonical(config) + "\n")
    selection_path = out / "selection.json"
    selection_path.write_text(canonical(names) + "\n")
    args.selection = str(selection_path)
    selected = selected_observations(args)
    set_seed(args.seed)
    model, model_name = load_checkpoint(args.checkpoint, "auto")
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    wb_root = out / "workbench"
    wb = Workbench(wb_root, [v for _, v in selected], model, model_name, config["checkpoint_hash"])
    feedback, events, rows, commands = [], [], [], []
    for index, (name, view) in enumerate(selected, 1):
        task, service, values, policy = simulation_values(view, name[:2].upper())
        fid = "sim-feedback-%03d" % index
        common = {"incident_id": view["incident_id"], "base_revision": 0,
                  "actor": "simulated_operator", "reason": "SIMULATION: " + policy + "; observable evidence only, not human review"}
        proposal = dict(common, action="propose", event_id=fid, task=task, service=service, values=values)
        preview = dict(common, action="preview", event_id=fid + "-preview", target_id=fid)
        decision = dict(common, action="accept_feedback", event_id=fid + "-accept", target_id=fid)
        commands.extend([proposal, preview, decision])
        current = wb.snapshot(view["incident_id"])
        if fid not in current["statuses"]:
            wb.command(proposal)
        current = wb.snapshot(view["incident_id"])
        if decision["event_id"] not in current["session"]["outcomes"]:
            wb.command(preview)
            current = wb.command(decision)
        session = current["session"]
        proposal_event = next(e for e in session["events"] if e["event_id"] == fid)
        outcome = session["outcomes"][decision["event_id"]]
        feedback.append(proposal_event)
        events.extend(session["events"])
        rows.append({"feedback_id": fid, "incident_id": view["incident_id"], "scenario": name[:2].upper(),
                     "task_form": task, "service": service, "policy": policy, "source": "simulation",
                     "execution_status": outcome["execution_status"], "state_revision": session["state_revision"],
                     "graph_changed": digest(current["baseline"]["graph"]) != digest(current["current"]["graph"]),
                     "prediction_changed": current["baseline"]["prediction"] != current["current"]["prediction"],
                     "session_path": str(wb.cases[view["incident_id"]][0])})
        if index % 10 == 0:
            print("collected %s/%s" % (index, args.limit), flush=True)
    for filename, data in (("feedback.jsonl", feedback), ("events.jsonl", events),
                           ("workbench_commands.jsonl", commands), ("feedback_results.jsonl", rows)):
        (out / filename).write_text("".join(canonical(row) + "\n" for row in data))
    if any(not torch.equal(frozen[k], v) for k, v in model.state_dict().items()):
        raise RuntimeError("model weights changed during collection")
    # Feedback collection has finished for every incident before any label scoring.
    args.out_dir = str(out / "evaluation")
    args.workbench_dir = str(wb_root)
    args.feedback_source, args.simulate_feedback = "simulation", False
    report = evaluate(args)
    pairs = json.loads((Path(args.out_dir) / "comparisons.json").read_text())
    deltas = [r["metric_delta"]["mrr"] for r in pairs if r["metric_delta"]["mrr"] is not None]
    summary = {"feedback_count": len(feedback), "audit_event_count": len(events), "collection_weights_unchanged": True,
               "distinct_incidents": len({r["incident_id"] for r in rows}),
               "scenarios": dict(Counter(r["scenario"] for r in rows)),
               "policies": dict(Counter(r["policy"] for r in rows)),
               "execution": dict(Counter(r["execution_status"] for r in rows)),
               "graphs_changed": sum(r["graph_changed"] for r in rows),
               "predictions_changed": sum(r["prediction_changed"] for r in rows),
               "mrr_improved": sum(d > 0 for d in deltas), "mrr_worsened": sum(d < 0 for d in deltas),
               "mrr_unchanged": sum(d == 0 for d in deltas), "evaluation": report}
    (out / "collection_summary.json").write_text(canonical(summary) + "\n")
    lines = ["# 100 条模拟反馈采集与固定模型复验" if args.limit == 100 else "# 模拟反馈复验", "",
             "反馈提议 %s 条，独立案例 %s 个，审计事件 %s 条。" % (len(feedback), summary["distinct_incidents"], len(events)),
             "任务分布：" + canonical(summary["scenarios"]), "执行结果：" + canonical(summary["execution"]),
             "图变化 %s 个；预测变化 %s 个。" % (summary["graphs_changed"], summary["predictions_changed"]),
             "根因排名改善 %s 个、恶化 %s 个、不变 %s 个。" % (summary["mrr_improved"], summary["mrr_worsened"], summary["mrr_unchanged"]),
             "", "| 组别 | 样本数 | Hit@1 | Hit@3 | MRR |", "|---|---:|---:|---:|---:|"]
    for row in report["summary"]:
        if row["task"] == "ALL":
            lines.append("| {group} | {n} | {hit1:.4f} | {hit3:.4f} | {mrr:.6f} |".format(**row))
    lines += ["", "逐任务结果和逐案例差值见 evaluation/summary.csv 与 evaluation/comparisons.json；改善不等于优于所有对照组。",
              "T2/T3 模拟确认已观察到的端点，不人为破坏基线制造收益；T4 使用观察到的版本组和等比例流量假设。",
              "反馈走工作台同一提议、预览、接受接口，明确标注 simulated_operator；不是人工用户研究。",
              "使用既有两训练、一验证样本的同一检查点，不重新训练或调参；权重不变。",
              "原始数据未修改。全部反馈生成完成后才读取根因标签评分；结论受小训练集和模拟策略限制。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    p = parser()
    p.description = __doc__
    p.set_defaults(limit=100)
    p.add_argument("--training-manifest", required=True)
    args = p.parse_args()
    summary = collect(args)
    print(canonical({k: v for k, v in summary.items() if k != "evaluation"}))


if __name__ == "__main__":
    main()

"""Local transactional workbench shared by the UI and controlled evaluation."""
from copy import deepcopy
from pathlib import Path
import os
import threading

from dyrca_rca.interaction.observation import canonical, digest
from dyrca_rca.interaction.state import apply_event, feedback_statuses, new_session
from dyrca_rca.interaction.pipeline import replay
from dyrca_rca.interaction.explain import compare_results
from dyrca_rca.interaction.tasks import task_feedback, task_options


class Workbench:
    def __init__(self, root, observations, model=None, model_name=None, model_hash="none"):
        import json
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.model, self.model_name = model, model_name
        self.lock = threading.Lock()
        self.cases = {}
        for view in observations:
            key = digest([view["observation_version"], model_hash, "workbench-v1"])
            path = self.root / (key + ".json")
            if path.exists():
                case = json.loads(path.read_text())
            else:
                session = new_session(view)
                result = self.replay(session)
                case = {"session": session, "baseline": result, "current": result,
                        "model_hash": model_hash, "history": []}
                self.persist(path, case)
            self.cases[view["incident_id"]] = (path, case)

    def replay(self, session, **kwargs):
        return replay(session, self.model, self.model_name, planner="rule", **kwargs)[0]

    @staticmethod
    def persist(path, case):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(canonical(case) + "\n")
        os.replace(str(temporary), str(path))

    def snapshot(self, incident):
        case = deepcopy(self.cases[incident][1])
        case["options"] = task_options(case["session"])
        case["statuses"] = feedback_statuses(case["session"])
        case["comparison"] = compare_results(case["baseline"], case["current"])
        return case

    def command(self, data):
        with self.lock:
            incident = data["incident_id"]
            path, original = self.cases[incident]
            case = deepcopy(original)
            session = case["session"]
            if data["base_revision"] != session["state_revision"]:
                raise ValueError("正式版本已变化，请刷新后重试")
            action = data["action"]
            if action == "preview":
                result = self.replay(session, mode="preview", feedback_id=data["target_id"])
                return {"preview": result, "comparison": compare_results(case["current"], result)}
            event_id = data["event_id"]
            if action == "propose":
                event = task_feedback(session, case["current"], data["task"], data["service"],
                                      data["values"], event_id, data["actor"], data["reason"])
            else:
                if action not in ("accept_feedback", "reject_feedback", "retract_feedback"):
                    raise ValueError("unknown action")
                event = {"event_id": event_id, "incident_id": incident, "event_kind": "decision",
                         "base_revision": data["base_revision"], "operation": action,
                         "target_id": data["target_id"], "payload": {}, "actor": data["actor"],
                         "reason": data["reason"], "scope": {"system_id": session["observation"]["system_id"],
                                                               "as_of": session["observation"]["as_of"]}}
            trial = {}
            def validate(candidate):
                trial["result"] = self.replay(candidate)
            updated = apply_event(session, event, validate)
            if updated == session:
                return self.snapshot(incident)
            if updated["state_revision"] != session["state_revision"]:
                case["history"].append({"event_id": event_id, "before": case["current"],
                                        "after": trial["result"],
                                        "comparison": compare_results(case["current"], trial["result"])})
                case["current"] = trial["result"]
            case["session"] = updated
            self.persist(path, case)
            self.cases[incident] = (path, case)
            return self.snapshot(incident)

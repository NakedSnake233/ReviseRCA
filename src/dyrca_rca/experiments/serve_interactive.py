"""Serve a local observation-only RCA workbench; no raw data writes."""
import argparse
import json
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from dyrca_rca.experiments.run_interactive import _file_hash, PROTOCOL
from dyrca_rca.experiments.inference import load_checkpoint
from dyrca_rca.interaction.observation import load_observation, canonical
from dyrca_rca.interaction.workbench import Workbench
from dyrca_rca.models.config import set_seed


def selected_observations(args):
    root = Path(args.data_root).resolve()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    if args.selection:
        names = json.loads(Path(args.selection).read_text())
    else:
        # Round-robin task selection prevents a bounded run selecting only T1.
        groups = [sorted(p.name for p in root.glob(t + "*") if (p / "metadata.json").is_file())
                  for t in ("t1", "t2", "t3", "t4")]
        names = [g[i] for i in range(max([len(g) for g in groups] or [0])) for g in groups if i < len(g)]
    selected = []
    for name in names[:args.limit]:
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise ValueError("selection requires experiment directory names")
        path = root / name
        metadata = json.loads((path / "metadata.json").read_text())
        system = metadata.get("environment", {}).get("namespace")
        if system != args.system:
            raise ValueError("selection contains a case from a different system")
        selected.append((name, load_observation(path, args.dataset_id, system, max_rows=args.max_rows)))
    if not selected or len({name for name, _ in selected}) != len(selected):
        raise ValueError("empty or duplicate selection")
    return selected


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/experiments")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--planner", choices=["rule"], default="rule")
    p.add_argument("--max-rows", type=int, default=2000)
    p.add_argument("--system", choices=["boutique", "trainticket"], default="boutique")
    p.add_argument("--dataset-id", default="dyrca-v1")
    p.add_argument("--checkpoint")
    p.add_argument("--selection", help="JSON list of experiment directory names")
    p.add_argument("--port", type=int, default=8765)
    return p


def main():
    args = parser().parse_args()
    if Path(args.out_dir).resolve() == Path(args.data_root).resolve() or Path(args.data_root).resolve() in Path(args.out_dir).resolve().parents:
        raise ValueError("out-dir must be outside raw data")
    set_seed(args.seed)
    model, name = load_checkpoint(args.checkpoint, "auto") if args.checkpoint else (None, None)
    selected = selected_observations(args)
    workbench = Workbench(args.out_dir, [v for _, v in selected], model, name,
                          _file_hash(args.checkpoint) if args.checkpoint else "none")
    token = secrets.token_hex(24)
    page = (Path(__file__).parents[1] / "interaction" / "workbench.html").read_text().replace("__TOKEN__", token)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, data, html=False):
            body = (data if html else canonical(data)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.reply(200, page, True)
            elif self.path == "/api/cases":
                self.reply(200, [dict(workbench.snapshot(view["incident_id"]), task_hint=name[:2].upper())
                                 for name, view in selected])
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/command" or self.headers.get("X-Workbench-Token") != token:
                return self.reply(403, {"error": "invalid request"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 65536:
                    raise ValueError("request size out of range")
                data = json.loads(self.rfile.read(size))
                result = workbench.command(data)
                if "session" in result:
                    result["task_hint"] = next(name[:2].upper() for name, view in selected
                                               if view["incident_id"] == data["incident_id"])
                self.reply(200, result)
            except Exception as exc:
                self.reply(400, {"error": str(exc)[:500]})

    # Prevent separate processes overwriting the same audit bundle.
    import fcntl
    with (Path(args.out_dir) / ".server.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("Workbench: http://127.0.0.1:%s" % args.port, flush=True)
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

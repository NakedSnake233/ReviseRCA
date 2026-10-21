#!/usr/bin/env python3
"""Create deterministic scheduler configs for the eight isolated copies."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out-dir", default="artifacts/tosem_four_copies/config")
    parser.add_argument("--prom-port", type=int, default=9190)
    parser.add_argument("--loki-port", type=int, default=3190)
    parser.add_argument("--tempo-port", type=int, default=3290)
    args = parser.parse_args()

    project = Path(args.project_dir).resolve()
    out = (project / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = {
        "boutique": project / "config/tosem_candidate_boutique_20260930.yaml",
        "trainticket": project / "config/tosem_candidate_trainticket_20260930.yaml",
    }
    for system, source in base.items():
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        for idx in range(1, 5):
            namespace = f"{system}-c{idx}"
            cfg = dict(raw)
            cfg["settings"] = dict(raw.get("settings", {}))
            cfg["settings"]["namespace"] = namespace
            cfg["settings"]["project_dir"] = str(project)
            cfg["observability"] = {
                "prom_url": f"http://127.0.0.1:{args.prom_port}",
                "loki_url": f"http://127.0.0.1:{args.loki_port}",
                "tempo_url": f"http://127.0.0.1:{args.tempo_port}",
            }
            data_dir = project / f"data/experiments_tosem_four_copies/{namespace}"
            cfg["output"] = dict(raw.get("output", {}))
            cfg["output"]["data_dir"] = str(data_dir)
            cfg["output"]["summary_file"] = str(data_dir / "batch_summary.json")
            cfg["output"]["log_file"] = str(data_dir / "batch.log")
            path = out / f"{namespace}.yaml"
            try:
                rendered = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
            except TypeError:  # PyYAML shipped with Python 3.7 lacks sort_keys.
                rendered = yaml.safe_dump(cfg, allow_unicode=True)
            path.write_text(rendered, encoding="utf-8")
    print(f"wrote {len(list(out.glob('*.yaml')))} configs to {out}")


if __name__ == "__main__":
    main()

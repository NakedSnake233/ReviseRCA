"""
Prometheus metrics collector.
Exports query_range results to CSV (per-metric) and a combined Parquet file.
"""
from __future__ import annotations
import csv
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from collector.config import METRIC_QUERIES, METRICS_CSV_COLUMNS, METRICS_STEP, NAMESPACE, PROM_URL

log = logging.getLogger(__name__)


def _iso_to_epoch(iso: str) -> float:
    """ISO8601 → Unix epoch seconds (float)."""
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _epoch_to_iso(epoch: float) -> str:
    """Unix epoch → ISO8601 UTC."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PrometheusCollector:
    """Fetches metrics from Prometheus HTTP API and writes CSV."""

    def __init__(self, base_url: str = PROM_URL):
        self.base_url = base_url.rstrip("/")

    def query_range(self, query: str, start_iso: str, end_iso: str,
                    step: str = METRICS_STEP) -> dict:
        """Run a Prometheus query_range request."""
        start = _iso_to_epoch(start_iso)
        end = _iso_to_epoch(end_iso)
        params = {"query": query, "start": start, "end": end, "step": step}
        resp = requests.get(f"{self.base_url}/api/v1/query_range",
                            params=params, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "success":
            raise RuntimeError(f"Prometheus error: {body}")
        return body["data"]

    # ── flatten response → list[dict] ────────────────────────────
    @staticmethod
    def _flatten(metric_name: str, data: dict) -> list[dict]:
        rows = []
        result_type = data.get("resultType", "")
        for series in data.get("result", []):
            labels = series.get("metric", {})
            # Remove __name__ from labels
            labels.pop("__name__", None)
            labels_json = json.dumps(labels, sort_keys=True)

            for ts_val, val_str in series.get("values", []):
                value = None
                if val_str not in ("NaN", "+Inf", "-Inf"):
                    try:
                        parsed = float(val_str)
                        value = parsed if math.isfinite(parsed) else None
                    except (TypeError, ValueError):
                        value = None
                rows.append({
                    "timestamp": _epoch_to_iso(float(ts_val)),
                    "metric_name": metric_name,
                    "labels_json": labels_json,
                    "value": value,
                })

            # Handle instant vector (single value per series)
            if "values" not in series and "value" in series:
                ts_val, val_str = series["value"]
                value = None
                if val_str not in ("NaN", "+Inf", "-Inf"):
                    try:
                        parsed = float(val_str)
                        value = parsed if math.isfinite(parsed) else None
                    except (TypeError, ValueError):
                        value = None
                rows.append({
                    "timestamp": _epoch_to_iso(float(ts_val)),
                    "metric_name": metric_name,
                    "labels_json": labels_json,
                    "value": value,
                })
        return rows

    # ── export all queries ───────────────────────────────────────
    def export_all(self, start_iso: str, end_iso: str,
                   output_dir: Path, namespace: str = None) -> dict[str, int]:
        """
        Run every query in METRIC_QUERIES and write CSVs.
        Returns {query_name: row_count}.
        """
        import collector.config as cfg
        ns = namespace or cfg.NAMESPACE
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        all_rows: list[dict] = []
        stats: dict[str, int] = {}

        for name, tmpl in METRIC_QUERIES.items():
            query = tmpl.format(ns=ns)
            log.info("  [Prometheus] query=%s  rows=...", name)
            try:
                data = self.query_range(query, start_iso, end_iso)
                rows = self._flatten(name, data)
            except Exception as exc:
                log.warning("  [Prometheus] query %s FAILED: %s", name, exc)
                rows = []

            # Per-metric CSV
            csv_path = output_dir / f"{name}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

            all_rows.extend(rows)
            stats[name] = len(rows)
            log.info("  [Prometheus] query=%s  rows=%d", name, len(rows))

        # Combined Parquet (if pyarrow available)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            if all_rows:
                table = pa.Table.from_pylist(all_rows)
                pq.write_table(table, output_dir / "all_metrics.parquet")
                log.info("  [Prometheus] all_metrics.parquet  rows=%d", len(all_rows))
        except ImportError:
            log.warning("  pyarrow not installed; skipping Parquet export")
        except Exception as exc:
            log.warning("  Parquet export failed: %s", exc)

        return stats

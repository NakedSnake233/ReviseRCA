"""
Tempo trace collector.
Searches for traces in a time window and exports them as JSON.
"""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from collector.config import BOUTIQUE_SERVICES, TEMPO_SEARCH_LIMIT, TEMPO_URL, get_services_for_namespace

log = logging.getLogger(__name__)


def _iso_to_epoch_s(iso: str) -> int:
    """ISO8601 → Unix epoch seconds."""
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def _iso_to_epoch_ns(iso: str) -> int:
    """ISO8601 → Nanoseconds."""
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1e9)


class TempoCollector:
    """Fetches traces from Tempo HTTP API."""

    def __init__(self, base_url: str = TEMPO_URL):
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/echo", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def search(self, service: Optional[str] = None,
               start_iso: Optional[str] = None,
               end_iso: Optional[str] = None,
               limit: int = TEMPO_SEARCH_LIMIT) -> list[dict]:
        """
        Search for traces. Returns list of trace summary objects.
        Each: {"traceID": "...", "startTime": epoch_s, "duration": ns, ...}
        """
        params: dict = {"limit": limit}
        if start_iso:
            params["start"] = _iso_to_epoch_s(start_iso)
        if end_iso:
            params["end"] = _iso_to_epoch_s(end_iso)
        if service:
            # TraceQL: { resource.service.name = "xxx" }
            params["q"] = f'{{resource.service.name="{service}"}}'

        try:
            resp = requests.get(
                f"{self.base_url}/api/search",
                params=params, timeout=30,
            )
            resp.raise_for_status()
            body = resp.json()
            return body.get("traces", [])
        except Exception as exc:
            log.warning("  [Tempo] search failed (service=%s): %s", service, exc)
            return []

    def get_trace(self, trace_id: str) -> Optional[dict]:
        """Fetch a single trace by ID (Jaeger-compatible format)."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/traces/{trace_id}",
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("  [Tempo] get_trace %s failed: %s", trace_id, exc)
            return None

    # ── build a lightweight span summary for CSV ─────────────────
    @staticmethod
    def _summarize_trace(trace_data: dict) -> list[dict]:
        """Extract one row per span from a trace."""
        spans = []
        for batch in trace_data.get("batches", []):
            resource = batch.get("resource", {}).get("attributes", [])
            resource_attrs = {a["key"]: a.get("value", {}).get("stringValue", "")
                              for a in resource}
            for scope_span in batch.get("scopeSpans", []):
                for span in scope_span.get("spans", []):
                    attrs = {a["key"]: a.get("value", {}).get("stringValue", "")
                             for a in span.get("attributes", [])}
                    svc_name = resource_attrs.get("service.name", "")
                    # Strip namespace suffix (e.g. "frontend.boutique" → "frontend")
                    if "." in svc_name:
                        service, namespace = svc_name.rsplit(".", 1)
                        if namespace in ("boutique", "trainticket") or \
                                namespace.startswith(("boutique-c", "trainticket-c")):
                            svc_name = service
                    spans.append({
                        "trace_id": span.get("traceId", ""),
                        "span_id": span.get("spanId", ""),
                        "parent_span_id": span.get("parentSpanId", ""),
                        "operation": span.get("name", ""),
                        "service": svc_name,
                        "start_time_unix_nano": span.get("startTimeUnixNano", "0"),
                        "duration_nano": span.get("endTimeUnixNano", "0") and
                                         int(span.get("endTimeUnixNano", 0)) -
                                         int(span.get("startTimeUnixNano", 0)),
                        "status_code": span.get("status", {}).get("code", ""),
                        "kind": span.get("kind", ""),
                    })
        return spans

    # ── export all traces in time window ─────────────────────────
    def export_traces(self, start_iso: str, end_iso: str,
                      output_dir: Path,
                      services: list = None) -> dict:
        """
        Search for traces across all services, fetch full data,
        and write to JSON + summary CSV.
        Returns {"trace_count": N, "span_count": M, "files": [...]}.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if services is None:
            services = BOUTIQUE_SERVICES

        log.info("  [Tempo] Searching traces in [%s → %s] ...", start_iso, end_iso)

        # Phase 1: Search for trace IDs
        all_trace_ids: set[str] = set()
        for svc in services:
            traces = self.search(service=svc, start_iso=start_iso, end_iso=end_iso)
            for t in traces:
                tid = t.get("traceID", "")
                if tid:
                    all_trace_ids.add(tid)
            if traces:
                log.info("    service=%-25s  found=%d", svc, len(traces))
            # Small delay to avoid overwhelming Tempo
            time.sleep(0.2)

        log.info("  [Tempo] Unique trace IDs: %d", len(all_trace_ids))

        # Phase 2: Fetch each trace
        all_traces: list[dict] = []
        all_span_rows: list[dict] = []
        fetched = 0
        for tid in sorted(all_trace_ids):
            trace = self.get_trace(tid)
            if trace:
                all_traces.append(trace)
                all_span_rows.extend(self._summarize_trace(trace))
                fetched += 1
            if fetched % 20 == 0:
                log.info("    fetched %d / %d traces ...", fetched, len(all_trace_ids))

        # Phase 3: Write raw traces JSON
        raw_path = output_dir / "traces_raw.json"
        with open(raw_path, "w") as f:
            json.dump({"data": all_traces, "total": len(all_traces)}, f, indent=2)
        log.info("  [Tempo] traces_raw.json  traces=%d  spans=%d",
                 len(all_traces), len(all_span_rows))

        # Phase 4: Write span summary CSV
        import csv
        if all_span_rows:
            summary_path = output_dir / "trace_summary.csv"
            fieldnames = ["trace_id", "span_id", "parent_span_id", "operation",
                          "service", "start_time_unix_nano", "duration_nano",
                          "status_code", "kind"]
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_span_rows)
            log.info("  [Tempo] trace_summary.csv  rows=%d", len(all_span_rows))

        return {
            "trace_count": len(all_traces),
            "span_count": len(all_span_rows),
            "files": [str(raw_path.name)] +
                     (["trace_summary.csv"] if all_span_rows else []),
        }

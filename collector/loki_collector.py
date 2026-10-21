"""
Loki log collector.
Exports query_range results to JSONL.
"""
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

from collector.config import LOKI_URL, LOG_LINE_LIMIT, LOG_QUERIES

log = logging.getLogger(__name__)


def _iso_to_epoch_ns(iso: str) -> int:
    """ISO8601 → Unix epoch nanoseconds."""
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1_000_000_000)


def _ns_to_iso(ns: int) -> str:
    """Nanoseconds → ISO8601 UTC."""
    ts = ns / 1_000_000_000
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
           f"{ns % 1_000_000_000:09d}"[:3] + "Z"


# Best-effort log-level extraction
_LEVEL_RE = re.compile(
    r"(?i)\b(level|severity)[\":\s]*(DEBUG|INFO|WARN|ERROR|FATAL|TRACE)\b"
    r"|(?<!\w)(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE)(?!\w)"
)
_STRUCTURED_RE = re.compile(
    r'"trace_?[iI][dD]"\s*:\s*"([0-9a-f]{16,32})"'
)


class LokiCollector:
    """Fetches logs from Loki HTTP API and writes JSONL."""

    def __init__(self, base_url: str = LOKI_URL):
        self.base_url = base_url.rstrip("/")

    def query_range(self, query: str, start_iso: str, end_iso: str,
                    limit: int = LOG_LINE_LIMIT) -> dict:
        """Run a Loki query_range request."""
        start_ns = _iso_to_epoch_ns(start_iso)
        end_ns = _iso_to_epoch_ns(end_iso)
        params = {
            "query": query,
            "start": start_ns,
            "end": end_ns,
            "limit": limit,
            "direction": "forward",
        }
        resp = requests.get(
            f"{self.base_url}/loki/api/v1/query_range",
            params=params, timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "success":
            raise RuntimeError(f"Loki error: {body}")
        return body["data"]

    # ── parse a single log entry ─────────────────────────────────
    @staticmethod
    def _parse_entry(ts_ns: int, line: str, stream_labels: dict) -> dict:
        """Convert a Loki log entry to a structured JSONL row."""
        # Extract best-effort fields
        level_match = _LEVEL_RE.search(line)
        level = level_match.group(2) or level_match.group(3) if level_match else "INFO"

        trace_match = _STRUCTURED_RE.search(line)
        trace_id = trace_match.group(1) if trace_match else ""

        # Extract service name from labels
        service = (
            stream_labels.get("app", "")
            or stream_labels.get("app_kubernetes_io_name", "")
            or stream_labels.get("service", "")
        )
        pod = stream_labels.get("pod", "") or stream_labels.get("pod_name", "")
        node = stream_labels.get("node", "") or stream_labels.get("node_name", "")
        # Static, namespace-scoped Promtail targets retain the actual pod log
        # filename even when Kubernetes discovery labels are unavailable.
        # Recover attribution from that filename rather than discarding it.
        if not pod:
            match = re.search(r"/var/log/pods/[^_/]+_([^_/]+)_[^/]+/", stream_labels.get("filename", ""))
            if match:
                pod = match.group(1)
        if not service and pod:
            service = re.sub(r"-[a-z0-9]{8,10}-[a-z0-9]{5}$", "", pod)
            if service == pod:
                service = re.sub(r"-[0-9]+$", "", pod)

        # Clean up the message (strip ANSI, trim)
        msg = line.strip()[:2000]

        return {
            "ts": _ns_to_iso(int(ts_ns)),
            "service": service,
            "pod": pod,
            "node": node,
            "level": level.upper(),
            "message": msg[:500],
            "trace_id": trace_id,
            "span_id": "",
            "raw": msg,
        }

    # ── export logs ──────────────────────────────────────────────
    def export_logs(self, start_iso: str, end_iso: str,
                    output_dir: Path, namespace: str = None) -> dict[str, int]:
        """
        Run each LOG_QUERY and write JSONL files.
        Returns {query_name: line_count}.
        """
        from collector.config import get_log_queries
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats: dict[str, int] = {}
        queries = get_log_queries(namespace)

        for name, query in queries.items():
            log.info("  [Loki] query=%s ...", name)
            try:
                data = self.query_range(query, start_iso, end_iso)
            except Exception as exc:
                log.warning("  [Loki] query %s FAILED: %s", name, exc)
                stats[name] = 0
                continue

            rows = []
            for stream in data.get("result", []):
                stream_labels = stream.get("stream", {})
                for ts_ns, line in stream.get("values", []):
                    rows.append(self._parse_entry(int(ts_ns), line, stream_labels))

            # Sort by timestamp
            rows.sort(key=lambda r: r["ts"])

            # Write JSONL
            out_path = output_dir / f"{name}.jsonl"
            with open(out_path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            stats[name] = len(rows)
            log.info("  [Loki] query=%s  lines=%d", name, len(rows))

        # If we have both "all" and "errors", also write a per-service split
        all_path = output_dir / "all.jsonl"
        if all_path.exists():
            self._split_by_service(all_path, output_dir)

        return stats

    @staticmethod
    def _split_by_service(all_jsonl: Path, output_dir: Path):
        """Split all.jsonl into per-service files (for convenience)."""
        by_service: dict[str, list] = {}
        with open(all_jsonl) as f:
            for line in f:
                row = json.loads(line)
                svc = row.get("service", "unknown")
                by_service.setdefault(svc, []).append(row)

        for svc, rows in by_service.items():
            svc_path = output_dir / f"{svc}.jsonl"
            with open(svc_path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("  [Loki] split into %d per-service files", len(by_service))

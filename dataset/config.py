"""
Configuration for the RCA dataset pipeline.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
PROJECT_DIR = Path("/home/user/dyrca")
DATA_DIR = PROJECT_DIR / "data" / "experiments"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

# ── Services ───────────────────────────────────────────────────
# Canonical service list (fixed node ordering)
SERVICES = [
    "frontend",
    "productcatalogservice",
    "cartservice",
    "recommendationservice",
    "checkoutservice",
    "paymentservice",
    "shippingservice",
    "currencyservice",
    "emailservice",
    "adservice",
    "redis-cart",
]
SERVICE_TO_IDX = {s: i for i, s in enumerate(SERVICES)}
NUM_SERVICES = len(SERVICES)

# ── Known call graph edges (src → dst) ────────────────────────
KNOWN_EDGES = [
    ("frontend", "productcatalogservice"),
    ("frontend", "cartservice"),
    ("frontend", "recommendationservice"),
    ("frontend", "currencyservice"),
    ("frontend", "adservice"),
    ("frontend", "checkoutservice"),
    ("recommendationservice", "productcatalogservice"),
    ("checkoutservice", "cartservice"),
    ("checkoutservice", "productcatalogservice"),
    ("checkoutservice", "shippingservice"),
    ("checkoutservice", "currencyservice"),
    ("checkoutservice", "paymentservice"),
    ("checkoutservice", "emailservice"),
]
EDGE_TO_IDX = {(s, d): i for i, (s, d) in enumerate(KNOWN_EDGES)}
NUM_EDGES = len(KNOWN_EDGES)

# ── Feature dimensions ────────────────────────────────────────
NODE_FEATURE_DIM = 12   # per service per timestep
EDGE_FEATURE_DIM = 4    # per edge per timestep
TOPO_CHANGE_DIM = 7     # topology change features

NODE_FEATURE_NAMES = [
    "cpu_usage",            # from cpu_usage.csv (pod→service avg)
    "request_rate_in",      # dest = this service
    "request_rate_out",     # src = this service
    "error_rate",           # 5xx rate among all requests
    "latency_p50",          # from latency_p50.csv
    "latency_p99",          # from latency_p99.csv
    "replica_count",        # from replica_count.csv
    "log_error_count",      # errors.jsonl count in window
    "log_total_count",      # all.jsonl count in window
    "trace_error_rate",     # spans with status != ""
    "trace_avg_duration",   # mean span duration (ms)
    "trace_span_count",     # number of spans
]

EDGE_FEATURE_NAMES = [
    "call_rate",            # request_rate for this src→dest
    "error_rate",           # 5xx rate for this edge
    "latency",              # avg span duration on this edge
    "response_flags",       # anomalous flag count
]

TOPO_CHANGE_NAMES = [
    "node_added",
    "node_removed",
    "edge_added",
    "edge_removed",
    "replica_delta_max",    # max |replica change| across services
    "version_split",        # 1 if version coexistence detected
    "change_score",         # from verification_report
]

# ── Change type labels ────────────────────────────────────────
CHANGE_TYPES = [
    "no_change",
    "node_scale",
    "node_drift",
    "edge_redirect",
    "edge_removal",
    "edge_split",
    "weight_shift",
    "mixed",
]
CHANGE_TYPE_TO_IDX = {t: i for i, t in enumerate(CHANGE_TYPES)}

# ── Time parameters ───────────────────────────────────────────
METRIC_STEP_SEC = 5            # Prometheus scrape interval
TIME_WINDOW_BINS = 60          # number of time bins for features

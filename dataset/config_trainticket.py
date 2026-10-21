"""
Configuration for the RCA dataset pipeline — TrainTicket variant.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
PROJECT_DIR = Path("/home/user/dyrca")
DATA_DIR = PROJECT_DIR / "data" / "experiments_trainticket"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed_trainticket"

# ── Services ───────────────────────────────────────────────────
# Canonical service list (fixed node ordering) — 30 core business services
SERVICES = [
    "ts-travel-service",
    "ts-travel2-service",
    "ts-ticketinfo-service",
    "ts-basic-service",
    "ts-preserve-service",
    "ts-preserve-other-service",
    "ts-order-service",
    "ts-order-other-service",
    "ts-seat-service",
    "ts-food-service",
    "ts-food-map-service",
    "ts-consign-service",
    "ts-consign-price-service",
    "ts-cancel-service",
    "ts-rebook-service",
    "ts-security-service",
    "ts-notification-service",
    "ts-payment-service",
    "ts-inside-payment-service",
    "ts-assurance-service",
    "ts-station-service",
    "ts-train-service",
    "ts-route-service",
    "ts-price-service",
    "ts-contacts-service",
    "ts-user-service",
    "ts-config-service",
    "ts-auth-service",
    "ts-verification-code-service",
    "ts-route-plan-service",
]
SERVICE_TO_IDX = {s: i for i, s in enumerate(SERVICES)}
NUM_SERVICES = len(SERVICES)

# ── Known call graph edges (src → dst) ────────────────────────
KNOWN_EDGES = [
    # ts-basic-service calls
    ("ts-basic-service", "ts-station-service"),
    ("ts-basic-service", "ts-train-service"),
    ("ts-basic-service", "ts-route-service"),
    ("ts-basic-service", "ts-price-service"),
    # ts-ticketinfo-service calls
    ("ts-ticketinfo-service", "ts-basic-service"),
    # ts-travel-service calls
    ("ts-travel-service", "ts-ticketinfo-service"),
    ("ts-travel-service", "ts-order-service"),
    ("ts-travel-service", "ts-train-service"),
    ("ts-travel-service", "ts-route-service"),
    ("ts-travel-service", "ts-seat-service"),
    # ts-travel2-service calls
    ("ts-travel2-service", "ts-ticketinfo-service"),
    ("ts-travel2-service", "ts-order-other-service"),
    ("ts-travel2-service", "ts-train-service"),
    ("ts-travel2-service", "ts-route-service"),
    ("ts-travel2-service", "ts-seat-service"),
    # ts-preserve-service calls (12 deps)
    ("ts-preserve-service", "ts-ticketinfo-service"),
    ("ts-preserve-service", "ts-seat-service"),
    ("ts-preserve-service", "ts-notification-service"),
    ("ts-preserve-service", "ts-user-service"),
    ("ts-preserve-service", "ts-assurance-service"),
    ("ts-preserve-service", "ts-station-service"),
    ("ts-preserve-service", "ts-security-service"),
    ("ts-preserve-service", "ts-travel-service"),
    ("ts-preserve-service", "ts-contacts-service"),
    ("ts-preserve-service", "ts-order-service"),
    ("ts-preserve-service", "ts-food-service"),
    ("ts-preserve-service", "ts-consign-service"),
    # ts-preserve-other-service calls (12 deps)
    ("ts-preserve-other-service", "ts-ticketinfo-service"),
    ("ts-preserve-other-service", "ts-seat-service"),
    ("ts-preserve-other-service", "ts-notification-service"),
    ("ts-preserve-other-service", "ts-user-service"),
    ("ts-preserve-other-service", "ts-assurance-service"),
    ("ts-preserve-other-service", "ts-station-service"),
    ("ts-preserve-other-service", "ts-security-service"),
    ("ts-preserve-other-service", "ts-travel2-service"),
    ("ts-preserve-other-service", "ts-contacts-service"),
    ("ts-preserve-other-service", "ts-order-other-service"),
    ("ts-preserve-other-service", "ts-food-service"),
    ("ts-preserve-other-service", "ts-consign-service"),
    # ts-order-service calls
    ("ts-order-service", "ts-station-service"),
    # ts-order-other-service calls
    ("ts-order-other-service", "ts-station-service"),
    # ts-seat-service calls
    ("ts-seat-service", "ts-travel-service"),
    ("ts-seat-service", "ts-order-service"),
    ("ts-seat-service", "ts-train-service"),
    ("ts-seat-service", "ts-config-service"),
    # ts-security-service calls
    ("ts-security-service", "ts-order-service"),
    ("ts-security-service", "ts-order-other-service"),
    # ts-food-service calls
    ("ts-food-service", "ts-food-map-service"),
    ("ts-food-service", "ts-travel-service"),
    ("ts-food-service", "ts-station-service"),
    # ts-consign-service calls
    ("ts-consign-service", "ts-consign-price-service"),
    # ts-cancel-service calls
    ("ts-cancel-service", "ts-notification-service"),
    ("ts-cancel-service", "ts-order-service"),
    ("ts-cancel-service", "ts-order-other-service"),
    ("ts-cancel-service", "ts-inside-payment-service"),
    ("ts-cancel-service", "ts-user-service"),
    # ts-inside-payment-service calls
    ("ts-inside-payment-service", "ts-order-service"),
    ("ts-inside-payment-service", "ts-order-other-service"),
    ("ts-inside-payment-service", "ts-payment-service"),
    # ts-rebook-service calls (14 deps)
    ("ts-rebook-service", "ts-ticketinfo-service"),
    ("ts-rebook-service", "ts-seat-service"),
    ("ts-rebook-service", "ts-notification-service"),
    ("ts-rebook-service", "ts-user-service"),
    ("ts-rebook-service", "ts-assurance-service"),
    ("ts-rebook-service", "ts-station-service"),
    ("ts-rebook-service", "ts-security-service"),
    ("ts-rebook-service", "ts-travel-service"),
    ("ts-rebook-service", "ts-travel2-service"),
    ("ts-rebook-service", "ts-contacts-service"),
    ("ts-rebook-service", "ts-order-service"),
    ("ts-rebook-service", "ts-order-other-service"),
    ("ts-rebook-service", "ts-food-service"),
    ("ts-rebook-service", "ts-consign-service"),
    # ts-route-plan-service calls
    ("ts-route-plan-service", "ts-travel-service"),
    ("ts-route-plan-service", "ts-station-service"),
    ("ts-route-plan-service", "ts-route-service"),
    # ts-auth-service calls
    ("ts-auth-service", "ts-verification-code-service"),
    # ts-user-service calls
    ("ts-user-service", "ts-auth-service"),
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

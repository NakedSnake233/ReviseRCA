"""
Configuration: API endpoints, queries, schemas.
Override any value via environment variables.
"""
import os

# ── API Endpoints (override with env vars) ──────────────────────
PROM_URL = os.getenv("PROM_URL", "http://localhost:9090")
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100")
TEMPO_URL = os.getenv("TEMPO_URL", "http://localhost:3200")
NAMESPACE = os.getenv("NAMESPACE", "boutique")

# ── Application detection ────────────────────────────────────────
def get_services_for_namespace(ns: str = None) -> list:
    """Return the service list (with namespace suffix) for the given namespace."""
    ns = ns or NAMESPACE
    if ns == "trainticket" or ns.startswith("trainticket-c"):
        services = TRAINTICKET_SERVICES
        source_ns = "trainticket"
    elif ns == "boutique" or ns.startswith("boutique-c"):
        services = BOUTIQUE_SERVICES
        source_ns = "boutique"
    else:
        raise ValueError(f"Unsupported application namespace: {ns}")
    suffix = f".{source_ns}"
    return [service[:-len(suffix)] + f".{ns}" for service in services]

# ── Query step interval ─────────────────────────────────────────
METRICS_STEP = os.getenv("METRICS_STEP", "5s")

# ── Prometheus queries (key → PromQL) ──────────────────────────
METRIC_QUERIES = {
    "request_rate": (
        'sum by (source_workload,destination_workload,response_code)'
        '(rate(istio_requests_total{{namespace="{ns}"}}[30s]))'
    ),
    "latency_p50": (
        'histogram_quantile(0.5, sum by (le,destination_workload)'
        '(rate(istio_request_duration_milliseconds_bucket{{namespace="{ns}"}}[30s])))'
    ),
    "latency_p99": (
        'histogram_quantile(0.99, sum by (le,destination_workload)'
        '(rate(istio_request_duration_milliseconds_bucket{{namespace="{ns}"}}[30s])))'
    ),
    "cpu_usage": (
        'sum by (pod)'
        '(rate(container_cpu_usage_seconds_total{{namespace="{ns}",container!="POD",container!=""}}[30s]))'
    ),
    "replica_count": (
        'kube_deployment_status_replicas{{namespace="{ns}"}}'
    ),
    "service_up": (
        # The kube-prometheus installation labels pod targets with several
        # jobs; namespace is the stable selector across both applications.
        'up{{namespace="{ns}"}}'
    ),
    "response_flags": (
        'sum by (response_flags,destination_workload)'
        '(rate(istio_requests_total{{namespace="{ns}"}}[30s]))'
    ),
}

# ── Loki queries (key → LogQL) ─────────────────────────────────
LOG_QUERIES = {
    "all": '{namespace="boutique"}',
    "errors": '{namespace="boutique"} |~ "(?i)(error|err|fail|exception|timeout|refused)"',
}


def get_log_queries(namespace: str = None) -> dict:
    """Return LOG_QUERIES with namespace substituted."""
    ns = namespace or NAMESPACE
    return {
        "all": f'{{namespace="{ns}"}}',
        "errors": f'{{namespace="{ns}"}} |~ "(?i)(error|err|fail|exception|timeout|refused)"',
    }

LOG_LINE_LIMIT = 5000

# ── Online Boutique services ────────────────────────────────────
BOUTIQUE_SERVICES = [
    "frontend.boutique",
    "productcatalogservice.boutique",
    "cartservice.boutique",
    "recommendationservice.boutique",
    "checkoutservice.boutique",
    "paymentservice.boutique",
    "shippingservice.boutique",
    "currencyservice.boutique",
    "emailservice.boutique",
    "adservice.boutique",
]

# ── TrainTicket services ───────────────────────────────────────
TRAINTICKET_SERVICES = [
    "ts-travel-service.trainticket",
    "ts-travel2-service.trainticket",
    "ts-ticketinfo-service.trainticket",
    "ts-basic-service.trainticket",
    "ts-preserve-service.trainticket",
    "ts-preserve-other-service.trainticket",
    "ts-order-service.trainticket",
    "ts-order-other-service.trainticket",
    "ts-seat-service.trainticket",
    "ts-food-service.trainticket",
    "ts-food-map-service.trainticket",
    "ts-consign-service.trainticket",
    "ts-consign-price-service.trainticket",
    "ts-cancel-service.trainticket",
    "ts-rebook-service.trainticket",
    "ts-security-service.trainticket",
    "ts-notification-service.trainticket",
    "ts-payment-service.trainticket",
    "ts-inside-payment-service.trainticket",
    "ts-assurance-service.trainticket",
    "ts-station-service.trainticket",
    "ts-train-service.trainticket",
    "ts-route-service.trainticket",
    "ts-price-service.trainticket",
    "ts-contacts-service.trainticket",
    "ts-user-service.trainticket",
    "ts-config-service.trainticket",
    "ts-auth-service.trainticket",
    "ts-verification-code-service.trainticket",
    "ts-route-plan-service.trainticket",
    "ts-execute-service.trainticket",
    "ts-voucher-service.trainticket",
]

TEMPO_SEARCH_LIMIT = 200


# ── CSV column schema for metrics ───────────────────────────────
# Each row: timestamp, metric_name, labels_json, value
METRICS_CSV_COLUMNS = [
    "timestamp",
    "metric_name",
    "labels_json",
    "value",
]

# ── JSONL column schema for logs ────────────────────────────────
LOG_JSONL_FIELDS = [
    "ts",           # ISO8601 UTC
    "service",      # app label
    "pod",          # pod name
    "node",         # node name
    "level",        # INFO / WARN / ERROR (best-effort)
    "message",      # log line content
    "trace_id",     # if present in structured log
    "span_id",      # if present in structured log
    "raw",          # full original log line
]

"""Prometheus metrics definitions for Aura-9."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# ── TAIS ────────────────────────────────────────────────────────────────────
TAIS_CURRENT_TEMP = Gauge(
    "tais_current_temp_celsius", "Current GPU temperature in Celsius"
)
TAIS_THROTTLE_EVENTS = Counter(
    "tais_throttle_events_total", "Number of TAIS throttle events"
)
TAIS_EMERGENCY_HALTS = Counter(
    "tais_emergency_halts_total", "Number of TAIS emergency halts"
)
TAIS_SENSOR_FAIL_EVENTS = Counter(
    "tais_sensor_fail_events_total", "Number of TAIS sensor failure events"
)
TAIS_RELOAD_DURATION = Histogram(
    "tais_reload_duration_seconds",
    "Duration of model reload during TAIS events",
    buckets=[1, 5, 10, 30, 60],
)

# ── ASD ─────────────────────────────────────────────────────────────────────
ASD_STATE_CHANGES = Counter(
    "asd_state_changes_total", "Total ASD state transitions", ["status"]
)
ASD_ACTIVE_TASKS = Gauge("asd_active_tasks", "Number of currently active tasks")

# ── Memory ──────────────────────────────────────────────────────────────────
MR1_ROUTING_DECISIONS = Counter(
    "mr1_routing_decisions_total",
    "Memory router routing decisions",
    ["content_type", "decision"],
)
MR1_CLASSIFICATION_LATENCY = Histogram(
    "mr1_classification_latency_ms",
    "Memory router classification latency in ms",
    buckets=[1, 5, 10, 50, 100, 500],
)

# ── Inference ────────────────────────────────────────────────────────────────
INFERENCE_REQUESTS = Counter(
    "inference_requests_total", "Total inference requests", ["model"]
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[1, 5, 10, 30, 60, 120],
)


def start_metrics_server(host: str = "127.0.0.1", port: int = 9001) -> None:
    """Start the Prometheus HTTP metrics server."""
    start_http_server(port, addr=host)

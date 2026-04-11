"""Prometheus-compatible metrics endpoint at localhost:9001/metrics."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Info

# -- Agent Health --
aura9_info = Info("aura9", "Aura-9 agent metadata")
aura9_uptime = Gauge("aura9_uptime_seconds", "Agent uptime in seconds")
task_completion_rate = Gauge("aura9_task_completion_rate", "Task completion rate (0-1)")
correction_cycles = Counter("aura9_correction_cycles_total", "Total correction cycles")
watchdog_alerts = Counter("aura9_watchdog_alerts_total", "Total Watchdog alerts")
watchdog_heartbeat_age = Gauge("aura9_watchdog_heartbeat_age_seconds", "Watchdog heartbeat age")

# -- TAIS --
tais_temp = Gauge("tais_current_temp_celsius", "Current GPU junction temperature")
tais_status = Info("tais_status_info", "Current TAIS status")
tais_throttle_events = Counter("tais_throttle_events_total", "Total TAIS throttle events")
tais_emergency_halts = Counter("tais_emergency_halts_total", "Total TAIS emergency halts")

# -- Memory --
l1_utilization = Gauge("apm_l1_utilization_percent", "Redis L1 utilization")
l2_collection_size = Gauge("apm_l2_collection_size", "Qdrant L2 collection size", ["collection"])
l3_node_count = Gauge("apm_l3_node_count", "FalkorDB node count")
l3_edge_count = Gauge("apm_l3_edge_count", "FalkorDB edge count")
memory_promotions = Counter("apm_memory_promotions_total", "Total L1→L2 memory promotions")

# -- Performance --
llm_latency = Gauge("llm_inference_latency_seconds", "LLM inference latency", ["model"])
tool_latency = Gauge("tool_call_latency_seconds", "Tool call latency", ["tool"])
context_utilization = Gauge("context_window_utilization_percent", "Context window usage")
mcp_calls = Gauge("mcp_calls_today", "MCP calls today", ["server_id"])

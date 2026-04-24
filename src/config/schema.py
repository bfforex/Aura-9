"""Pydantic v2 configuration schema for Aura-9."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaisThresholdsConfig(BaseModel):
    normal_max: int = 74
    throttle_trigger: int = 75
    cooldown_trigger: int = 80
    emergency_trigger: int = 83
    cooldown_resume: int = 72
    emergency_resume: int = 72


class TaisConfig(BaseModel):
    thresholds: TaisThresholdsConfig = Field(default_factory=TaisThresholdsConfig)
    polling_interval_seconds: int = 5
    sensor_fail_fallback: str = "THROTTLE"
    sensor_retry_interval_seconds: int = 30


class ModelConfig(BaseModel):
    primary: str = "qwen3.5:9b-instruct-q5_k_m"
    primary_fallback: str = "qwen3.5:9b-instruct-q4_k_m"
    watchdog: str = "qwen3.5:1.5b"
    embedding: str = "nomic-embed-text"
    vision: str = "moondream2"
    temperature: float = 0.3
    context_window: int = 32768


class L1RedisConfig(BaseModel):
    default_ttl_hours: int = 24
    suspended_ttl_days: int = 7
    verbatim_cap_tokens: int = 2048
    promotion_confidence_threshold: float = 0.85


class L2QdrantConfig(BaseModel):
    collections: list[str] = Field(
        default_factory=lambda: [
            "expertise",
            "documentation",
            "skill_library",
            "past_missions",
            "failure_analysis",
        ]
    )
    default_top_k: int = 5
    prefetch_limit: int = 20
    rrf_constant: int = 60
    embedding_dim: int = 768


class L3FalkorDBConfig(BaseModel):
    retry_max_attempts: int = 3
    retry_interval_seconds: int = 60
    retry_queue_ttl_hours: int = 1
    shadow_write_mode: str = "async"


class SignificanceConfig(BaseModel):
    r_normalization_window_days: int = 90
    f_half_life_days: int = 30
    archive_threshold: float = 0.2
    grace_period_days: int = 30
    pin_bonus: float = 1.0


class IsecConfig(BaseModel):
    similarity_threshold: float = 0.92
    min_successful_uses: int = 10
    progress_key: str = "isec:progress"


class MemoryConfig(BaseModel):
    l1_redis: L1RedisConfig = Field(default_factory=L1RedisConfig)
    l2_qdrant: L2QdrantConfig = Field(default_factory=L2QdrantConfig)
    l3_falkordb: L3FalkorDBConfig = Field(default_factory=L3FalkorDBConfig)
    significance: SignificanceConfig = Field(default_factory=SignificanceConfig)
    isec: IsecConfig = Field(default_factory=IsecConfig)


class ContextAllocationsConfig(BaseModel):
    system_prompt: int = 2048
    asd_state_injection: int = 1024
    l2_retrieval: int = 8192
    l3_graph_context: int = 2048
    l1_episodic: int = 4096
    scratchpad: int = 4096
    tool_results: int = 4096
    output_budget: int = 3072


class ContextConfig(BaseModel):
    usable_tokens: int = 28672
    safety_buffer_tokens: int = 4096
    allocations: ContextAllocationsConfig = Field(default_factory=ContextAllocationsConfig)
    compression_trigger_percent: int = 100


class InferenceConfig(BaseModel):
    max_concurrent_orchestration_threads: int = 3
    throttle_concurrent_threads: int = 2
    ollama_host: str = "http://localhost:11434"
    ollama_watchdog_host: str = "http://localhost:11435"
    inference_timeout_seconds: int = 120
    max_retries: int = 3
    retry_backoff_seconds: list[int] = Field(default_factory=lambda: [5, 10, 15])


class RedisConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 6379
    password: str = ""
    db: int = 0


class QdrantConfig(BaseModel):
    host: str = "127.0.0.1"
    rest_port: int = 6333
    grpc_port: int = 6334
    snapshot_dir: str = "./backups/qdrant"
    snapshot_retention_days: int = 30


class FalkorDBConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 6380
    graph_name: str = "aura9"
    snapshot_dir: str = "./backups"
    snapshot_retention_days: int = 30


class RedZoneEntry(BaseModel):
    name: str
    action: str


class SecurityConfig(BaseModel):
    mcp_credentials_path: str = "./secrets/mcp-credentials.yaml"
    red_zones: list[RedZoneEntry] = Field(default_factory=list)
    financial_gate_enabled: bool = True
    pii_scrub_on_l2_promotion: bool = True
    pii_scrub_on_audit_write: bool = True


class WatchdogConfig(BaseModel):
    heartbeat_interval_seconds: int = 30
    heartbeat_ttl_seconds: int = 90
    heartbeat_key: str = "watchdog:heartbeat"
    buffer_key: str = "watchdog:buffer"
    buffer_ttl_seconds: int = 300
    max_restart_attempts: int = 3
    restart_interval_seconds: int = 30
    hard_kill_consecutive_identical: int = 50
    hardware_tier: str = "auto"


class ContinuityConfig(BaseModel):
    checkpoint_interval_minutes: int = 15
    health_check_interval_minutes: int = 5
    falkordb_snapshot_interval_hours: int = 24
    qdrant_snapshot_interval_hours: int = 24
    stale_session_max_hours: int = 24
    checkpoint_ttl_days: int = 30


class ObservabilityConfig(BaseModel):
    metrics_port: int = 9001
    metrics_host: str = "127.0.0.1"
    log_dir: str = "./logs"
    log_rotation_mb: int = 10
    log_retention_days: int = 7
    log_compression: str = "gz"
    audit_path: str = "/mnt/c/aura9-audit/audit.log"
    audit_format: str = "jsonl"
    audit_compress_monthly: bool = True
    audit_archive_months: int = 12


class SessionConfig(BaseModel):
    max_ttl_hours: int = 24
    max_gates_per_mission: int = 5
    gate_coalesce_window_seconds: int = 60
    gate_timeout_renotify_minutes: int = 30
    gate_timeout_suspend_minutes: int = 120


class PathsConfig(BaseModel):
    workspace: str = "."
    logs: str = "./logs"
    backups: str = "./backups"
    archive: str = "./archive"
    skills: str = "./skills"
    migrations: str = "./migrations"
    secrets: str = "./secrets"


class SkillForgeConfig(BaseModel):
    max_synthesis_minutes: int = 30
    min_test_vectors: int = 3
    relevance_threshold: float = 0.75
    test_export_dir: str = "./skills/tests"


class VlaConfig(BaseModel):
    keep_alive_threshold_gb: float = 2.2
    unload_threshold_gb: float = 2.0
    keep_alive_min_cycles: int = 3


class TierConfig(BaseModel):
    approval: str
    description: str


class McpConfig(BaseModel):
    default_daily_limit: int = 100
    tiers: dict[str, TierConfig] = Field(default_factory=dict)


class Aura9Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    tais: TaisConfig = Field(default_factory=TaisConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    falkordb: FalkorDBConfig = Field(default_factory=FalkorDBConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    watchdog: WatchdogConfig = Field(default_factory=WatchdogConfig)
    continuity: ContinuityConfig = Field(default_factory=ContinuityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    skill_forge: SkillForgeConfig = Field(default_factory=SkillForgeConfig)
    vla: VlaConfig = Field(default_factory=VlaConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)

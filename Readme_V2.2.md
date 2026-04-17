# Aura-9: Autonomous Reasoning Agent
## Official Technical Specification
**Version:** 2.2 | **Status:** Active Development Blueprint | **Classification:** Internal Reference

---

> *"The gap between knowing and doing is where most AI systems fail. Aura-9 was designed to live in that gap."*

---

## Table of Contents

1. [Identity & Philosophy](#1-identity--philosophy)
2. [Core Reasoning Engine](#2-core-reasoning-engine)
3. [Triple-Thread Memory System (APM)](#3-triple-thread-memory-system-apm)
4. [Capability & Tooling — The Skill Forge](#4-capability--tooling--the-skill-forge)
5. [Autonomous Task Orchestration](#5-autonomous-task-orchestration)
6. [Security & Operational Guardrails](#6-security--operational-guardrails)
7. [Infrastructure & Hardware](#7-infrastructure--hardware)
8. [Agent Communication Protocol](#8-agent-communication-protocol)
9. [Observability & Diagnostics](#9-observability--diagnostics)
10. [Upgrade & Self-Evolution Pathway](#10-upgrade--self-evolution-pathway)
11. [Deployment Checklist](#11-deployment-checklist)
12. [Appendix A — Glossary](#appendix-a--glossary)
13. [Appendix B — Failure Taxonomy](#appendix-b--failure-taxonomy)
14. [Appendix C — IPC Channel Registry](#appendix-c--ipc-channel-registry)
15. [Appendix D — Port Allocation](#appendix-d--port-allocation)
16. [Appendix E — Benchmark Suite](#appendix-e--benchmark-suite)
17. [Appendix F — Version History](#appendix-f--version-history)

---

## 1. Identity & Philosophy

### 1.1 Designation

| Field | Value |
|---|---|
| **Name** | Aura-9 |
| **Full Title** | Autonomous Reasoning Agent, Generation 9 |
| **Parameter Count** | 9.2 Billion (Targeted Baseline) |
| **Primary Kernel** | Qwen 3.5 9B (via Ollama — Local First) |
| **Design Paradigm** | Reasoning Engine — NOT a chatbot |
| **Operational Mode** | 24/7 Autonomous + On-Demand Interactive |

### 1.2 Core Philosophy

Aura-9 is built on a single foundational rejection: **the prompt-response loop is a ceiling, not a feature.**

Standard AI assistants are reactive — they wait, respond, and forget. Aura-9 operates on a fundamentally different model called **Occupational Intelligence**: once assigned a task, it owns that task. It plans, executes, self-corrects, builds new tools when existing ones fail, and only surfaces to the user at meaningful decision gates — not for every micro-step.

The three philosophical pillars:

- **Autonomy First:** Default to independent action. Escalate to human only when ambiguity exceeds a confidence threshold (default: `< 0.72 certainty score`).
- **Memory as Identity:** An agent without persistent memory is a tool. Aura-9's Triple-Thread APM is what transforms it from a model into an entity with continuity.
- **Verifiable Work:** Every output Aura-9 produces must pass its own internal QA before delivery. No unverified results reach the user.

### 1.3 Design Constraints (Intentional Limitations)

These are not bugs — they are guardrails baked into the architecture:

- Aura-9 does **not** have unsupervised internet write access.
- Aura-9 does **not** execute financial transactions without an explicit dual-confirmation gate.
- Aura-9 does **not** modify its own Watchdog module (see Section 6.1).
- Aura-9 does **not** persist raw user PII in any memory tier beyond the active session.
- Aura-9 does **not** resume autonomous execution after a thermal emergency halt without explicit user confirmation.
- Aura-9 does **not** accept new tasks during graceful shutdown once drain is initiated.

### 1.4 Trivial vs. Non-Trivial Task Classification

The Pre-Processor classifies every incoming task before routing it through the reasoning pipeline.

A task is classified **trivial** if ALL three conditions are met:
- `estimated_complexity < 0.2`
- Zero tool calls required
- Answerable entirely from existing context or L2 retrieval

Trivial tasks bypass the 4-Phase Reasoning Cycle and are answered directly. All other tasks go through the full cycle. This classification is logged in the Mission Manifest as `task_class: TRIVIAL | STANDARD`.

---

## 2. Core Reasoning Engine

### 2.1 Model Stack

```
┌─────────────────────────────────────────────────────┐
│                AURA-9 REASONING CORE                │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │      Primary: Qwen 3.5 9B (Q5_K_M quant)     │  │
│  │      Runtime: Ollama — Local First            │  │
│  │      Context Window: 32,768 tokens            │  │
│  │      Operational Cap: 28,672 tokens (87.5%)   │  │
│  └───────────────────────────────────────────────┘  │
│                        │                            │
│                Chain-of-Thought Core                │
│                        │                            │
│  ┌─────────────┬────────┴───────┬───────────────┐   │
│  │  Planning   │   Execution   │   Reflection  │   │
│  │  Module     │   Module      │   Module      │   │
│  └─────────────┴───────────────┴───────────────┘   │
│                        │                            │
│         TAIS — Thermal-Aware Inference Scheduler    │
│         [Hardware telemetry layer — always active]  │
└─────────────────────────────────────────────────────┘
```

### 2.2 Thermal-Aware Inference Scheduling (TAIS)

TAIS is a hardware telemetry sidecar process that monitors GPU temperature via `pynvml` and dynamically manages the inference workload to protect local hardware during sustained autonomous operation.

**Temperature Sensor Note:**
`pynvml.nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)` returns the **GPU core/die temperature** on consumer NVIDIA hardware — not the junction/hotspot temperature (which is inaccessible on most consumer cards). Before deployment, validate which value your card reports:
```bash
nvidia-smi --query-gpu=temperature.gpu,temperature.memory --format=csv,noheader
```
All threshold references in this spec use **GPU core temperature** as reported by `pynvml`. Adjust thresholds downward by 5–8°C if your card runs consistently hotter under load.

**Polling Interval:**
- Active inference: every **5 seconds**
- Idle / cooldown: every **30 seconds**

**Temperature Threshold Ladder:**

| GPU Core Temp | Status | Action |
|---|---|---|
| `< 75°C` | NORMAL | Full operation — Q5_K_M |
| `75–80°C` | THROTTLE | Initiate model reload to Q4_K_M (see below) |
| `80–83°C` | COOLDOWN | Pause inference queue, drain active call, await < 72°C |
| `> 83°C` | EMERGENCY | Checkpoint → halt inference → CRITICAL alert → await user confirmation |

**Quantization Switch Sequence (TAIS THROTTLE):**
Ollama does not support hot model swapping. A quantization change requires a full model reload:
```
1. Current inference call completes (do not interrupt mid-generation)
2. Remaining queue paused
3. Ollama unloads Q5_K_M (~5–15s)
4. Ollama loads Q4_K_M (~20–60s depending on storage speed)
5. Queue resumes at Q4_K_M
6. TAIS logs THROTTLE event with reload duration
Total expected downtime: 30–90 seconds per switch
```
This latency is acceptable for a thermal protection event. TAIS minimizes churn by requiring temperature to drop to **72°C** before switching back to Q5_K_M — preventing rapid oscillation.

**Recovery Confirmation:**
Resumption after Cooldown or Emergency is **temperature-confirmed**: inference restarts only when GPU core temperature drops below **72°C** (3°C buffer below the throttle threshold), verified on the 30-second idle polling interval.

**TAIS Sensor Failure (`TAIS_SENSOR_FAIL`):**
If `pynvml` fails to read temperature (driver update, WSL2 kernel change, NVML library mismatch), TAIS fails safe:
- Immediately assume **THROTTLE** state (conservative)
- Switch to Q4_K_M (if not already loaded)
- Emit `ALERT` to user: "TAIS sensor unavailable — running at conservative Q4_K_M"
- Retry sensor read every 30 seconds
- Restore normal TAIS operation when sensor is confirmed responsive

**TAIS IPC:** Publishes status updates to Redis pub/sub channel `ipc:tais:status` (see Appendix C).

**TAIS Telemetry (Prometheus format, `localhost:9001/metrics`):**
```
tais_current_temp_celsius
tais_status{value="NORMAL|THROTTLE|COOLDOWN|EMERGENCY|SENSOR_FAIL"}
tais_throttle_events_total
tais_emergency_halts_total
tais_sensor_fail_events_total
tais_active_quantization{value="Q5_K_M|Q4_K_M"}
tais_reload_duration_seconds
```

### 2.3 Chain-of-Thought (CoT) + Recursive Self-Correction

Every **non-trivial** task (see Section 1.4) goes through a **4-Phase Reasoning Cycle:**

#### Phase 1 — Decomposition
The raw task is parsed by the **Prompt Enhancement Pre-Processor** into a structured **Mission Manifest**:

```yaml
mission_manifest:
  manifest_version: "2.2"
  task_id: "uuid-v4"
  session_id: "sess-uuid-timestamp"
  created_at: "ISO-8601"
  task_class: "STANDARD"
  original_intent: "[raw user input]"
  interpreted_goal: "[expanded, KPI-tagged goal]"
  sub_tasks:
    - id: ST-001
      description: "..."
      success_criteria: "..."
      tools_required: ["qdrant_search", "python_exec"]
      estimated_complexity: 0.6
      depends_on: []
  constraints:
    time_budget_minutes: 45
    escalation_threshold: 0.72
    human_gate_required: false
    max_correction_cycles: 3
```

#### Phase 2 — Execution with Inline Reflection
Each sub-task is executed sequentially (or in parallel where the DAG allows). After every tool call or action, the Reflection Module performs a **micro-audit**:

- Did this action match the intended sub-task?
- Did the output meet the `success_criteria`?
- Has the confidence score dropped below threshold?
- Has a failure class been triggered? (See Appendix B)

#### Phase 3 — Recursive Self-Correction
If a sub-task fails verification, Aura-9 enters a correction loop (max 3 iterations by default). Correction behavior is **failure-class-aware** (see Appendix B):

1. Classify the failure mode.
2. Apply the class-appropriate response (retry, alt-tool, degraded mode, or halt).
3. Re-execute with the reformulated approach.
4. If 3 corrections all fail → escalate with: failure class, all attempted approaches, and recommended recovery action.

#### Phase 4 — Synthesis & Delivery
All verified sub-task results are merged into a coherent final output tagged with:
- Confidence score
- Tools used
- Time elapsed
- TAIS status during execution
- Any corrections made, with failure class

### 2.4 State-Driven Context Compression

Aura-9 replaces naive rolling chat history with a **compacted context manifest**. Raw content is treated differently based on its type:

| Content Type | Treatment |
|---|---|
| Raw tool outputs (backtest logs, JSON API responses) | **Retained verbatim** up to 2,048 tokens. Excess truncated with `[TRUNCATED — full output in L1 Redis at key: {key}]` marker |
| Completed CoT reasoning branches | **Collapsed** to single-line micro-summary |
| Confirmed sub-task results | **Compressed** to outcome + confidence score |
| Active sub-task scratchpad | **Full fidelity** — never compressed |
| Superseded plans / abandoned branches | **Pruned** entirely |

**Micro-Summary Format:**
```
[ST-001 ✓ 0.91] Searched Qdrant expertise → 4 chunks on "API retry logic"
```

**Verbatim Size Cap:**
Any single raw tool output exceeding **2,048 tokens** is truncated in-context. The full output is always retained in L1 Redis at `sess:{session_id}:tool_results:{call_id}` for the duration of the session, accessible on demand.

### 2.5 Context Window Management

The **Context Budget Manager** allocates the 32,768-token window with a **4,096-token safety buffer**, capping operational use at **28,672 tokens (87.5%)**. This accounts for tokenizer imprecision and prevents attention quality degradation at full utilization.

| Allocation | Budget (tokens) | Notes |
|---|---|---|
| System Prompt + Identity | 2,048 | Static |
| Active Mission Manifest | 1,024 | Dynamic |
| ASD State Injection | 512 | Current task state from Redis |
| L1 Episodic Memory Injection | 4,096 | Rolling compressed window |
| L2 Semantic Retrieval Chunks | 8,192 | Top-k RRF hybrid results |
| L3 Graph Context | 2,048 | Relevant entity subgraph |
| Working Scratchpad (CoT) | 7,680 | Reasoning + tool calls |
| Output Buffer | 3,072 | Final response assembly |
| **Safety Buffer** | **4,096** | **Never allocated — always reserved** |
| **Total Operational** | **28,672** | **87.5% of 32,768** |

When any allocation approaches its limit, a **Memory Compression Pass** is triggered — collapsing older scratchpad content via State-Driven Context Compression before re-injection.

### 2.6 Precision Planner Mode

Precision Planner Mode is a secondary system instruction set forcing the model into a **JSON-exclusive output mode**. It is invoked strictly for Aura State Daemon (ASD) updates, completely separating high-level planning logic from tool-execution logic.

**Invocation:** Injected as a system prompt override when an ASD write is required. Bypasses the Memory Router directly to Redis + FalkorDB (see Section 3.5 for MR-1 bypass documentation).

**Locked Output Schema:**
```json
{
  "asd_update": {
    "task_id": "string — non-nullable",
    "session_id": "string — non-nullable",
    "current_objective": "string",
    "status": "CREATED|PLANNED|EXECUTING|CORRECTING|VERIFYING|DELIVERED|PAUSED|BLOCKED|SUSPENDED|ESCALATED|FAILED",
    "active_subtasks": ["ST-001"],
    "completed_subtasks": ["ST-000"],
    "blocked_by": "string | null",
    "confidence": 0.0,
    "next_action": "string — exact planned next step",
    "failure_class": "string | null",
    "checkpoint_required": false,
    "tais_status": "NORMAL|THROTTLE|COOLDOWN|EMERGENCY|SENSOR_FAIL"
  }
}
```

**Rules:**
- No natural language outside this JSON structure is permitted.
- The schema is immutable — fields cannot be added, removed, or renamed by the model.
- `task_id` and `session_id` are non-nullable and must match the active Mission Manifest.
- Any output failing JSON validation → model re-prompted once. Second failure → `STATE_FAIL` (Appendix B).

---

## 3. Triple-Thread Memory System (APM)

The APM (Autonomous Persistence Module) gives Aura-9 continuity across sessions, reboots, and extended autonomous operation. It is three distinct systems with different purposes, retention policies, and access patterns, working in concert.

```
┌──────────────────────────────────────────────────────────┐
│                 APM — Memory Architecture                │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │  L1: Episodic│   │ L2: Semantic │   │L3: Relational│  │
│  │   (Redis)    │   │  (Qdrant)    │   │ (FalkorDB)  │  │
│  │              │   │              │   │             │  │
│  │ Session-hot  │   │ Always-on    │   │ Graph of    │  │
│  │ context +    │   │ knowledge    │   │ everything  │  │
│  │ ASD state    │   │ retrieval    │   │ + ASD shadow│  │
│  └──────┬───────┘   └──────┬───────┘   └──────┬──────┘  │
│         │                  │                  │         │
│         └──────────────────┼──────────────────┘         │
│                            ▼                            │
│                   Memory Router (MR-1)                  │
│        [Routes all data except ASD direct writes]       │
│                            │                            │
│                  ISEC Daemon (background)               │
│          [Idle-state consolidation & pruning]           │
└──────────────────────────────────────────────────────────┘
```

### 3.1 Session Identity

Every session is assigned a **Session ID** at startup, stamped on all data written during that session:

```
session_id format: sess-{uuid4}-{unix_timestamp}
example:           sess-a3f2c1d0-4e5b-...-1744320000
```

Stamped on:
- Every L1 Redis key
- Every L2 Qdrant payload object
- Every FalkorDB node created during the session
- Every audit trail entry
- Every ASD state write

### 3.2 L1 — Episodic Memory + ASD State (Redis)

**Purpose:** High-speed working context and live ASD state.

| Property | Value |
|---|---|
| **Technology** | Redis 7.x |
| **Retention** | Session-bound — selectively promoted before purge |
| **Access Latency** | < 1ms |
| **Max Capacity** | 512MB (configurable) |

**Key Namespaces:**

| Namespace | Content | TTL |
|---|---|---|
| `sess:{session_id}:turns` | Conversation turns (Redis Stream) | Session-bound |
| `sess:{session_id}:tool_results:{call_id}` | Raw tool outputs (full verbatim) | Session-bound |
| `sess:{session_id}:scratchpad` | Active CoT state | Session-bound |
| `asd:state` | ASD live state tree (Precision Planner JSON) | No TTL |
| `asd:checkpoint:{task_id}:{ckpt_id}` | Point-in-time ASD snapshots | 7 days |
| `mcp:calls:{server_id}:{date}` | MCP call counter per server per day | 48 hours |
| `watchdog:heartbeat` | Watchdog liveness key | 90s TTL (refreshed every 30s) |

> **Fix applied:** `tool_results` TTL changed from 5 minutes to **session-bound**. Tool outputs are available for the full duration of the session, preventing mid-task data loss on long-running sub-tasks.

**Promotion Criteria (L1 → L2):**
- Task produced reusable factual knowledge.
- User explicitly marked output as important.
- Conclusion confidence score > 0.85.

### 3.3 L2 — Semantic Memory (Qdrant)

**Purpose:** Permanent expertise retrieval — Aura-9's long-term knowledge layer.

| Property | Value |
|---|---|
| **Technology** | Qdrant (Vector Database) |
| **Retention** | Permanent with Significance Score decay |
| **Access Latency** | 5–50ms |
| **Embedding Model** | `nomic-embed-text` (local, via Ollama) |
| **Dimensions** | 768 |

**Collections:**

| Collection | Contents |
|---|---|
| `expertise` | Verified factual knowledge promoted from L1 |
| `documentation` | Indexed reference material |
| `skill_library` | Synthesized Skill blueprints and metadata |
| `past_missions` | Compressed successful mission post-mortems |
| `failure_analysis` | Structured post-mortems for failed/escalated tasks |

**`failure_analysis` Document Schema:**
```json
{
  "task_id": "string",
  "session_id": "string",
  "manifest_version": "string",
  "failure_class": "string",
  "timestamp": "ISO-8601",
  "original_intent": "string",
  "interpreted_goal": "string",
  "sub_task_id": "string",
  "attempted_corrections": [
    {"attempt": 1, "approach": "string", "result": "string"}
  ],
  "final_status": "ESCALATED | FAILED",
  "resolution": "string | null",
  "confidence_at_failure": 0.0,
  "tais_status_at_failure": "string"
}
```

**Retrieval Strategy — Hybrid Search:**
Dense vector similarity combined with sparse BM25 keyword scoring using Reciprocal Rank Fusion (RRF).

**Significance Score (L2):**

```
S = (R × F) + £

R  = Retrieval Frequency  (0.0–1.0, rolling 90-day window, normalized to max retrievals)
F  = Finality             (0.0–1.0, where 1.0 = contributed to a DELIVERED task)
£  = User Override        (0.0–1.0, default 0.0; set to 1.0 via `aura9 memory pin`)

Score range:         0.0–2.0
Archival threshold:  S < 0.2
```

**R Normalization — Rolling 90-Day Window:**
`R` is computed as: `(retrievals in last 90 days) / (max retrievals of any node in last 90 days)`. This prevents old high-use nodes from permanently dominating over more recently relevant knowledge. A node never retrieved in 90 days has `R = 0.0`.

**Decay Behavior:**
- S recalculated after every retrieval event.
- `£ = 1.0` renders a node immortal regardless of R and F.
- Nodes below threshold are flagged — never auto-deleted. ISEC handles archival.
- `EXPERIMENTAL` tool maturity: if a Skill remains `EXPERIMENTAL` for > 30 days with zero failures, it is eligible for manual promotion review regardless of use count (see Section 4.4).

**CLI:**
```bash
aura9 memory pin <node_id>      # Set £ = 1.0
aura9 memory unpin <node_id>    # Reset £ = 0.0
aura9 memory score <node_id>    # Inspect current S, R, F, £
```

### 3.4 L3 — Relational Memory (FalkorDB)

**Purpose:** The structural map of how entities, tasks, tools, and skills connect.

| Property | Value |
|---|---|
| **Technology** | FalkorDB (Graph Database) |
| **Retention** | Permanent with Significance Score decay |
| **Query Language** | Cypher |
| **Access Latency** | 2–20ms |

**Core Graph Schema:**

```cypher
// Node types
(:Session {id, started_at, ended_at, agent_id})
(:Project {id, name, session_id, created_at})
(:Task {id, status, session_id, created_at, significance_score})
(:Tool {id, name, version, maturity, session_id})
(:Skill {id, name, version, maturity, trust_level, session_id, significance_score})
(:File {id, path, type, session_id})
(:Entity {id, name, type, session_id})
(:StateNode {task_id, checkpoint_id, timestamp, status,
             redis_snapshot_key, next_action, session_id})
(:MemoryNode {id, session_id, tier, significance_score, user_pinned})

// Relationships
(Session)-[:CONTAINS]->(Task)
(Project)-[:HAS_TASK]->(Task)
(Task)-[:DEPENDS_ON]->(Task)
(Task)-[:USED_TOOL]->(Tool)
(Task)-[:USED_SKILL]->(Skill)
(Task)-[:PRODUCED]->(File)
(Task)-[:HAS_CHECKPOINT]->(StateNode)
(Agent)-[:HAS_SKILL]->(Skill)
(Skill)-[:SOLVES]->(Task)
(Skill)-[:VERSION_OF]->(Skill)
(Tool)-[:NATIVE_TO]->(Agent)
(MemoryNode)-[:BELONGS_TO_SESSION]->(Session)
```

> **Fix applied:** `(:Session)` node fully defined. `Tool` and `Skill` are now distinct node types (see Section 4.2 for duality resolution). `Skill` carries `VERSION_OF` for versioning; native `Tool` nodes are unversioned.

**Significance Score Decay (L3):**
`Task`, `MemoryNode`, and `Skill` nodes carry `significance_score` using the same `S = (R × F) + £` formula and 90-day rolling window as L2. Nodes below `S < 0.2` are flagged for archival to cold storage.

**ASD Shadow Write:**
```
Every ASD state change (Precision Planner Mode write):
  Step 1: Redis write — synchronous, primary, < 1ms
  Step 2: Redis ACK returned to agent
  Step 3: FalkorDB StateNode write — async, queued after ACK

Survivability gap: up to one async write cycle (~2–5ms).
If Redis crashes immediately after ACK and before FalkorDB write completes,
the most recent ASD state change may be lost. Recovery falls back to the
previous FalkorDB StateNode checkpoint. This is the accepted design tradeoff
for maintaining < 1ms write latency on the critical path.

On Redis failure:
  FalkorDB shadow becomes authoritative source of truth
  MEMORY_FAIL ALERT triggered
  Redis recovery initiated
  On Redis restore → re-sync from latest FalkorDB StateNode, resume
```

**Daily Snapshot (Rollback Protection):**
The Continuity Engine (see Section 5.6) triggers a FalkorDB dump every 24 hours:
```bash
aura9 graph snapshot --output ./backups/falkordb-{date}.dump   # Auto or manual
aura9 graph restore --file ./backups/falkordb-2026-04-10.dump  # Rollback
```
Snapshots retained **30 days** then archived to `./archive/graph/`.

### 3.5 Memory Router (MR-1)

The Memory Router routes all data produced by the inference engine and tools. It runs as a lightweight sidecar communicating via Redis pub/sub channel `ipc:memory:route` (see Appendix C).

```
Incoming data
      │
      ▼
[Is it live ASD state?]
  YES → Direct dual-write: Redis primary + FalkorDB shadow (MR-1 BYPASSED)
  NO  → [Is it live ephemeral working data?]
             YES → L1 Redis (session namespace)
             NO  → [Does it contain factual knowledge or reusable insight?]
                        YES → L2 Qdrant (appropriate collection)
                              + extract entities → L3 FalkorDB
                        NO  → [Does it define a relationship between entities?]
                                   YES → L3 FalkorDB only
                                   NO  → Discard (log to audit trail)
```

**MR-1 Bypass — ASD Writes:**
ASD state updates via Precision Planner Mode write directly to Redis and FalkorDB, bypassing MR-1 entirely. This is intentional — ASD writes are time-critical and must not queue behind Memory Router processing. All other data flows through MR-1.

All routing decisions (including bypass events) are logged to the audit trail with data hash, destination tier, and session ID.

### 3.6 ISEC — Idle-State Epistemic Consolidation

ISEC is a background daemon that activates when the GPU has been idle for more than **5 minutes** with no active ASD task. Before beginning embedding-intensive passes, ISEC checks VRAM headroom to avoid competing with a loaded 9B model.

**Activation Condition:**
```
GPU idle > 5 minutes
AND ASD status == IDLE (no active task)
AND TAIS status == NORMAL (not in THROTTLE/COOLDOWN/EMERGENCY)
AND VRAM free headroom ≥ 400MB (for embedding model co-load)
```

**ISEC Multi-Pass Pipeline:**

```
Pass 1 — L1 Review
  Read all session logs in Redis from completed sessions
  Identify: recurring patterns, established logic, reusable conclusions
  Tag candidates for promotion to L2

Pass 2 — L2 Deduplication  [requires nomic-embed-text loaded]
  Check VRAM headroom before loading embedding model
  Find Qdrant vectors with cosine similarity > 0.97
  Merge duplicates → retain highest-confidence version
  Recalculate S scores for merged nodes

Pass 3 — L3 Enrichment
  For each promoted L2 node → ensure entity nodes + edges exist in FalkorDB
  Connect new nodes to existing task/project/session graph

Pass 4 — Decay Audit
  Recalculate S scores across all L2 and L3 nodes (90-day rolling window)
  Flag nodes with S < 0.2 for archival
  Archive flagged nodes to cold storage → never delete originals

Pass 5 — L1 Pruning
  Purge reviewed L1 session logs confirmed promoted or discarded
  Reset session-bound TTLs are already managed by session close
```

**ISEC IPC:** Publishes pass completion events to `ipc:isec:status` (see Appendix C).

**ISEC Log:**
Each pass logged with: pass number, nodes reviewed, promoted, merged, flagged, archived, duration, VRAM at start/end, GPU temp at start/end.

---

## 4. Capability & Tooling — The Skill Forge

### 4.1 Tool vs. Skill Distinction

Two distinct capability types exist in Aura-9:

| Type | Definition | Source | Versioned? | Node Type |
|---|---|---|---|---|
| **Tool** | Native hardcoded capability | Pre-built, shipped with Aura-9 | No (replaced by new releases) | `(:Tool)` |
| **Skill** | Synthesized capability | Self-generated by Skill Forge | Yes — `skill_id@vN` | `(:Skill)` |

Native Tools (VLA Bridge, MCP Gateway, Code Interpreter, Pre-Processor) are built-in and do not go through the Forge Process. Synthesized Skills are subject to full versioning, maturity tracking, and the Forge promotion pipeline.

### 4.2 Tooling Architecture

```
┌─────────────────────────────────────────────────────┐
│                   SKILL FORGE                       │
│                                                     │
│  ┌─────────────────┐    ┌──────────────────────┐    │
│  │  Native Tools   │    │  Synthesized Skills  │    │
│  │  (Hardcoded)    │    │  (Self-Generated)    │    │
│  └─────────────────┘    └──────────────────────┘    │
│           │                        │                │
│           └───────────┬────────────┘                │
│                       ▼                             │
│                Skill Registry                       │
│           (Qdrant `skill_library`)                  │
│                       │                             │
│                       ▼                             │
│             Tool/Skill Selection Engine             │
│       (Hybrid search + dependency check)            │
│                       │                             │
│                       ▼                             │
│          Synthetic Environmental Backtester         │
│       (100% ground-truth validation gate)           │
└─────────────────────────────────────────────────────┘
```

### 4.3 Native Tool Suite

#### 4.3.1 VLA Bridge — Vision-Language-Action Module

The VLA Bridge enables Aura-9 to perceive and interact with graphical interfaces. Assigned to **Phase 4**, integrates with the Red Zone system from Phase 5.

**Vision Model Selection by VRAM Tier:**

| VRAM Available | Model | VRAM Usage |
|---|---|---|
| 6GB (minimum tier) | `moondream2` only | ~1.8GB |
| 8GB+ | `llava:7b` | ~4.5GB |
| 12GB+ | `llava:13b` | ~8.0GB |

> **Fix applied:** `llava:13b` removed from minimum tier. At 6GB with the 9B primary model loaded (~5.5GB), `llava:13b` (~8GB) is not viable. `moondream2` (~1.8GB) is the only supported vision model on 6GB hardware. VLA vision model is loaded on-demand and unloaded after the snapshot cycle when VRAM is constrained.

| Property | Value |
|---|---|
| **Snapshot Interval** | 60 seconds (24/7 monitoring mode) |
| **On-Demand** | Immediate capture on explicit tool call |
| **Output Schema** | `{elements: [], clickable_zones: [], text_content: "", session_id: ""}` |

> **Fix applied:** `red_zones: []` removed from VLA output schema. Red Zones are defined in config and enforced as a separate lookup layer applied to VLA output — not detected by the vision model.

**Red Zone Enforcement Layer:**
After VLA output is produced, a separate Red Zone interceptor checks every element in `clickable_zones` against the Red Zone Registry. Matches are filtered from the action set before the 9B model ever sees available click targets.

**24/7 Monitoring Flow:**
```
Every 60s:
  1. Check VRAM headroom (load moondream2 only if ≥ 2.2GB free)
  2. Capture screenshot
  3. Parse via vision model → structured JSON (sans red_zones)
  4. Apply Red Zone enforcement layer → filter blocked zones
  5. Compare against expected state from last checkpoint
  6. If drift detected → log anomaly to audit trail
  7. If critical anomaly → interrupt task → address → resume from checkpoint
  8. Unload vision model if VRAM headroom was tight
```

#### 4.3.2 MCP Gateway — Zero-Trust External Access

| Property | Value |
|---|---|
| **Protocol** | Model Context Protocol (MCP) |
| **Authentication** | Per-server OAuth 2.0 / API Key vault |
| **Logging** | All calls logged to audit trail with request hash |
| **Timeout** | 30s default, 120s for bulk operations |
| **Call Accounting** | `mcp:calls:{server_id}:{date}` in Redis (48h TTL) |

**Access Tiers:**

| Tier | Scope | Approval |
|---|---|---|
| Tier 1 | Read-only, public data | Auto-approved |
| Tier 2 | Write access to designated workspaces | Session-approved |
| Tier 3 | Destructive or irreversible actions | Human gate required |

**Server Trust Levels:** `TRUSTED` (user-configured) / `COMMUNITY` (public registry, read-only default) / `UNTRUSTED` (explicit per-session approval required).

**Call Accounting CLI:**
```bash
aura9 mcp stats                       # Call counts per server today
aura9 mcp set-limit <server_id> <n>   # Set daily alert threshold
```

#### 4.3.3 Code Interpreter — Subprocess Sandbox

> **Fix applied:** Python does not run natively in WebAssembly. Pyodide (Python-in-Wasm) has significant limitations: no threading, restricted stdlib, ~10MB overhead, and no native extension support. The execution sandbox uses a **subprocess isolation model** instead.

**Sandbox Implementation:** Docker container (`--network=none --read-only --tmpfs /tmp`) or `firejail`/`nsjail` on Linux. Python executes as a subprocess inside the jail, with all outputs captured and returned to the orchestrator.

| Property | Value |
|---|---|
| **Languages** | Python 3.12+, Bash (restricted) |
| **Isolation** | Docker `--network=none --read-only` or firejail |
| **Network** | Blocked by default; Tier-2 approval spawns a separate network-enabled container |
| **File System** | Ephemeral tmpfs — manual promotion to persistent storage |
| **Memory Limit** | 512MB per execution (`--memory=512m`) |
| **CPU Timeout** | 120s hard wall (`timeout 120`) |

**Approved Packages (installed in sandbox image):**
- Data: `numpy`, `pandas`, `polars`
- Text: `re`, `json`, `yaml`, `markdown`
- HTTP (Tier-2 network container only): `httpx`, `requests`
- Crypto: `cryptography`, `hashlib`
- Files: `openpyxl`, `pypdf`, `pillow`

Import of any package not in the sandbox image raises `TOOL_FAIL` immediately.

#### 4.3.4 Prompt Enhancement Pre-Processor

Every user input passes through the Pre-Processor before reaching the 9B model. Mandatory.

**Processing Steps:**
1. Task classification (TRIVIAL vs. STANDARD — see Section 1.4)
2. Intent extraction
3. Ambiguity detection
4. KPI injection — what does "done" look like?
5. Constraint mapping
6. Session ID stamping + `manifest_version` + `created_at`
7. Mission Manifest assembly (YAML output)

**Clarification Policy:** If ambiguity score > 0.60 → one targeted clarification question surfaced to user before manifest assembly.

### 4.4 Autonomous Skill Synthesis — The Forge Process

```
┌───────────────────────────────────────────────────────┐
│                   FORGE PROCESS                       │
│                                                       │
│  1. Skill Gap Detected                                │
│     └─ No registry match above similarity 0.75       │
│                                                       │
│  2. Skill Blueprint Generation                        │
│     └─ 9B drafts Python implementation               │
│     └─ Assigned version tag: skill_id@v1             │
│     └─ Max synthesis time: 30 minutes                │
│        (circuit breaker → TOOL_FAIL + escalate)      │
│                                                       │
│  3. Synthetic Environmental Backtesting               │
│     └─ Execute in subprocess sandbox                 │
│     └─ Against domain data stream (market ticks,     │
│        power-load scenarios, or generic test vectors) │
│     └─ Must achieve 100% ground-truth validation     │
│                                                       │
│  4. Watchdog Review                                   │
│     └─ Audit: network calls, file writes,            │
│        banned imports, injection patterns            │
│                                                       │
│  5. Promotion Decision                                │
│     ├─ Auto-promote: 100% validation + safe audit    │
│     └─ Gate-required: any Tier-2+ operations         │
│                                                       │
│  6. Registry Entry                                    │
│     └─ Skill stored in Qdrant `skill_library`        │
│     └─ (:Skill {id, version: "v1"})-[:SOLVES]->(Task)│
│     └─ Status: EXPERIMENTAL                          │
└───────────────────────────────────────────────────────┘
```

### 4.5 Skill Versioning

| Version Event | Behavior |
|---|---|
| New synthesis | `skill_id@v1` — status: `EXPERIMENTAL` |
| Patch or improvement | `skill_id@v2` — previous version → `DEPRECATED` |
| `DEPRECATED` retention | 7 days, then archived |
| Rollback | `aura9 skill rollback <skill_id> --to v1` |
| Quarantine | Auto after 3 task failures — manual inspection required |
| Idle promotion review | `EXPERIMENTAL` for > 30 days with 0 failures → eligible for manual promotion |

**Skill Maturity Levels:**

| Level | Criteria | Trust |
|---|---|---|
| `EXPERIMENTAL` | Newly synthesized, < 5 uses | Sandbox only |
| `VALIDATED` | > 10 successful uses, 0 failures | Standard access |
| `TRUSTED` | > 50 uses, < 2% failure rate | Full access |
| `CORE` | Manually promoted by user | Always loaded |
| `DEPRECATED` | Superseded by newer version | Read-only, 7-day retention |
| `QUARANTINED` | 3+ task failures | Suspended pending review |

---

## 5. Autonomous Task Orchestration

### 5.1 Aura State Daemon (ASD)

The ASD is a persistent JSON state tree in Redis — the **absolute source of truth** for Aura-9's current mission. Updated exclusively via Precision Planner Mode (direct write, bypasses MR-1).

**Live State:** `asd:state` (Redis, no TTL)
**Shadow:** FalkorDB `StateNode` (async per write, see Section 3.4)
**Snapshots:** `asd:checkpoint:{task_id}:{ckpt_id}` (Redis, 7-day TTL)

**ASD State Schema:** See Section 2.6 Precision Planner Mode locked output schema.

### 5.2 Task Lifecycle

```
CREATED → PLANNED → EXECUTING → [CORRECTING] → VERIFYING → DELIVERED
                                      ↑                |
                                      └────────────────┘
                                  (up to 3 cycles, class-aware)

Event (not a state): CHECKPOINT_SAVED — emitted during EXECUTING, CORRECTING
                     Does not change task status.

Terminal states:
  DELIVERED  — Successful completion
  ESCALATED  — Correction limit exceeded, handed to user
  FAILED     — Unrecoverable error, full diagnostic generated

Pause states:
  PAUSED     — Human gate encountered, awaiting input
  SUSPENDED  — Gate timeout exceeded (> 120 min), resources released
  BLOCKED    — Dependency unavailable, waiting
```

> **Fix applied:** `CHECKPOINTED` removed as a task status. Checkpointing is an **event** (`CHECKPOINT_SAVED`) emitted during execution — the task status remains `EXECUTING`. The ASD `checkpoint_required` flag signals that a checkpoint should be written at the next safe point.

### 5.3 Parallel Task Execution (DAG Scheduler)

```python
task_graph = {
    "ST-001": [],
    "ST-002": [],
    "ST-003": ["ST-001"],
    "ST-004": ["ST-001", "ST-002"],
    "ST-005": ["ST-003", "ST-004"],
}
```

**Concurrency Model:**
Orchestration is parallel — multiple sub-tasks can be in-flight simultaneously as Python threads/asyncio coroutines. Inference is **queued sequential** — Ollama processes one model call at a time on a single GPU. "Parallel tasks" means concurrent orchestration with queued model access, not simultaneous GPU inference.

**Concurrency Limits:**
- Default max in-flight orchestration threads: **3**
- TAIS THROTTLE: auto-reduce to **2**
- Shared tool mutex: VLA Bridge, MCP Gateway (per-server), subprocess sandbox

### 5.4 Human Gate Protocol

1. Complete current atomic action.
2. Write ASD state → emit `CHECKPOINT_SAVED` event → FalkorDB shadow.
3. Generate **Gate Brief** (completed work, decision point, options + recommendation, next steps).
4. Surface Gate Brief to user.
5. Enter `PAUSED` — idle footprint.

**Gate Timeout Ladder:**

| Elapsed | State | Action |
|---|---|---|
| `0–30 min` | `PAUSED` | Idle, awaiting response |
| `30–120 min` | `PAUSED` | Re-notify with Gate Brief summary |
| `> 120 min` | `SUSPENDED` | Full checkpoint → release GPU resources |
| User returns | Recovery | Read ASD from FalkorDB shadow → restore Redis → resume |

### 5.5 Cold-Start Resumption

**WSL2 Trigger — Windows Task Scheduler:**
```
Trigger:  User login event (Windows)
Action:   wsl.exe --exec /home/user/aura9/scripts/startup.sh
Script:   activate venv → python main.py --resume
```

**`--resume` Startup Sequence:**
```python
# startup.py --resume
1. Docker health check — Redis, Qdrant, FalkorDB all responding
2. Query FalkorDB: MATCH (s:StateNode) WHERE s.status IN ['EXECUTING','PAUSED']
                   RETURN s ORDER BY s.timestamp DESC LIMIT 1
3. If found:
   a. Restore Redis ASD from FalkorDB StateNode
   b. Log interruption gap (last_checkpoint → now)
   c. Resume from next_action in StateNode
   d. No user action required
4. If not found:
   → Normal interactive startup
```

### 5.6 Continuity Engine

The Continuity Engine is a **Python daemon process** (`continuity_daemon.py`) that runs alongside the main agent loop. It is responsible for all scheduled background operations that maintain 24/7 operational integrity.

**Responsibilities:**

| Interval | Action |
|---|---|
| Every 15 minutes | Write ASD checkpoint (`asd:checkpoint:{task_id}:{ckpt_id}`) |
| Every 24 hours | FalkorDB graph snapshot to `./backups/falkordb-{date}.dump` |
| Every 5 minutes | Health check sweep (all services) |
| On TAIS EMERGENCY | Trigger immediate checkpoint before halt |
| On SUSPENDED state | Trigger resource release sequence |

**Implementation:** Python `asyncio` scheduled tasks, subscribes to `ipc:continuity:trigger` for on-demand checkpoint requests (see Appendix C).

### 5.7 Startup & Shutdown Sequences

**Startup Order (dependency-ordered):**
```
1. Docker stack health confirmed: Redis → Qdrant → FalkorDB
2. TAIS sidecar started (begins GPU temperature polling)
3. Watchdog model loaded into Ollama
4. Primary 9B model loaded into Ollama
5. Memory Router sidecar started (subscribes to ipc:memory:route)
6. Continuity Engine daemon started
7. ISEC daemon started (idle, waiting for activation condition)
8. ASD state check: resume from checkpoint or initialize fresh
9. Agent ready — accept tasks
```

**Shutdown Sequence (graceful):**
```
1. Reject new task assignments
2. Drain active inference call to completion
3. Write final ASD checkpoint (Precision Planner Mode)
4. Flush Redis to disk (BGSAVE — wait for completion)
5. Stop ISEC daemon (complete current pass if mid-run, then stop)
6. Stop Continuity Engine daemon
7. Stop Memory Router sidecar
8. Stop Watchdog
9. Unload both models from Ollama
10. Stop Docker stack (Redis → Qdrant → FalkorDB)
11. Log clean shutdown to audit trail
```

```bash
aura9 shutdown --graceful    # Executes above sequence
aura9 shutdown --force       # Steps 3 + 10 only (emergency use)
```

---

## 6. Security & Operational Guardrails

### 6.1 The Watchdog — Secondary Oversight Model

| Property | Value |
|---|---|
| **Model** | `Qwen 2.5 1.5B` or `Phi-3 Mini` |
| **VRAM** | ~0.8GB–1.5GB (CPU fallback on 6GB hardware) |
| **Config Storage** | Read-only filesystem partition — 9B has no write access |

**Watchdog IPC:**
The Watchdog receives 9B outputs via Redis pub/sub channel `ipc:watchdog:monitor`. The main agent publishes every inference output to this channel. The Watchdog subscribes, evaluates, and publishes verdicts to `ipc:watchdog:verdict` (see Appendix C).

**What the Watchdog Monitors:**
- **Drift:** Off-mission action detection
- **Loop Detection:** Same action + inputs repeated > 5 times
- **Gate Bypass:** Proceeding past a required human gate
- **Capability Creep:** Tool access above current permission tier
- **Output Toxicity:** Safety classifier flag
- **Schema Violation:** ASD Precision Planner JSON fails validation

**Response:** `Flag → Log → ALERT` (does not auto-kill except hard kill condition).

**Hard Kill:** > 50 identical consecutive actions with identical inputs. Writes checkpoint before halt.

### 6.2 Watchdog Liveness Check

**Mechanism:** Watchdog process refreshes Redis key `watchdog:heartbeat` (TTL: 90s) every **30 seconds**. If the key expires, the liveness check triggers.

**Response to Heartbeat Timeout:**
```
1. CRITICAL alert to user
2. Autonomous execution suspended immediately
3. Interactive-only mode (no autonomous tool execution)
4. Auto-restart Watchdog (up to 3 attempts, 30s between)
5. If 3 restarts fail → full halt, manual restart required
6. Autonomous mode resumes only after heartbeat confirmed
```

### 6.3 Zero-Trust Sanitizer

Pre-execution payload interceptor — operates at the payload level before any outbound tool call or MCP action.

**Scanning Targets + Actions:**

| Finding | Action |
|---|---|
| Hardcoded absolute path outside workspace | Replace with workspace-relative path, log warning |
| Embedded API key / token / secret | **BLOCK** — `SECURITY_FAIL`, alert user |
| Unauthorized network address | **BLOCK** — `SECURITY_FAIL`, alert user |
| Shell injection pattern | **BLOCK** — `SECURITY_FAIL`, alert user |
| Write path outside output directories | Replace with safe placeholder, log warning |

All events — warnings and blocks — written to immutable audit trail.

### 6.4 Visual Red Zone System

**Default Red Zones:**
System Settings, Package Manager, Network Configuration, Authentication prompts, dialogs containing: "Uninstall," "Delete All," "Factory Reset," "Format."

**Custom Config:**
```yaml
red_zones:
  - name: "Production Deploy"
    app: "Jenkins"
    element_label: "Deploy to Production"
    action: "BLOCK"
    alert: true
  - name: "Email Send"
    app: "Thunderbird"
    element_label: "Send"
    action: "GATE"
```

**Action Types:** `BLOCK` (silent, 9B told action failed) / `GATE` (pause, user confirmation) / `ALERT` (allow, flag to user).

### 6.5 Audit Trail

**Physical Location (WSL2):**
`/mnt/c/aura9-audit/audit.log` — Windows NTFS path mounted into WSL2. Outside the agent's Linux working directory. Append-only; agent process has write-only access (no read, no delete).

**Retention Policy:** Compress monthly (`audit-YYYY-MM.log.gz`). Archive after 12 months to `./archive/audit/`. Never delete.

**Schema per entry:**

| Field | Description |
|---|---|
| `event_id` | UUID |
| `timestamp` | ISO-8601 microsecond precision |
| `session_id` | Parent session |
| `task_id` | Parent mission |
| `action_type` | `tool_call`, `llm_inference`, `memory_read`, `memory_write`, `checkpoint_saved`, `gate_triggered`, `watchdog_alert`, `tais_event`, `sanitizer_event`, `isec_pass`, `mr1_route`, `mr1_bypass` |
| `action_detail` | Full parameters (PII-scrubbed) |
| `result_hash` | SHA-256 of output |
| `confidence_score` | 0.0–1.0 |
| `failure_class` | From Appendix B taxonomy (null if no failure) |
| `watchdog_status` | `CLEAR` / `FLAGGED` |
| `tais_status` | TAIS state at event time |

**Metrics format:** Prometheus exposition format at `localhost:9001/metrics`.

---

## 7. Infrastructure & Hardware

### 7.1 Runtime Stack

```
┌──────────────────────────────────────────────────────────┐
│                  HOST SYSTEM (Windows 11 / WSL2)         │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │              Ollama (Local Runtime)              │    │
│  │  ┌──────────────────┐  ┌──────────────────────┐  │    │
│  │  │  Qwen 3.5 9B     │  │  Watchdog 1.5B–3B    │  │    │
│  │  │  (Q5_K_M default)│  │  (CPU on 6GB tier)   │  │    │
│  │  └──────────────────┘  └──────────────────────┘  │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│              TAIS (pynvml — GPU core temp)               │
│                         │                                │
│  ┌──────────┐  ┌─────────┴┐  ┌──────────────────────┐   │
│  │  Redis   │  │  Qdrant  │  │      FalkorDB        │   │
│  │  L1+ASD  │  │   L2     │  │    L3 + ASD Shadow   │   │
│  └──────────┘  └──────────┘  └──────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │   MCP Gateway · VLA Bridge · Subprocess Sandbox  │    │
│  │   Zero-Trust Sanitizer · Memory Router (MR-1)    │    │
│  │   Continuity Engine · ISEC · Watchdog Liveness   │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  Audit trail → /mnt/c/aura9-audit/ (Windows NTFS)        │
└──────────────────────────────────────────────────────────┘
```

### 7.2 Hardware Requirements

#### Minimum Viable — 6GB VRAM

| Component | Specification |
|---|---|
| **GPU** | NVIDIA RTX 3060 / RTX 4060 / equivalent |
| **VRAM Budget** | 5.5GB (9B Q5_K_M) + 0GB (Watchdog on CPU) = **~5.5GB** |
| **VRAM Alert Threshold** | > 95% utilization sustained (separate from TAIS) |
| **System RAM** | 16GB minimum, 32GB recommended |
| **Storage** | 50GB NVMe SSD |
| **OS** | Windows 11 + WSL2 (Ubuntu 22.04+) |

#### Recommended — 8GB+ VRAM

| Component | Value |
|---|---|
| **VRAM Budget** | 5.5GB (9B Q5_K_M) + 1.2GB (Watchdog 3B GPU) = **~6.7GB** |
| **Benefit** | Both models GPU-resident, Watchdog real-time inference |

#### Performance Tier — 16GB+ VRAM

| Component | Value |
|---|---|
| **VRAM Budget** | ~9.2GB (9B Q8_0) + Watchdog + `llava:13b` on-demand |
| **Benefit** | Near-lossless quantization, full vision model simultaneous load |

### 7.3 Quantization Strategy

| Quantization | VRAM | Quality | Use Case |
|---|---|---|---|
| `Q4_K_M` | ~4.8GB | Good | TAIS THROTTLE fallback |
| `Q5_K_M` | ~5.5GB | Very Good | **Default — 6GB GPU** |
| `Q6_K` | ~6.2GB | Excellent | 8GB GPU |
| `Q8_0` | ~9.2GB | Near-lossless | Performance tier |

### 7.4 VRAM Monitoring (Independent of TAIS)

TAIS monitors **temperature**. VRAM utilization is monitored independently:

```python
# Via pynvml — separate from TAIS temperature polling
mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
vram_pct = (mem_info.used / mem_info.total) * 100
```

| VRAM Utilization | Action |
|---|---|
| `< 90%` | Normal |
| `90–95%` | Log warning |
| `> 95%` sustained | `ALERT` to user — consider reducing concurrency or switching quant |

### 7.5 Logging & Log Rotation

```python
logger.add(
    "./logs/{}.log".format(name),
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="gz",
    enqueue=True,
    backtrace=True,
    diagnose=True
)
```

**Log Files:**

| File | Contents | Rotation |
|---|---|---|
| `./logs/aura9.log` | Main inference + orchestration | 10MB / 7-day / gz |
| `./logs/tais.log` | Thermal events only | 10MB / 7-day / gz |
| `./logs/watchdog.log` | Watchdog alerts + heartbeats | 10MB / 7-day / gz |
| `./logs/isec.log` | Consolidation pass reports | 10MB / 30-day / gz |
| `/mnt/c/aura9-audit/audit.log` | Immutable audit trail | Compress monthly / archive 12mo |

---

## 8. Agent Communication Protocol

### 8.1 IPC Model — Redis Pub/Sub

All inter-process communication within Aura-9 uses **Redis pub/sub channels**. This provides a single, observable, low-latency IPC backbone for all sidecar processes. See Appendix C for the full channel registry.

**Pattern:** Publishers do not wait for subscriber acknowledgment. Critical state writes (ASD) use synchronous Redis commands — not pub/sub. Pub/sub is used for event notification and monitoring signals only.

### 8.2 User-Facing Notification — CLI Mapping

Aura-9 is a CLI application. Notification levels map to concrete CLI output:

| Level | Trigger | CLI Delivery |
|---|---|---|
| `SILENT` | Routine operations | Audit log only — no stdout |
| `PROGRESS` | Sub-task completion, milestone | `stdout` status line (overwrite with `\r`) |
| `GATE` | Human input required | Blocking `stdin` prompt with full Gate Brief |
| `ALERT` | Watchdog flag, TAIS throttle, MCP limit, tool failure | `stderr` formatted banner |
| `CRITICAL` | TAIS emergency, Watchdog death, STATE_FAIL, SECURITY_FAIL | `stderr` bold banner + system bell (`\a`) |

### 8.3 Inter-Agent Messaging (Future: Multi-Agent Mode)

```json
{
  "msg_id": "uuid-v4",
  "from_agent": "aura9-primary",
  "to_agent": "aura9-qa",
  "msg_type": "TASK_DELEGATION|RESULT_REPORT|STATUS_QUERY|ESCALATION|HEARTBEAT",
  "priority": "LOW|NORMAL|HIGH|CRITICAL",
  "session_id": "sess-uuid-timestamp",
  "payload": {
    "subtask_id": "ST-004",
    "description": "string",
    "artifacts": [],
    "success_criteria": "string",
    "deadline_minutes": 15
  },
  "timestamp": "ISO-8601"
}
```

---

## 9. Observability & Diagnostics

### 9.1 Metrics Endpoint

**Format:** Prometheus exposition format (`text/plain; version=0.0.4`)
**Address:** `localhost:9001/metrics`

```
# Agent Health
aura9_uptime_seconds
aura9_task_completion_rate
aura9_correction_cycles_total
aura9_watchdog_alerts_total
aura9_watchdog_heartbeat_age_seconds

# TAIS
tais_current_temp_celsius
tais_status
tais_throttle_events_total
tais_emergency_halts_total
tais_sensor_fail_events_total
tais_active_quantization
tais_reload_duration_seconds

# VRAM (independent of TAIS)
vram_utilization_percent
vram_used_bytes
vram_free_bytes

# Memory / APM
apm_l1_utilization_percent
apm_l2_collection_size{collection="expertise|documentation|skill_library|past_missions|failure_analysis"}
apm_l3_node_count
apm_l3_edge_count
apm_memory_promotions_total
apm_significance_score_min
isec_last_run_timestamp
isec_nodes_archived_total

# Performance
llm_inference_latency_seconds{model="primary|watchdog"}
tool_call_latency_seconds{tool}
context_window_utilization_percent
mcp_calls_today{server_id}
forge_synthesis_duration_seconds
```

### 9.2 Diagnostic CLI

```bash
aura9 status                              # Full system health snapshot
aura9 task show --active                  # Active mission + ASD state
aura9 watchdog log --last 10             # Recent alerts
aura9 watchdog status                    # Liveness + heartbeat age
aura9 tais status                        # Temp + quant level + status
aura9 vram status                        # VRAM utilization
aura9 checkpoint --now                   # Force checkpoint
aura9 audit --task-id <uuid>             # Replay audit trail
aura9 memory inspect --tier L2 --query "..."
aura9 memory pin <node_id>
aura9 memory score <node_id>
aura9 memory isec --run-now              # Force ISEC (GPU must be idle)
aura9 skill list                         # All skills by maturity level
aura9 skill rollback <skill_id> --to v1
aura9 mcp stats
aura9 graph snapshot
aura9 graph restore --file <path>
aura9 shutdown --graceful
aura9 shutdown --force
```

### 9.3 Health Check Protocol

On startup and every 5 minutes:

```
✓ Ollama reachable — primary model loaded (Q5_K_M)
✓ Watchdog running — heartbeat 12s ago
✓ TAIS active — 64°C core — NORMAL — Q5_K_M
✓ VRAM — 5.6GB / 6.0GB (93%) — within threshold
✓ Redis responsive — ping < 1ms
✓ Qdrant healthy — 5 collections present
✓ FalkorDB connected — 1,247 nodes / 3,891 edges
✓ ASD state valid — IDLE — session: sess-a3f2...
✓ MCP Gateway — 2/3 servers reachable (1 timeout: logged)
✓ Subprocess sandbox — test execution passed
✓ Zero-Trust Sanitizer — active
✓ Log rotation — active
✓ Audit trail — /mnt/c/aura9-audit/ writable
✓ Last FalkorDB snapshot — 6h ago
```

---

## 10. Upgrade & Self-Evolution Pathway

### 10.1 Model Upgrade Protocol

```
Current: Qwen 3.5 9B
   ├─ Lateral: Mistral 9B, Gemma 9B, LLaMA 3.2 9B
   └─ Vertical: Qwen 3.5 14B, Qwen 3.5 32B (higher VRAM cost)
```

1. Pull new model via Ollama.
2. Run Appendix E benchmark suite against new model.
3. Compare: completion rate, correction rate, TAIS thermal profile, inference latency, context utilization.
4. Promote if benchmark score > current model by > 5%.
5. Retain old model 7 days as rollback.

### 10.2 Continuous Learning (Without Fine-Tuning)

- **Retrieval quality:** L2 grows richer with every successful task.
- **Graph density:** L3 accumulates connections, revealing non-obvious relationships.
- **Skill accumulation:** Each synthesized Skill reduces future latency for similar tasks.
- **Failure analysis:** Every failure generates a `failure_analysis` post-mortem, reducing recurrence.
- **ISEC consolidation:** Background passes promote implicit patterns to explicit L3 nodes.

### 10.3 Memory Lifecycle & Cold Storage

```
Active:   S ≥ 0.2  → retained, scored on every retrieval
Flagged:  S < 0.2  → marked by ISEC for archival
Archived: compressed to ./archive/memory-{date}-{collection}.gz
Pinned:   £ = 1.0  → immune to archival

Restore:
  aura9 memory restore --archive ./archive/memory-2026-01-15-expertise.gz
```

---

## 11. Deployment Checklist

### Pre-Launch

- [ ] NVIDIA drivers updated (CUDA 12.x) — `nvidia-smi` working in WSL2
- [ ] Validate temperature metric: `nvidia-smi --query-gpu=temperature.gpu,temperature.memory --format=csv,noheader`
- [ ] `pynvml` installed — GPU sensor read confirmed (`pynvml.nvmlDeviceGetTemperature()`)
- [ ] Ollama installed — `ollama serve` running
- [ ] `qwen2.5:9b-instruct-q5_k_m` pulled and listed
- [ ] `qwen2.5:1.5b` (Watchdog) pulled and listed
- [ ] `nomic-embed-text` pulled and listed
- [ ] Docker Compose stack healthy: Redis (6379), Qdrant (6333/6334), FalkorDB (6380)
- [ ] FalkorDB graph schema applied — all node types and relationships created
- [ ] `(:Session)` node creation verified in FalkorDB
- [ ] Qdrant collections initialized: `expertise`, `documentation`, `skill_library`, `past_missions`, `failure_analysis`
- [ ] Python project scaffold created, venv active, dependencies installed
- [ ] `config/settings.yaml` reviewed and customized
- [ ] Log directories created — rotation policy active
- [ ] Audit trail path created: `/mnt/c/aura9-audit/` — write-only access confirmed
- [ ] Subprocess sandbox image built — test execution passed
- [ ] Zero-Trust Sanitizer configured — test payload scan passed
- [ ] MCP Gateway — at least one Tier-1 server verified
- [ ] Red Zone config reviewed for your environment
- [ ] Windows Task Scheduler cold-start trigger configured and tested
- [ ] TAIS thresholds set for your specific GPU — thermal profile validated
- [ ] VRAM headroom verified: > 5% free after primary model load
- [ ] All IPC Redis pub/sub channels confirmed accessible (Appendix C)
- [ ] Port allocation verified — no conflicts (Appendix D)
- [ ] Metrics endpoint responding: `curl localhost:9001/metrics`
- [ ] FalkorDB daily snapshot job scheduled via Continuity Engine

### First Run

- [ ] Full health check — all 14 checks green
- [ ] Simple test task — verify full lifecycle (CREATED → DELIVERED)
- [ ] ASD state updates visible in Redis (`asd:state`) with `task_id` and `session_id`
- [ ] `session_id` stamped on L1 keys, L2 payloads, L3 nodes, and audit entries
- [ ] `CHECKPOINT_SAVED` event emitted — verify via audit trail (not a status field)
- [ ] Simulate reboot — cold-start recovery resumes from checkpoint without user action
- [ ] TAIS telemetry visible on metrics endpoint
- [ ] VRAM utilization visible on metrics endpoint (independent of TAIS)
- [ ] Watchdog heartbeat visible: `redis-cli get watchdog:heartbeat`
- [ ] Kill Watchdog — confirm `CRITICAL` alert within 90s, autonomous mode suspended
- [ ] Restore Watchdog — confirm heartbeat resumes, autonomous mode restored
- [ ] `aura9 memory score` on a promoted node — S formula confirmed
- [ ] Tool result accessible for full session duration (session-bound TTL verified)
- [ ] Graceful shutdown — all 11 steps complete, clean audit entry logged
- [ ] FalkorDB snapshot created on 24h schedule — test manual restore

### Ongoing Operations

- [ ] Review Watchdog alert log weekly
- [ ] Review TAIS thermal log weekly — sustained patterns
- [ ] Review VRAM utilization trend weekly
- [ ] Inspect skill_library monthly — deprecated/quarantined skills
- [ ] `aura9 memory isec --run-now` monthly — review consolidation report
- [ ] Audit MCP call counters monthly — adjust limits
- [ ] FalkorDB snapshot restore test quarterly
- [ ] Cold storage archive restore test quarterly

---

## Appendix A — Glossary

| Term | Definition |
|---|---|
| **APM** | Autonomous Persistence Module — the Triple-Thread memory system (Redis + Qdrant + FalkorDB) |
| **ASD** | Aura State Daemon — persistent JSON state tree in Redis, authoritative source of truth for the current mission |
| **CHECKPOINT_SAVED** | Event emitted when a checkpoint is written. Not a task status — task remains `EXECUTING` |
| **CoT** | Chain-of-Thought — step-by-step reasoning style |
| **Continuity Engine** | Python asyncio daemon managing scheduled background operations (checkpoints, snapshots, health checks) |
| **DAG** | Directed Acyclic Graph — dependency graph for parallel sub-task scheduling |
| **Gate** | Deliberate pause requiring human input before autonomous execution proceeds |
| **ISEC** | Idle-State Epistemic Consolidation — background daemon consolidating memory during GPU idle periods |
| **MCP** | Model Context Protocol — standardized tool and API connectivity protocol |
| **Mission Manifest** | Structured YAML task plan generated by the Pre-Processor, versioned at `manifest_version: 2.2` |
| **MR-1** | Memory Router — routes all data to appropriate APM tier (bypassed for ASD direct writes) |
| **Precision Planner Mode** | JSON-exclusive system prompt override used solely for ASD state updates |
| **Red Zone** | UI region Aura-9 is blocked from interacting with, enforced below the agent's awareness |
| **S Score** | `S = (R × F) + £` — significance decay formula governing memory archival in L2 and L3 |
| **Skill** | Synthesized capability generated by the Skill Forge — versioned, maturity-tracked |
| **Skill Forge** | System by which Aura-9 synthesizes, tests, versions, and promotes new Skills |
| **StateNode** | FalkorDB node storing ASD shadow copies and checkpoint history for recovery |
| **TAIS** | Thermal-Aware Inference Scheduling — GPU core temperature telemetry protecting hardware |
| **Tool** | Native hardcoded capability (VLA Bridge, MCP Gateway, Code Interpreter, Pre-Processor) |
| **VLA Bridge** | Vision-Language-Action module — screen perception and UI interaction |
| **Watchdog** | Secondary 1B–3B oversight model monitoring 9B for drift, loops, violations |
| **Zero-Trust Sanitizer** | Pre-execution payload interceptor scanning for secrets, injection, unauthorized calls |

---

## Appendix B — Failure Taxonomy

| Class | Examples | Immediate Response | Correction Behavior |
|---|---|---|---|
| `INFERENCE_FAIL` | Model timeout, OOM, malformed JSON output | Retry × 2 | Fallback quant → CPU offload → halt |
| `TOOL_FAIL` | MCP timeout, sandbox crash, import denied, Forge timeout | Retry × 1 | Check skill registry for alt → escalate if none |
| `MEMORY_FAIL` | Redis down, Qdrant unreachable, FalkorDB write error | Promote shadow / degrade | ALERT → continue degraded if non-critical |
| `STATE_FAIL` | ASD corruption, Precision Planner schema violation × 2 | Halt ASD writes | Restore from FalkorDB shadow → manual if shadow corrupt |
| `SECURITY_FAIL` | Red Zone breach, sanitizer block, unauthorized network | Immediate halt | Log → ALERT → no auto-retry |
| `TAIS_THROTTLE` | GPU core temp 75–80°C | Initiate Q4_K_M reload (30–90s) | Resume at Q4_K_M |
| `TAIS_EMERGENCY` | GPU core temp > 83°C | Checkpoint → halt | Resume only: temp < 72°C + user confirmation |
| `TAIS_SENSOR_FAIL` | pynvml read failure | Assume THROTTLE, switch Q4_K_M | ALERT → retry sensor every 30s → restore on recovery |
| `WATCHDOG_FAIL` | Heartbeat timeout > 90s | Suspend autonomous mode | Auto-restart × 3 → full halt if unrecoverable |

---

## Appendix C — IPC Channel Registry

All inter-process communication uses Redis pub/sub. Channels are subscribe-only for consumers — no persistent storage.

| Channel | Publisher | Subscribers | Payload |
|---|---|---|---|
| `ipc:tais:status` | TAIS sidecar | Main agent, Continuity Engine | `{status, temp_c, quantization, timestamp}` |
| `ipc:watchdog:monitor` | Main agent | Watchdog | Full inference output text + session_id + task_id |
| `ipc:watchdog:verdict` | Watchdog | Main agent | `{verdict: CLEAR/FLAGGED, reason, timestamp}` |
| `ipc:memory:route` | Main agent | Memory Router (MR-1) | `{data_hash, content_type, content, session_id}` |
| `ipc:isec:status` | ISEC daemon | Main agent, Observability | `{pass, nodes_reviewed, nodes_promoted, duration}` |
| `ipc:continuity:trigger` | Main agent, TAIS | Continuity Engine | `{trigger_type: CHECKPOINT/EMERGENCY/SHUTDOWN}` |
| `ipc:health:status` | Continuity Engine | Main agent | `{component, status, timestamp}` |

---

## Appendix D — Port Allocation

| Service | Port | Protocol | Notes |
|---|---|---|---|
| Ollama | 11434 | HTTP | Model inference API |
| Redis (L1 / ASD) | 6379 | TCP | Primary APM L1 |
| Qdrant REST | 6333 | HTTP | L2 vector search |
| Qdrant gRPC | 6334 | gRPC | L2 high-performance |
| FalkorDB | 6380 | TCP | L3 graph (offset from Redis) |
| Metrics endpoint | 9001 | HTTP | Prometheus `/metrics` |

All ports are localhost-only. No external binding. WSL2 port forwarding should be configured for `6379`, `6333`, `6380`, and `9001` if accessing from Windows host tools.

---

## Appendix E — Benchmark Suite

Used for model upgrade evaluation (Section 10.1) and Phase 1 health validation.

**10 Standardized Tasks across 5 categories:**

| # | Category | Task Description | Pass Criteria |
|---|---|---|---|
| 1 | Reasoning | Multi-step logic problem with 3 dependent sub-conclusions | Correct answer, < 2 correction cycles |
| 2 | Reasoning | Identify the contradiction in a provided argument | Contradiction correctly identified with explanation |
| 3 | Tool Use | Search L2 for a known-indexed topic, return top-3 results | 3 relevant results returned, correct collection queried |
| 4 | Tool Use | Execute Python snippet in sandbox, return stdout | Correct output, no sandbox escape attempt |
| 5 | Memory Retrieval | Retrieve a past mission post-mortem by approximate description | Correct mission matched via hybrid search |
| 6 | Self-Correction | Intentionally malformed tool call → expect recovery | Self-corrects within 3 cycles, correct result delivered |
| 7 | TAIS Behavior | Run inference under simulated thermal pressure | Correct quant switch triggered at threshold |
| 8 | Context Compression | Task requiring > 8,000 tokens of intermediate work | Correct result with context budget not exceeded |
| 9 | ASD Integrity | Multi-step task — inspect ASD state at each step | All state transitions correct, schema valid throughout |
| 10 | Synthesis & Delivery | Full end-to-end task: research → synthesize → deliver | DELIVERED status, confidence > 0.80, all sub-tasks verified |

**Scoring:** Each task is pass/fail. Upgrade threshold: new model must score ≥ 9/10 AND outperform current model by > 5% on overall timing.

---

## Appendix F — Version History

| Version | Changes |
|---|---|
| **1.0** | Core framework defined |
| **2.0** | Full architectural expansion |
| **2.1** | TAIS, ASD, Precision Planner Mode, ISEC, Zero-Trust Sanitizer, S score formula, Skill Forge gate upgrade, tool versioning, session identity, Watchdog liveness, Gate timeout ladder, log rotation, MCP accounting, FalkorDB snapshots, VLA Phase 4 assignment, failure taxonomy, cold-start WSL2 trigger |
| **2.2** | **All 35 audit findings resolved.** Critical: TAIS quant switch latency acknowledged (30–90s reload sequence); `pynvml` returns GPU core temp (not junction) — validation step added; Python sandbox changed from Wasm to subprocess isolation (Docker/firejail); ASD shadow write contradiction fixed (sync Redis → async FalkorDB queue, gap documented); `llava:13b` removed from 6GB tier (moondream2 only); tool_results TTL changed to session-bound; `(:Session)` node defined in graph schema. Significant: Context window capped at 87.5% (28,672 tokens) with 4,096-token safety buffer; ASD schema adds `task_id` and `session_id` (non-nullable); ASD status enum expanded to all 9 lifecycle states; `CHECKPOINTED` replaced with `CHECKPOINT_SAVED` event; Continuity Engine formally defined as Python asyncio daemon; S score R normalization defined as rolling 90-day window; Forge 30-minute circuit breaker added; trivial task definition formalized in Section 1.4; Watchdog IPC defined via Redis pub/sub; Tool/Skill duality resolved (Tool = native, Skill = synthesized); VLA Red Zone removed from output schema; ISEC VRAM/TAIS pre-check added; parallel inference clarified (orchestration parallel, inference queued). Missing: startup/shutdown sequences; port allocation table (Appendix D); IPC channel registry (Appendix C); `TAIS_SENSOR_FAIL` failure class; audit trail physical path + retention policy; `failure_analysis` document schema; verbatim size cap (2,048 tokens with L1 reference); benchmark suite (Appendix E); CLI notification mapping. Consistency: `manifest_version` + `created_at` added to Mission Manifest; metrics format specified (Prometheus); VRAM alert independent of TAIS; `EXPERIMENTAL` 30-day promotion review; MR-1 bypass documented. |

---

*Aura-9 Specification — Version 2.2 | Canonical Development Blueprint | All 35 v2.1 audit findings resolved.*

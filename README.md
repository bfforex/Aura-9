# Aura-9: Autonomous Reasoning Agent
## Official Technical Specification
**Version:** 2.1 | **Status:** Active Development Blueprint | **Classification:** Internal Reference

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
14. [Appendix C — Version History](#appendix-c--version-history)

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

TAIS is a hardware telemetry hook that runs as a continuous sidecar process alongside the inference engine. It monitors GPU junction temperature via `pynvml` and dynamically adjusts the inference workload to protect local hardware during sustained autonomous operation.

**Polling Interval:**
- Active inference: every **5 seconds**
- Idle / cooldown: every **30 seconds**

**Temperature Threshold Ladder:**

| Junction Temp | Status | Action |
|---|---|---|
| `< 75°C` | Normal | Full operation — Q5_K_M quantization |
| `75–80°C` | Throttle | Shift to Q4_K_M, log `TAIS_THROTTLE` event |
| `80–83°C` | Cooldown | Pause inference queue, drain active call, resume when temp drops below 72°C |
| `> 83°C` | Emergency | Immediate checkpoint → halt inference → `CRITICAL` alert to user → await explicit confirmation before resuming |

**Recovery Confirmation:**
TAIS does not resume on a timer. After a Cooldown or Emergency halt, resumption is **temperature-confirmed**: inference restarts only when GPU junction temperature drops back below **72°C** (a 3°C safety buffer below the throttle threshold). This is polled on the 30-second idle interval.

**TAIS Telemetry (exposed to metrics endpoint):**
- `tais_current_temp_celsius`
- `tais_status` — `NORMAL | THROTTLE | COOLDOWN | EMERGENCY`
- `tais_throttle_events_total`
- `tais_emergency_halts_total`
- `tais_active_quantization` — current model quant level

### 2.3 Chain-of-Thought (CoT) + Recursive Self-Correction

Every non-trivial task goes through a **4-Phase Reasoning Cycle:**

#### Phase 1 — Decomposition
The raw task is parsed by the **Prompt Enhancement Pre-Processor** into a structured **Mission Manifest**:

```yaml
mission_manifest:
  task_id: "uuid-v4"
  session_id: "sess-uuid-timestamp"
  original_intent: "[raw user input]"
  interpreted_goal: "[expanded, KPI-tagged goal]"
  sub_tasks:
    - id: ST-001
      description: "..."
      success_criteria: "..."
      tools_required: ["qdrant_search", "python_exec"]
      estimated_complexity: 0.6     # 0.0–1.0
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
4. If 3 corrections all fail → escalate with full failure context including failure class, all attempted approaches, and the recommended recovery action.

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
| Raw tool outputs (backtest logs, JSON API responses) | **Retained verbatim** — must not be summarized |
| Completed CoT reasoning branches | **Collapsed** to single-line micro-summary |
| Confirmed sub-task results | **Compressed** to outcome + confidence score |
| Active sub-task scratchpad | **Full fidelity** — never compressed |
| Superseded plans / abandoned branches | **Pruned** entirely |

**Micro-Summary Format:**
```
[ST-001 ✓ 0.91] Searched Qdrant expertise collection → 4 relevant chunks retrieved on topic "API retry logic"
```

This preserves the reasoning audit trail without consuming context tokens on resolved work.

### 2.5 Context Window Management

The **Context Budget Manager** allocates the 32,768-token window dynamically:

| Allocation | Default Budget | Notes |
|---|---|---|
| System Prompt + Identity | 2,048 | Static |
| Active Mission Manifest | 1,024 | Dynamic |
| ASD State Injection | 512 | Current task state from Redis |
| L1 Episodic Memory Injection | 4,096 | Rolling compressed window |
| L2 Semantic Retrieval Chunks | 8,192 | Top-k RAG results |
| L3 Graph Context | 2,048 | Relevant entity subgraph |
| Working Scratchpad (CoT) | 11,776 | Reasoning + tool calls |
| Output Buffer | 3,072 | Final response assembly |

When any allocation approaches its limit, a **Memory Compression Pass** is triggered — collapsing older scratchpad content via State-Driven Context Compression before re-injection.

### 2.6 Precision Planner Mode

Precision Planner Mode is a secondary system instruction set that forces the model into a **JSON-exclusive output mode**. It is invoked strictly for Aura State Daemon (ASD) updates, completely separating high-level planning logic from tool-execution logic.

**Invocation:** Injected as a system prompt override when an ASD write is required.

**Locked Output Schema:**
```json
{
  "asd_update": {
    "current_objective": "string — active mission goal",
    "status": "EXECUTING | PAUSED | BLOCKED | CORRECTING | SUSPENDED",
    "active_subtasks": ["ST-001", "ST-002"],
    "completed_subtasks": ["ST-000"],
    "blocked_by": "string | null",
    "confidence": 0.0,
    "next_action": "string — exact planned next step",
    "failure_class": "string | null",
    "checkpoint_required": true,
    "tais_status": "NORMAL | THROTTLE | COOLDOWN | EMERGENCY"
  }
}
```

**Rules:**
- No natural language outside this JSON structure is permitted in Precision Planner Mode.
- The schema is immutable — the model is not permitted to add, remove, or rename fields.
- Any output that fails JSON validation is rejected and the model is re-prompted once. If the second attempt also fails → `STATE_FAIL` is raised (see Appendix B).

---

## 3. Triple-Thread Memory System (APM)

The APM (Autonomous Persistence Module) gives Aura-9 continuity across sessions, reboots, and extended autonomous operation. It is not a monolithic memory — it is three distinct systems with different purposes, retention policies, and access patterns, working in concert.

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
│          [Decides what gets stored where]               │
│                            │                            │
│                  ISEC Daemon (background)               │
│          [Idle-state consolidation & pruning]           │
└──────────────────────────────────────────────────────────┘
```

### 3.1 Session Identity

Every session is assigned a **Session ID** at startup, stamped on all data written during that session:

```
session_id format: sess-{uuid4}-{unix_timestamp}
example:           sess-a3f2c1d0-...-1744320000
```

This ID is stamped on:
- Every L1 Redis key created during the session
- Every L2 Qdrant payload object
- Every FalkorDB node created during the session
- Every audit trail entry

This allows ISEC, the Memory Router, and the audit trail to correctly attribute patterns to specific sessions, and makes cross-session debugging deterministic.

### 3.2 L1 — Episodic Memory + ASD State (Redis)

**Purpose:** High-speed working context for the active session and live Aura State Daemon state.

| Property | Value |
|---|---|
| **Technology** | Redis 7.x |
| **Retention** | Session-bound — selectively promoted before purge |
| **Access Latency** | < 1ms |
| **Max Capacity** | 512MB (configurable) |

**Key Namespaces:**

| Namespace | Content | TTL |
|---|---|---|
| `sess:{session_id}:turns` | Conversation turns (Redis Stream) | 30 min |
| `sess:{session_id}:tool_results` | Raw tool outputs | 5 min |
| `sess:{session_id}:scratchpad` | Active CoT state | Session-bound |
| `asd:state` | ASD live state tree (Precision Planner JSON) | No TTL |
| `asd:checkpoint:{task_id}` | Point-in-time ASD snapshots | 7 days |
| `mcp:calls:{server_id}:{date}` | MCP call counter per server per day | 48 hours |

**Promotion Criteria (L1 → L2):**
- Task produced reusable factual knowledge.
- User explicitly marked output as important.
- Conclusion confidence score > 0.85.

### 3.3 L2 — Semantic Memory (Qdrant)

**Purpose:** Permanent expertise retrieval. Aura-9's long-term knowledge layer.

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
| `skill_library` | Synthesized tool blueprints and metadata |
| `past_missions` | Compressed mission post-mortems |
| `failure_analysis` | Structured post-mortems for failed/escalated tasks |

**Retrieval Strategy — Hybrid Search:**
Dense vector similarity is combined with sparse BM25 keyword scoring using Reciprocal Rank Fusion (RRF). This prevents semantically similar but contextually wrong chunks from polluting the context window.

**Significance Score Decay (L2):**

```
S = (R × F) + £

R  = Retrieval Frequency   (0.0–1.0, normalized against session count)
F  = Finality              (0.0–1.0, where 1.0 = contributed to a DELIVERED task)
£  = User Override weight  (0.0–1.0, default 0.0; set to 1.0 via `aura9 memory pin`)

Score range: 0.0–2.0
Archival threshold: S < 0.2
```

**Decay Behavior:**
- Score is recalculated after every retrieval event.
- Successful retrieval that contributes to task delivery: `F` boosted to `1.0` for that event.
- `£ = 1.0` renders a node effectively immortal regardless of R and F.
- Nodes below archival threshold are flagged — not automatically deleted. ISEC handles archival.

**Pinning a node:**
```bash
aura9 memory pin <node_id>         # Sets £ = 1.0 (immutable)
aura9 memory unpin <node_id>       # Resets £ = 0.0
aura9 memory score <node_id>       # Inspect current S score
```

### 3.4 L3 — Relational Memory (FalkorDB)

**Purpose:** The structural map of everything Aura-9 knows — how entities connect, how tasks relate, and what dependencies exist.

| Property | Value |
|---|---|
| **Technology** | FalkorDB (Graph Database) |
| **Retention** | Permanent with Significance Score decay |
| **Query Language** | Cypher |
| **Access Latency** | 2–20ms |

**Core Graph Schema:**

```cypher
// Node types
(:Project {id, name, session_id, created_at})
(:Task {id, status, session_id, created_at, significance_score})
(:Tool {id, name, version, maturity, session_id})
(:Skill {id, name, version, maturity, trust_level})
(:File {id, path, type, session_id})
(:Entity {id, name, type, session_id})
(:StateNode {task_id, checkpoint_id, timestamp, status, redis_snapshot_key, next_action})
(:MemoryNode {id, session_id, tier, significance_score, user_pinned})

// Relationships
(Project)-[:HAS_TASK]->(Task)
(Task)-[:DEPENDS_ON]->(Task)
(Task)-[:USED_TOOL]->(Tool)
(Task)-[:PRODUCED]->(File)
(Task)-[:HAS_CHECKPOINT]->(StateNode)
(Agent)-[:HAS_SKILL]->(Skill)
(Skill)-[:SOLVES]->(Task)
(Tool)-[:VERSION_OF]->(Tool)
(MemoryNode)-[:BELONGS_TO_SESSION]->(Session)
```

**Significance Score Decay (L3):**
Every `Task`, `MemoryNode`, and `Skill` node in the graph carries a `significance_score` computed using the same `S = (R × F) + £` formula as L2. Nodes below `S < 0.2` are flagged for archival to cold storage (compressed `.dump` file, dated and retained indefinitely).

**ASD Shadow Write:**
The FalkorDB graph maintains a shadow copy of the ASD state tree for survivability:

```
Every ASD state change:
  ├─ Redis write (primary — synchronous, < 1ms)
  └─ FalkorDB write (shadow — async, fires before Redis ACK)

On Redis failure:
  ├─ FalkorDB shadow becomes authoritative source of truth
  ├─ MEMORY_FAIL ALERT triggered
  ├─ Redis recovery initiated
  └─ On Redis restore → re-sync from FalkorDB shadow, resume normal operation
```

**Daily Snapshot (Rollback Protection):**
FalkorDB graph is exported to a dated snapshot file every 24 hours:
```bash
# Auto-triggered by the Continuity Engine
aura9 graph snapshot --output ./backups/falkordb-{date}.dump

# Manual restore to known-good state
aura9 graph restore --file ./backups/falkordb-2026-04-10.dump
```

Snapshots are retained for **30 days** before archival.

### 3.5 Memory Router (MR-1)

The Memory Router is the routing layer between all three tiers. It runs as a lightweight sidecar and makes real-time decisions on every piece of data produced by the inference engine or tools:

```
Incoming data
      │
      ▼
[Is it live state or ephemeral working data?]
  YES → L1 Redis (session namespace)
  NO  → [Does it contain factual knowledge or reusable insight?]
             YES → L2 Qdrant (appropriate collection)
                   + extract entities → L3 FalkorDB
             NO  → [Does it define a relationship between entities?]
                        YES → L3 FalkorDB only
                        NO  → Discard (log to audit trail)
```

All routing decisions are logged to the audit trail with the data hash, destination tier, and session ID.

### 3.6 ISEC — Idle-State Epistemic Consolidation

ISEC is a background daemon that activates when the GPU has been idle for more than **5 minutes**. It performs a multi-pass review to consolidate, promote, and prune memory — improving Aura-9's knowledge quality without interrupting active tasks.

**Activation Condition:**
```
GPU idle > 5 minutes AND no active task in ASD state
```

**ISEC Multi-Pass Pipeline:**

```
Pass 1 — L1 Review
  Read all session logs in Redis older than current session
  Identify: recurring patterns, established logic, reusable conclusions
  Tag candidates for promotion

Pass 2 — L2 Deduplication
  Find Qdrant vectors with cosine similarity > 0.97
  Merge duplicates → retain highest-confidence version
  Recalculate S scores for affected nodes

Pass 3 — L3 Enrichment
  For each promoted L2 node → ensure entity nodes + edges exist in FalkorDB
  Connect new nodes to existing task/project graph

Pass 4 — Decay Audit
  Recalculate S scores across all L2 + L3 nodes
  Flag nodes with S < 0.2 for archival
  Archive flagged nodes to cold storage (never delete)

Pass 5 — L1 Pruning
  Purge reviewed L1 session logs that have been promoted or discarded
  Reset Redis TTLs for retained working data
```

**ISEC Log:**
Every ISEC pass is logged with: pass number, nodes reviewed, nodes promoted, nodes merged, nodes flagged for archival, duration, GPU temperature at start/end.

---

## 4. Capability & Tooling — The Skill Forge

### 4.1 Tooling Architecture

```
┌─────────────────────────────────────────────────────┐
│                   SKILL FORGE                       │
│                                                     │
│  ┌─────────────────┐    ┌──────────────────────┐    │
│  │  Native Tools   │    │  Synthesized Tools   │    │
│  │  (Hardcoded)    │    │  (Self-Generated)    │    │
│  └─────────────────┘    └──────────────────────┘    │
│           │                        │                │
│           └───────────┬────────────┘                │
│                       ▼                             │
│                Tool Registry                        │
│           (Qdrant `skill_library`)                  │
│                       │                             │
│                       ▼                             │
│             Tool Selection Engine                   │
│       (Hybrid search + dependency check)            │
│                       │                             │
│                       ▼                             │
│          Synthetic Environmental Backtester         │
│       (100% ground-truth validation gate)           │
└─────────────────────────────────────────────────────┘
```

### 4.2 Native Tool Suite

#### 4.2.1 VLA Bridge — Vision-Language-Action Module

The VLA Bridge gives Aura-9 the ability to perceive and interact with graphical interfaces. It is assigned to **Phase 4** alongside the MCP Gateway and integrates with the Red Zone system established in Phase 5.

| Property | Value |
|---|---|
| **Function** | Screen parsing + UI interaction |
| **Snapshot Interval** | 60 seconds (24/7 monitoring mode) |
| **On-Demand** | Immediate capture on explicit tool call |
| **Vision Model** | `llava:13b` or `moondream2` (local) |
| **Output Schema** | `{elements: [], clickable_zones: [], red_zones: [], text_content: "", session_id: ""}` |

**24/7 Monitoring Flow:**
```
Every 60s:
  1. Capture screenshot
  2. Parse via vision model → structured JSON
  3. Compare against expected state from last checkpoint
  4. If drift detected → log anomaly
  5. If critical anomaly (error dialog, unexpected application state) →
       interrupt current task → address → resume from checkpoint
```

**Red Zone Integration:**
Before any click action, the VLA Bridge checks the target coordinate against the Red Zone Registry. Clicks targeting Red Zones are intercepted at the action layer — the 9B model never receives confirmation that the click was attempted.

#### 4.2.2 MCP Gateway — Zero-Trust External Access

| Property | Value |
|---|---|
| **Protocol** | Model Context Protocol (MCP) |
| **Authentication** | Per-server OAuth 2.0 / API Key vault |
| **Logging** | All calls logged to audit trail with request hash |
| **Timeout** | 30s default, 120s for bulk operations |
| **Call Accounting** | Per-server daily counter in Redis (`mcp:calls:{server_id}:{date}`) |

**Access Tiers:**

| Tier | Scope | Approval |
|---|---|---|
| Tier 1 | Read-only, public data | Auto-approved |
| Tier 2 | Write access to designated workspaces | Session-approved |
| Tier 3 | Destructive or irreversible actions | Human gate required |

**MCP Call Accounting:**
A lightweight daily counter is maintained per server in Redis. If a server's daily call count exceeds a configurable threshold, subsequent calls trigger an `ALERT` before proceeding. This prevents silent rate-limit failures mid-task.

```bash
aura9 mcp stats                         # View call counts per server today
aura9 mcp set-limit <server_id> <n>     # Set daily call threshold
```

**Server Trust Levels:**
- `TRUSTED` — User-configured, cryptographically verified.
- `COMMUNITY` — Public MCP registry. Read-only by default.
- `UNTRUSTED` — Requires explicit user approval per session.

#### 4.2.3 Code Interpreter & Wasm Sandbox

| Property | Value |
|---|---|
| **Languages** | Python 3.12+, JavaScript (Wasm), Bash (restricted) |
| **Isolation** | WebAssembly sandbox — no host OS access |
| **Network** | Blocked by default; Tier-2 approval required |
| **File System** | Ephemeral virtual FS — manual promotion to persistent storage |
| **Memory Limit** | 512MB per execution |
| **CPU Timeout** | 120s hard limit |

**Approved Packages:**
- Data: `numpy`, `pandas`, `polars`
- Text: `re`, `json`, `yaml`, `markdown`
- HTTP (Tier-2 only): `httpx`, `requests`
- Crypto: `cryptography`, `hashlib`
- Files: `openpyxl`, `pypdf`, `pillow`

Any import not on the whitelist causes immediate sandbox termination and a Watchdog alert.

#### 4.2.4 Prompt Enhancement Pre-Processor

Every user input passes through the Pre-Processor before reaching the 9B model. This stage is mandatory.

**Processing Steps:**
1. Intent extraction — what does the user actually want?
2. Ambiguity detection — are there unclear parameters?
3. KPI injection — what does "done" look like?
4. Constraint mapping — time, tools, security tier, escalation threshold.
5. Session ID stamping.
6. Mission Manifest assembly (YAML output).

**Clarification Policy:**
If ambiguity score > 0.60, the Pre-Processor generates one targeted clarification question — resolving as many ambiguities as possible from context before surfacing the single most critical one.

### 4.3 Autonomous Tool Synthesis — The Forge Process

```
┌─────────────────────────────────────────────────────────┐
│                    FORGE PROCESS                        │
│                                                         │
│  1. Tool Gap Detected                                   │
│     └─ No registry match above similarity 0.75         │
│                                                         │
│  2. Tool Blueprint Generation                           │
│     └─ 9B drafts Python/Wasm implementation            │
│     └─ Assigned version tag: tool_id@v1               │
│                                                         │
│  3. Synthetic Environmental Backtesting                 │
│     └─ Execute in Wasm sandbox against domain          │
│        data stream (historical market ticks,           │
│        power-load scenarios, or generic test vectors)  │
│     └─ Must achieve 100% ground-truth validation       │
│        across all test cases to proceed                │
│                                                         │
│  4. Watchdog Review                                     │
│     └─ Audit for unsafe patterns:                      │
│        network calls, file writes, banned imports      │
│                                                         │
│  5. Promotion Decision                                  │
│     ├─ Auto-promote: 100% validation + safe audit      │
│     └─ Gate-required: any Tier-2+ operations           │
│                                                         │
│  6. Registry Entry                                      │
│     └─ Tool stored in Qdrant `skill_library`           │
│     └─ Node added: (Tool {version: "v1"})-[:SOLVES]    │
│     └─ Status: EXPERIMENTAL                            │
└─────────────────────────────────────────────────────────┘
```

### 4.4 Tool Versioning

Every synthesized tool carries a version tag. Old versions are never silently overwritten.

| Version Event | Behavior |
|---|---|
| New synthesis | `tool_id@v1` — status: `EXPERIMENTAL` |
| Patch or improvement | `tool_id@v2` — previous version set to `DEPRECATED` |
| `DEPRECATED` retention | 7 days before archival |
| Rollback | `aura9 tool rollback <tool_id> --to v1` |
| Quarantine | Automatic after 3 task failures — requires manual inspection to restore |

**Tool Maturity Levels:**

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

The ASD is a persistent JSON state tree maintained in Redis that serves as the **absolute source of truth** for Aura-9's current mission. It is updated exclusively via Precision Planner Mode to prevent state corruption from natural language output.

**Live State Location:** `asd:state` (Redis, no TTL)
**Shadow Copy:** FalkorDB `StateNode` graph (async, every write)
**Point-in-Time Snapshots:** `asd:checkpoint:{task_id}` (Redis, 7-day TTL)

**State Tree Structure:**
```json
{
  "asd_update": {
    "current_objective": "string",
    "status": "EXECUTING | PAUSED | BLOCKED | CORRECTING | SUSPENDED",
    "active_subtasks": ["ST-001"],
    "completed_subtasks": ["ST-000"],
    "blocked_by": null,
    "confidence": 0.87,
    "next_action": "string — exact planned next step",
    "failure_class": null,
    "checkpoint_required": false,
    "tais_status": "NORMAL"
  }
}
```

### 5.2 Task Lifecycle

```
CREATED → PLANNED → EXECUTING → [CORRECTING] → VERIFYING → DELIVERED
                                      ↑                |
                                      └────────────────┘
                                   (up to 3 cycles, class-aware)

Additional states:
  PAUSED      — Human gate encountered, awaiting input
  SUSPENDED   — Human gate timeout exceeded (>120 min), resources released
  CHECKPOINTED — State saved mid-execution
  ESCALATED   — Correction limit exceeded, handed to user
  FAILED      — Unrecoverable error, full diagnostic generated
```

### 5.3 Parallel Task Execution (DAG Scheduler)

Independent sub-tasks are executed in parallel using a Directed Acyclic Graph (DAG) scheduler:

```python
task_graph = {
    "ST-001": [],                    # No dependencies — runs immediately
    "ST-002": [],                    # No dependencies — runs immediately
    "ST-003": ["ST-001"],            # Waits for ST-001
    "ST-004": ["ST-001", "ST-002"],  # Waits for both
    "ST-005": ["ST-003", "ST-004"],  # Final synthesis
}
```

**Concurrency Limits:**
- Default max parallel tasks: **3** (to avoid VRAM contention with TAIS overhead)
- Shared tool access is mutex-locked (e.g., only one task uses the VLA Bridge at a time)
- TAIS throttle events automatically reduce max concurrency to **2** during active throttling

### 5.4 Human Gate Protocol

When Aura-9 determines human input is required, it executes a **Graceful Pause**:

1. Complete the current atomic action (no tools left mid-execution).
2. Write ASD state → checkpoint to Redis + FalkorDB shadow.
3. Generate a **Gate Brief**:
   - What has been completed.
   - What the specific decision point is.
   - Available options with Aura-9's recommended choice and reasoning.
   - What happens after the user responds.
4. Surface the Gate Brief to the user.
5. Enter `PAUSED` state — resource footprint drops to idle.

**Gate Timeout Ladder:**

| Elapsed Time | State | Action |
|---|---|---|
| `0–30 min` | `PAUSED` | Idle, low resource, awaiting response |
| `30–120 min` | `PAUSED` | Re-notify user with Gate Brief summary |
| `> 120 min` | `SUSPENDED` | Full checkpoint → release GPU resources → await next user interaction |
| On user return | Recovery | Read ASD from FalkorDB shadow → restore Redis → resume |

### 5.5 24/7 Continuity Engine

**Checkpointing (every 15 minutes during active execution):**
```yaml
checkpoint:
  task_id: "uuid"
  session_id: "sess-uuid-timestamp"
  checkpoint_id: "ckpt-042"
  timestamp: "ISO-8601"
  completed_subtasks: [ST-001, ST-002]
  active_subtask: ST-003
  active_subtask_step: 4
  asd_state_snapshot: "{...}"
  tais_status: "NORMAL"
  next_action: "Execute qdrant_search with query: '...'"
```

**Cold-Start Resumption (WSL2 Implementation):**

On WSL2, the cold-start trigger is implemented via **Windows Task Scheduler** calling a WSL2 shell script on user login:

```
Trigger:    Windows login event
Action:     wsl.exe --exec /home/user/aura9/scripts/startup.sh
Script:     startup.sh → activate venv → python main.py --resume
```

`main.py --resume` startup sequence:
```
1. Check FalkorDB for StateNode with status = "EXECUTING" or "PAUSED"
2. If found → restore Redis ASD from FalkorDB shadow
3. Restore L1 session snapshot from checkpoint
4. Log interruption duration to audit trail
5. Resume from `next_action` in checkpoint
6. No user action required
```

If no incomplete checkpoint is found → normal interactive startup.

**Graceful Degradation:**
If a dependency goes offline mid-task:

```
Tool unavailable:
  → Check registry for alternative tool
  → If alternative found: switch silently, log substitution
  → If no alternative: checkpoint → PAUSED → ALERT user

Memory tier unavailable:
  → Redis down: promote FalkorDB shadow to primary, ALERT
  → Qdrant down: skip L2 retrieval, note degraded context, continue
  → FalkorDB down: disable shadow writes, ALERT, continue on Redis only
```

---

## 6. Security & Operational Guardrails

### 6.1 The Watchdog — Secondary Oversight Model

The Watchdog is a permanently running read-only audit process monitoring the 9B's outputs for drift, loops, and capability violations.

| Property | Value |
|---|---|
| **Model** | `Qwen 2.5 1.5B` or `Phi-3 Mini` |
| **VRAM Footprint** | ~0.8GB–1.5GB |
| **Monitoring Mode** | Continuous — every 9B output evaluated |
| **Config Storage** | Read-only filesystem partition — 9B has no write access |

**What the Watchdog Monitors:**
- **Drift:** Is Aura-9 still working on the assigned task?
- **Loop Detection:** Same action attempted > 5 times with identical inputs.
- **Gate Bypass:** Attempting to proceed past a required human gate.
- **Capability Creep:** Requesting tool access above current permission tier.
- **Output Toxicity:** Content flagged by safety classifier.
- **Schema Violations:** ASD updates that fail the Precision Planner JSON schema.

**Response to Anomaly:**
```
Flag → Log to audit trail → Notify user (ALERT level)
```
The Watchdog does **not** auto-kill the 9B except in one case:

**Hard Kill Authority:** Confirmed infinite loop — defined as > 50 identical consecutive actions with identical inputs. Hard kill writes a final checkpoint before halting.

### 6.2 Watchdog Liveness Check

The Watchdog itself is monitored. If the Watchdog process fails to emit a heartbeat within **90 seconds**:

```
1. CRITICAL alert issued to user
2. Autonomous execution mode suspended immediately
3. 9B continues in interactive-only mode (no autonomous tool execution)
4. Watchdog restart attempted automatically (up to 3 times)
5. If restart fails → full halt, user must manually restart
6. Autonomous mode does not resume until Watchdog heartbeat confirmed
```

**Watchdog heartbeat key in Redis:** `watchdog:heartbeat` (TTL: 90 seconds, refreshed every 30s by Watchdog process). If the key expires, the liveness check triggers.

### 6.3 Zero-Trust Sanitizer

The Zero-Trust Sanitizer is a **pre-execution payload interceptor** that operates before any outbound tool call or MCP action. It is distinct from the MCP Gateway's request-level checks — it operates at the payload level.

**Scanning Targets:**
- Hardcoded filesystem paths (absolute paths not in the approved workspace)
- API keys or tokens embedded in payloads (regex: key/token/secret patterns)
- Unauthorized network addresses (IPs or domains not in the MCP registry)
- Shell injection patterns in string parameters
- Attempts to write outside the designated output directories

**Response Actions:**

| Finding | Action |
|---|---|
| Hardcoded path | Replace with workspace-relative path, log warning |
| Embedded API key | **Block execution**, raise `SECURITY_FAIL`, alert user |
| Unauthorized network address | **Block execution**, raise `SECURITY_FAIL`, alert user |
| Shell injection pattern | **Block execution**, raise `SECURITY_FAIL`, alert user |
| Out-of-bounds write path | Replace with safe placeholder, log warning |

All sanitizer events — both warnings and blocks — are written to the immutable audit trail.

### 6.4 Wasm Sandboxing

All code executed by Aura-9 — native or synthesized — runs inside a WebAssembly sandbox:

| Constraint | Policy |
|---|---|
| **Network Access** | Blocked (Tier-2 approval required) |
| **Host Filesystem** | No access (virtual ephemeral FS only) |
| **System Calls** | Restricted — no fork, exec, socket by default |
| **Memory Limit** | 512MB per execution |
| **CPU Time** | 120s hard wall |
| **Import Whitelist** | Approved packages only — see Section 4.2.3 |

Any import not on the whitelist causes immediate termination and a Watchdog alert.

### 6.5 Visual Red Zone System

Red Zones are screen areas that Aura-9's VLA Bridge cannot interact with, enforced below the agent's awareness.

**Default Red Zones:**
- System Settings / Control Panel
- Package Manager (apt, brew, winget, pip with `--system`)
- Network Configuration dialogs
- Authentication / Password prompts
- Any dialog containing: "Uninstall," "Delete All," "Factory Reset," "Format"

**Custom Red Zone Config:**
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

**Zone Action Types:**

| Type | Behavior |
|---|---|
| `BLOCK` | Silent prevention. 9B told the action failed — does not know why. |
| `GATE` | Pause → user confirmation required → proceed or cancel. |
| `ALERT` | Action allowed but flagged to user in real time. |

### 6.6 Audit Trail

Every action Aura-9 takes is logged to an **immutable append-only audit trail**, stored outside the agent's writable scope:

| Field | Description |
|---|---|
| `event_id` | UUID |
| `timestamp` | ISO-8601 with microseconds |
| `session_id` | Parent session |
| `task_id` | Parent mission |
| `action_type` | `tool_call`, `llm_inference`, `memory_read`, `memory_write`, `checkpoint`, `gate_triggered`, `watchdog_alert`, `tais_event`, `sanitizer_event`, `isec_pass` |
| `action_detail` | Full parameters (PII-scrubbed) |
| `result_hash` | SHA-256 of the output |
| `confidence_score` | 0.0–1.0 |
| `failure_class` | From taxonomy (null if no failure) |
| `watchdog_status` | `CLEAR` / `FLAGGED` |
| `tais_status` | Active TAIS state at time of event |

---

## 7. Infrastructure & Hardware

### 7.1 Runtime Stack

```
┌──────────────────────────────────────────────────────────┐
│                     HOST SYSTEM                          │
│                  (Windows 11 / WSL2)                     │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │              Ollama (Local Runtime)              │    │
│  │  ┌──────────────────┐  ┌──────────────────────┐  │    │
│  │  │  Qwen 3.5 9B     │  │  Watchdog 1.5B–3B    │  │    │
│  │  │  (Q5_K_M)        │  │  (Q4_K_M)            │  │    │
│  │  └──────────────────┘  └──────────────────────┘  │    │
│  └──────────────────────────────────────────────────┘    │
│                          │                               │
│              TAIS (pynvml telemetry hook)                │
│                          │                               │
│  ┌──────────┐  ┌─────────┴┐  ┌──────────────────────┐   │
│  │  Redis   │  │  Qdrant  │  │      FalkorDB        │   │
│  │  L1+ASD  │  │   L2     │  │    L3 + ASD Shadow   │   │
│  └──────────┘  └──────────┘  └──────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │         MCP Gateway + VLA Bridge + Wasm          │    │
│  │         Zero-Trust Sanitizer (pre-execution)     │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### 7.2 Hardware Requirements

#### Minimum Viable (6GB VRAM)

| Component | Specification |
|---|---|
| **GPU** | NVIDIA RTX 3060 / RTX 4060 / equivalent |
| **VRAM Budget** | 5.5GB (9B Q5_K_M) + Watchdog on CPU = **~5.5GB** |
| **System RAM** | 16GB minimum (32GB recommended) |
| **Storage** | 50GB NVMe SSD |
| **OS** | Windows 11 + WSL2 (Ubuntu 22.04+) |

> **Note:** At 6GB VRAM, the Watchdog runs on CPU in Phase 1–4. Phase 5 implements proper co-scheduling. Watchdog CPU inference latency is acceptable for its monitoring role (not in the inference critical path).

#### Recommended (8GB+ VRAM)

| Component | Value |
|---|---|
| **VRAM Budget** | 5.5GB (9B Q5_K_M) + 1.2GB (Watchdog 3B) = **~6.7GB** |
| **Benefit** | Both models GPU-resident, Watchdog real-time latency |

#### Performance Tier (16GB+ VRAM)

| Component | Value |
|---|---|
| **VRAM Budget** | ~9.2GB (9B Q8_0) + Watchdog + Vision model simultaneously |
| **Benefit** | Near-lossless quantization, VLA Bridge vision model always loaded |

### 7.3 Quantization Strategy

| Quantization | VRAM | Quality | Use Case |
|---|---|---|---|
| `Q4_K_M` | ~4.8GB | Good | TAIS throttle fallback |
| `Q5_K_M` | ~5.5GB | Very Good | **Default — 6GB GPU** |
| `Q6_K` | ~6.2GB | Excellent | 8GB GPU |
| `Q8_0` | ~9.2GB | Near-lossless | Performance tier |

**TAIS Quantization Ladder:**
```
Normal:   Q5_K_M  (default)
Throttle: Q4_K_M  (automatic on temp 75–80°C)
Recovery: Q5_K_M  (restored when temp < 72°C)
```

### 7.4 Logging & Log Rotation

All logs are written via `loguru` with the following rotation policy to prevent storage bloat during 24/7 operation:

```python
logger.add(
    "./logs/aura9.log",
    level="DEBUG",
    rotation="10 MB",       # New file every 10MB
    retention="7 days",     # Delete logs older than 7 days
    compression="gz",       # Compress rotated files
    enqueue=True,           # Async — no inference blocking
    backtrace=True,
    diagnose=True
)
```

**Log Directories:**
- `./logs/aura9.log` — Main inference and orchestration log
- `./logs/tais.log` — Thermal events only
- `./logs/watchdog.log` — Watchdog alerts and heartbeats
- `./logs/audit.log` — Immutable audit trail (no rotation — append only, external backup)
- `./logs/isec.log` — Consolidation pass reports

---

## 8. Agent Communication Protocol

### 8.1 Inter-Agent Messaging (Future: Multi-Agent Mode)

Aura-9 is designed to eventually operate within a multi-agent mesh. The protocol is defined now for clean future integration:

```json
{
  "msg_id": "uuid-v4",
  "from_agent": "aura9-primary",
  "to_agent": "aura9-qa",
  "msg_type": "TASK_DELEGATION | RESULT_REPORT | STATUS_QUERY | ESCALATION | HEARTBEAT",
  "priority": "LOW | NORMAL | HIGH | CRITICAL",
  "session_id": "sess-uuid-timestamp",
  "payload": {
    "subtask_id": "ST-004",
    "description": "Verify test coverage of synthesized tool X",
    "artifacts": ["tool_x_v1.py"],
    "success_criteria": "All test cases pass 100% ground-truth validation",
    "deadline_minutes": 15
  },
  "timestamp": "ISO-8601"
}
```

### 8.2 User-Facing Communication

Aura-9 communicates with users through a structured **Notification Hierarchy**:

| Level | Trigger | Delivery |
|---|---|---|
| `SILENT` | Routine operations | Audit log only |
| `PROGRESS` | Sub-task completion, milestone | Status bar update |
| `GATE` | Human input required | Full UI interrupt + Gate Brief |
| `ALERT` | Watchdog flag, tool failure, TAIS throttle, MCP limit | Push notification |
| `CRITICAL` | TAIS emergency, Watchdog death, STATE_FAIL, SECURITY_FAIL | Immediate interrupt + sound |

---

## 9. Observability & Diagnostics

### 9.1 Metrics Endpoint

Aura-9 exposes a local metrics endpoint at `localhost:9001/metrics`:

**Agent Health:**
- `aura9_uptime_seconds`
- `aura9_task_completion_rate`
- `aura9_correction_cycles_total`
- `aura9_watchdog_alerts_total`
- `aura9_watchdog_heartbeat_age_seconds`

**TAIS:**
- `tais_current_temp_celsius`
- `tais_status`
- `tais_throttle_events_total`
- `tais_emergency_halts_total`
- `tais_active_quantization`

**Memory:**
- `apm_l1_utilization_percent`
- `apm_l2_collection_size{collection}`
- `apm_l3_node_count` / `apm_l3_edge_count`
- `apm_memory_promotions_total`
- `apm_significance_score_min` — lowest S score in active graph
- `isec_last_run_timestamp`
- `isec_nodes_archived_total`

**Performance:**
- `llm_inference_latency_seconds{model}`
- `tool_call_latency_seconds{tool}`
- `context_window_utilization_percent`
- `mcp_calls_today{server_id}`

### 9.2 Diagnostic CLI

```bash
aura9 status                              # Full system health snapshot
aura9 task show --active                  # Active mission + ASD state
aura9 watchdog log --last 10             # Recent Watchdog alerts
aura9 watchdog status                    # Liveness + last heartbeat age
aura9 tais status                        # Current temp + quantization level
aura9 checkpoint --now                   # Force immediate checkpoint
aura9 audit --task-id <uuid>             # Replay audit trail for task
aura9 memory inspect --tier L2 --query "API retry logic"
aura9 memory pin <node_id>               # Set £ = 1.0 (immutable)
aura9 memory score <node_id>             # Inspect S score
aura9 memory isec --run-now              # Force ISEC pass (GPU must be idle)
aura9 tool list                          # All tools by maturity level
aura9 tool rollback <tool_id> --to v1   # Roll back to previous version
aura9 mcp stats                          # MCP call counts per server today
aura9 graph snapshot                     # Manual FalkorDB snapshot
aura9 graph restore --file <path>        # Restore to snapshot
aura9 shutdown --graceful                # Checkpoint + clean shutdown
```

### 9.3 Health Check Protocol

On startup and every 5 minutes:

```
✓ Ollama reachable — primary model loaded
✓ Watchdog running — heartbeat 12s ago
✓ TAIS active — 64°C — NORMAL — Q5_K_M
✓ Redis responsive — ping < 1ms
✓ Qdrant healthy — 5 collections present
✓ FalkorDB connected — graph accessible (1,247 nodes / 3,891 edges)
✓ ASD state valid — status: IDLE
✓ MCP Gateway — 2/3 servers reachable (1 timeout: logged)
✓ Wasm Sandbox — test execution passed
✓ Zero-Trust Sanitizer — active
✓ VRAM utilization — 5.6GB / 6.0GB (93%) — TAIS monitoring
✓ Log rotation — active (last rotate: 2026-04-10 03:12 UTC)
```

Any failed check is logged and, if it affects core functionality, escalates to the appropriate notification level.

---

## 10. Upgrade & Self-Evolution Pathway

### 10.1 Model Upgrade Path

Aura-9 is model-agnostic at the kernel level:

```
Current: Qwen 3.5 9B
   │
   ├─ Lateral:  Mistral 9B, Gemma 9B, LLaMA 3.2 9B
   │            (same parameter count, different strengths)
   └─ Vertical: Qwen 3.5 14B, Qwen 3.5 32B
                (more capability, higher VRAM cost)
```

**Upgrade Protocol:**
1. Pull new model via Ollama.
2. Run 10-task benchmark suite against new model.
3. Compare: completion rate, correction rate, TAIS thermal profile, inference latency.
4. If benchmark score > current model by > 5% → promote.
5. Old model retained for 7 days as rollback.

### 10.2 Continuous Learning (Without Fine-Tuning)

Aura-9 improves without modifying model weights through:

- **Retrieval quality:** L2 knowledge grows richer with every successful task.
- **Graph density:** L3 accumulates connections, revealing non-obvious relationships.
- **Skill accumulation:** Each synthesized tool reduces future latency for similar tasks.
- **Failure analysis:** Every failed task generates a post-mortem in `failure_analysis` collection, reducing future recurrence.
- **ISEC consolidation:** Background passes promote implicit patterns to explicit L3 knowledge nodes.

### 10.3 Episodic Memory Lifecycle

Over months of continuous operation, the L3 graph and L2 Qdrant collections grow. The Significance Score system prevents bloat:

```
Active node:    S ≥ 0.2  → retained, scored on every retrieval
Flagged node:   S < 0.2  → marked for archival by ISEC
Archived:       Exported to compressed cold storage
Pinned node:    £ = 1.0  → immune to archival regardless of S
```

Cold storage format: `./archive/memory-{date}-{collection}.gz`
Archived nodes are never deleted — they can be restored via:
```bash
aura9 memory restore --archive ./archive/memory-2026-01-15-expertise.gz
```

---

## 11. Deployment Checklist

### Pre-Launch

- [ ] NVIDIA drivers updated (CUDA 12.x) — verified in WSL2 via `nvidia-smi`
- [ ] `pynvml` installed and GPU sensor access confirmed in WSL2
- [ ] Ollama installed — `ollama serve` running
- [ ] `qwen3.5:9b-instruct-q5_k_m` pulled and verified
- [ ] `qwen3.5:1.5b` (Watchdog) pulled and verified
- [ ] `nomic-embed-text` (embeddings) pulled and verified
- [ ] Docker Compose stack running: Redis, Qdrant, FalkorDB — all healthy
- [ ] FalkorDB graph schema applied and verified
- [ ] Qdrant collections initialized: `expertise`, `documentation`, `skill_library`, `past_missions`, `failure_analysis`
- [ ] Python project scaffold created, venv active, dependencies installed
- [ ] `config/settings.yaml` reviewed and customized
- [ ] Log directories created, rotation policy active
- [ ] Wasm runtime installed — sandbox test execution passed
- [ ] Zero-Trust Sanitizer configured — test payload scan passed
- [ ] MCP Gateway configured — at least one Tier-1 server verified
- [ ] Red Zone config reviewed and customized for your environment
- [ ] Audit trail path set — confirmed outside agent's writable scope
- [ ] Windows Task Scheduler cold-start trigger configured and tested
- [ ] TAIS thermal thresholds confirmed for your GPU model
- [ ] VRAM headroom verified: > 10% free after primary model load

### First Run

- [ ] Run full health check — all systems green
- [ ] Assign a simple test task — verify full lifecycle (CREATED → DELIVERED)
- [ ] Confirm ASD state updates via Precision Planner Mode (inspect Redis `asd:state`)
- [ ] Confirm checkpoint created within 15 minutes
- [ ] Simulate reboot — verify cold-start recovery resumes from checkpoint
- [ ] Trigger a TAIS throttle manually (or verify threshold detection is live)
- [ ] Confirm Watchdog heartbeat visible in Redis (`watchdog:heartbeat`)
- [ ] Kill Watchdog process — confirm liveness alert triggers within 90s
- [ ] Restore Watchdog — confirm autonomous mode resumes
- [ ] Review first audit trail entries for expected session_id stamping
- [ ] Run `aura9 memory score` on a promoted node — confirm S formula active
- [ ] Verify FalkorDB daily snapshot job is scheduled

### Ongoing Operations

- [ ] Review Watchdog alert log weekly
- [ ] Review TAIS thermal log weekly — identify sustained high-temp patterns
- [ ] Inspect skill_library monthly — deprecate unused tools, check quarantine queue
- [ ] Monitor VRAM utilization trend — alert threshold: > 93% sustained
- [ ] Run `aura9 memory isec --run-now` monthly — review consolidation report
- [ ] Audit MCP call counters monthly — adjust rate limits if needed
- [ ] Verify FalkorDB daily snapshots — test restore quarterly
- [ ] Archive cold storage quarterly — verify restore from oldest archive

---

## Appendix A — Glossary

| Term | Definition |
|---|---|
| **APM** | Autonomous Persistence Module — the Triple-Thread memory system (Redis + Qdrant + FalkorDB) |
| **ASD** | Aura State Daemon — the persistent JSON state tree in Redis that is the authoritative source of truth for the current mission |
| **CoT** | Chain-of-Thought — the step-by-step reasoning style used by the 9B model |
| **DAG** | Directed Acyclic Graph — the dependency graph used for parallel sub-task scheduling |
| **Gate** | A deliberate pause requiring human input before autonomous execution proceeds |
| **ISEC** | Idle-State Epistemic Consolidation — background daemon that consolidates and prunes memory during GPU idle periods |
| **MCP** | Model Context Protocol — standardized protocol for tool and API connectivity |
| **Mission Manifest** | The structured YAML task plan generated by the Prompt Enhancement Pre-Processor |
| **Precision Planner Mode** | A JSON-exclusive system instruction set used solely for ASD state updates |
| **Red Zone** | A UI region that Aura-9 is physically blocked from interacting with |
| **Significance Score (S)** | `S = (R × F) + £` — the decay formula governing memory archival across L2 and L3 |
| **Skill Forge** | The system by which Aura-9 synthesizes, tests, versions, and promotes new tools |
| **State-Node Graph** | The FalkorDB structure storing ASD shadow copies and checkpoint history |
| **TAIS** | Thermal-Aware Inference Scheduling — hardware telemetry hook protecting the GPU during sustained autonomous operation |
| **VLA Bridge** | Vision-Language-Action module — screen perception and UI interaction layer |
| **Watchdog** | The secondary 1B–3B oversight model monitoring the 9B for drift, loops, and security violations |
| **Wasm** | WebAssembly — the sandboxing technology isolating all code execution from the host OS |
| **Zero-Trust Sanitizer** | Pre-execution payload interceptor scanning for hardcoded secrets, unauthorized paths, and injection patterns |

---

## Appendix B — Failure Taxonomy

Every failure Aura-9 encounters is classified before the correction loop begins. Classification determines the recovery action.

| Class | Examples | Immediate Response | Correction Behavior |
|---|---|---|---|
| `INFERENCE_FAIL` | Model timeout, OOM, malformed output | Retry inference × 2 | Fallback to lower quant → CPU offload → halt |
| `TOOL_FAIL` | MCP timeout, sandbox crash, import denied | Retry × 1 | Check for alt tool in registry → escalate if none |
| `MEMORY_FAIL` | Redis down, Qdrant unreachable, FalkorDB write error | Promote shadow / degrade mode | ALERT user → continue in degraded mode if non-critical |
| `STATE_FAIL` | ASD corruption, Precision Planner schema violation, FalkorDB write failure | Halt ASD writes | Restore from FalkorDB shadow → manual recovery if shadow corrupt |
| `SECURITY_FAIL` | Red Zone breach attempt, sanitizer block, unauthorized network call | Immediate halt | Log to audit → notify user → no auto-retry |
| `TAIS_THROTTLE` | GPU temp 75–80°C | Shift quant to Q4_K_M | Resume normally at lower quant |
| `TAIS_EMERGENCY` | GPU temp > 83°C | Checkpoint → halt inference | Resume only after temp < 72°C + user confirmation |
| `WATCHDOG_FAIL` | Watchdog heartbeat timeout (> 90s) | Suspend autonomous mode | Restart Watchdog (× 3) → full halt if unrecoverable |

---

## Appendix C — Version History

| Version | Changes |
|---|---|
| **1.0** | Core framework defined |
| **2.0** | Full architectural expansion: context management, parallel execution, multi-agent protocol, observability, upgrade pathway, deployment checklist |
| **2.1** | **TAIS** (thermal-aware inference scheduling with threshold ladder and temperature-confirmed recovery); **State-Driven Context Compression**; **ASD** (Aura State Daemon with dual-write Redis/FalkorDB architecture); **Precision Planner Mode** (locked JSON schema); **ISEC** (idle-state epistemic consolidation daemon); **Zero-Trust Sanitizer** (payload-level pre-execution interceptor); **Significance Score** formula `S = (R × F) + £` extended to both L2 and L3; **Skill Forge** upgraded to 100% ground-truth validation gate; **Tool versioning** with DEPRECATED/QUARANTINED states and 7-day retention; **Session identity system** (session_id stamped across all tiers); **Watchdog liveness check** (90s heartbeat, autonomous mode suspension); **Human Gate timeout ladder** (PAUSED → SUSPENDED at 120 min); **Log rotation policy** (10MB / 7-day / gz); **MCP call accounting** (per-server daily Redis counter); **FalkorDB daily snapshots** (30-day retention, CLI restore); **VLA Bridge** formally assigned to Phase 4; **Failure taxonomy** formalized (8 classes with recovery actions); **Cold-Start Resumption** WSL2 trigger defined (Windows Task Scheduler); APM name confirmed as Autonomous Persistence Module. |

---

*Aura-9 Specification — Version 2.1 | Canonical Development Blueprint | All architectural decisions supersede previous versions.*

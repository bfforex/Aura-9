# Aura-9: Autonomous Reasoning Agent
## Official Technical Specification
**Version:** 2.4 | **Status:** Active Development Blueprint | **Classification:** Internal Reference

---

> *"The gap between knowing and doing is where most AI systems fail. Aura-9 was designed to live in that gap."*

---

## Table of Contents

1. [Identity & Philosophy](#1-identity--philosophy)
2. [Core Reasoning Engine](#2-core-reasoning-engine)
   - 2.7 [Confidence Score Computation](#27-confidence-score-computation)
   - 2.8 [System Prompt Definitions](#28-system-prompt-definitions)
   - 2.9 [Tool-Calling Format & Function Schema](#29-tool-calling-format--function-schema)
3. [Triple-Thread Memory System (APM)](#3-triple-thread-memory-system-apm)
   - 3.4.1 [FalkorDB Failure Handling](#341-falkordb-failure-handling)
4. [Capability & Tooling — The Skill Forge](#4-capability--tooling--the-skill-forge)
5. [Autonomous Task Orchestration](#5-autonomous-task-orchestration)
   - 5.8 [Session Lifecycle](#58-session-lifecycle)
   - 5.9 [Race Condition & Consistency Resolutions](#59-race-condition--consistency-resolutions)
6. [Security & Operational Guardrails](#6-security--operational-guardrails)
   - 6.0 [Authentication & Authorization](#60-authentication--authorization)
   - 6.6 [PII Scrubbing](#66-pii-scrubbing)
   - 6.7 [Dual-Confirmation Gate for Financial Transactions](#67-dual-confirmation-gate-for-financial-transactions)
7. [Infrastructure & Hardware](#7-infrastructure--hardware)
   - 7.6 [Project Structure & Module Boundaries](#76-project-structure--module-boundaries)
   - 7.7 [Ollama API Interaction Patterns](#77-ollama-api-interaction-patterns)
   - 7.8 [Infrastructure Latency Degradation](#78-infrastructure-latency-degradation)
8. [Agent Communication Protocol](#8-agent-communication-protocol)
9. [Observability & Diagnostics](#9-observability--diagnostics)
   - 9.4 [Command Behavior During Active Tasks](#94-command-behavior-during-active-tasks)
   - 9.5 [Graceful Shutdown During ISEC](#95-graceful-shutdown-during-isec)
10. [Upgrade & Self-Evolution Pathway](#10-upgrade--self-evolution-pathway)
11. [Deployment Checklist](#11-deployment-checklist)
12. [Appendix A — Glossary](#appendix-a--glossary)
13. [Appendix B — Failure Taxonomy](#appendix-b--failure-taxonomy)
14. [Appendix C — IPC Channel Registry](#appendix-c--ipc-channel-registry)
15. [Appendix D — Port Allocation](#appendix-d--port-allocation)
16. [Appendix E — Benchmark Suite](#appendix-e--benchmark-suite)
17. [Appendix F — Version History](#appendix-f--version-history)
18. [Appendix G — Configuration Schema](#appendix-g--configuration-schema)
19. [Appendix H — Testing Strategy](#appendix-h--testing-strategy)
20. [Appendix I — Graph Schema Migration](#appendix-i--graph-schema-migration)

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

> **Fix applied (v2.4):** Qwen 3.5 9B confirmed as the target model (`qwen3.5:9b-instruct` family via Ollama). Verified specs: Q5_K_M quantization ~5.5 GB VRAM, 32,768-token context window with native tool-calling support. Model selection criteria: JSON/tool-calling support required, >=9B parameters recommended for complex reasoning, local Ollama compatibility required. If adapting the spec for a different model, validate VRAM estimates and context window before adjusting TAIS thresholds.

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


> **Fix applied (v2.4):** Concrete Ollama API calls for TAIS quantization switch defined. Temperature jump behavior (skip COOLDOWN) defined.

**Concrete Ollama API Calls for Quant Switch:**

Model names in Ollama must match pulled model tags exactly:
- Q5_K_M: `qwen3.5:9b-instruct-q5_k_m`
- Q4_K_M: `qwen3.5:9b-instruct-q4_k_m`

Both quant variants must be pulled before deployment:
```bash
ollama pull qwen3.5:9b-instruct-q5_k_m
ollama pull qwen3.5:9b-instruct-q4_k_m
```

**Unload current model (Q5_K_M):**
```python
import httpx

async def unload_model(model_name: str, ollama_host: str = "http://localhost:11434"):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ollama_host}/api/generate",
            json={"model": model_name, "prompt": "", "keep_alive": 0},
            timeout=30.0
        )
        response.raise_for_status()
```

**Load replacement model (Q4_K_M) and verify:**
```python
async def load_model(model_name: str, ollama_host: str = "http://localhost:11434"):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ollama_host}/api/generate",
            json={"model": model_name, "prompt": "ping", "keep_alive": -1, "stream": False},
            timeout=120.0
        )
        response.raise_for_status()

async def check_model_loaded(model_name: str, ollama_host: str = "http://localhost:11434") -> bool:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ollama_host}/api/ps", timeout=5.0)
        running = response.json().get("models", [])
        return any(m["name"] == model_name for m in running)
```

**Temperature Jump Past COOLDOWN to EMERGENCY (RC-41):**
If GPU temperature jumps directly from NORMAL/THROTTLE to >83C (skipping COOLDOWN):
```
1. Immediately halt any active inference call (interrupt the httpx stream)
2. Active inference call result is discarded; task enters CORRECTING on recovery
3. Write emergency checkpoint (ASD state with tais_halt_reason="EMERGENCY")
4. Unload all models (keep_alive=0 on both primary and Watchdog)
5. Emit CRITICAL alert
6. Resume only after: temp < 72C + explicit user confirmation
```
Key difference from normal COOLDOWN: inference is **killed immediately**, not drained.

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
  manifest_version: "2.4"
  task_id: "uuid-v4"
  session_id: "sess-uuid-timestamp"
  created_at: "ISO-8601"
  task_class: "STANDARD"
  priority: "LOW|NORMAL|HIGH|CRITICAL"   # default: NORMAL
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

> **Fix applied (v2.3):** `priority` field added to Mission Manifest schema. Valid values: `LOW|NORMAL|HIGH|CRITICAL`, default `NORMAL`. Aligns with the inter-agent message schema (Section 8.3).

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


> **Fix applied (v2.4):** Context Budget Manager eviction policy and checkpoint content defined.

**Eviction Policy (Budget Exceeded):**
When a context budget allocation is exceeded during retrieval or generation:

```
Priority order for eviction (least important first):
1. Superseded plans / abandoned branches -> always pruned first
2. Oldest L1 episodic memory turns (by timestamp) -> collapsed to micro-summaries
3. Lowest-scoring L2 retrieval chunks (by RRF score) -> dropped from injection
4. Oldest completed sub-task scratchpad entries -> collapsed to single-line summary
5. Lowest-significance L3 graph context nodes -> dropped from injection

NEVER evicted:
- System Prompt (2,048 token budget is hard; exceeded = configuration error)
- Active sub-task scratchpad (current working state)
- ASD State Injection (required for Precision Planner correctness)
- Safety Buffer (4,096 tokens; never allocated)
```

If after eviction the total still exceeds 28,672 tokens, a **Memory Compression Pass** is forced: collapse the oldest 50% of L1 episodic content to micro-summaries before re-injection.

**Checkpoint Content Definition:**
An ASD checkpoint (`asd:checkpoint:{task_id}:{ckpt_id}`) contains:

```json
{
  "checkpoint_id": "ckpt-uuid",
  "task_id": "task-uuid",
  "session_id": "sess-uuid",
  "timestamp": "ISO-8601",
  "asd_state": { },
  "scratchpad_summary": "Compressed CoT summary (not full scratchpad)",
  "completed_subtask_results": [
    {"id": "ST-001", "result_summary": "...", "confidence": 0.92}
  ],
  "in_flight_tool_result_keys": ["sess:...:tool_results:call-123"],
  "tais_status_at_checkpoint": "NORMAL",
  "vram_at_checkpoint_bytes": 5873270374,
  "correction_cycles_used": 1,
  "confidence_at_checkpoint": 0.87
}
```

**What is NOT in a checkpoint:**
- Full raw tool outputs (stored in L1 Redis by their `tool_results` keys)
- Full conversation history (reconstructed from L1 Redis on resume)
- Full scratchpad (only compressed summary; resume from `next_action`)

**ASD Checkpoint TTL Alignment (P2 #48):**
ASD checkpoint TTL extended to **30 days** (matching Qdrant/FalkorDB snapshot retention) to ensure consistent backup/restore windows.

**Output Buffer Overflow:**
If a response requires more than **3,072 tokens**, it is split: the primary response fills the Output Buffer and is delivered immediately; the continuation is stored in L1 Redis at `sess:{session_id}:continuation:{turn_id}`. On the user's next interaction, the continuation is automatically prepended and delivered. This is transparent to the user.

> **Fix applied (v2.3):** Output Buffer overflow behavior defined. Continuations stored in L1 Redis with session-bound TTL, surfaced on follow-up turn.

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
    "status": "CREATED|PLANNED|EXECUTING|CORRECTING|VERIFYING|DELIVERED|PAUSED|BLOCKED|SUSPENDED|ESCALATED|FAILED|IDLE",
    "active_subtasks": ["ST-001"],
    "completed_subtasks": ["ST-000"],
    "blocked_by": "string | null",
    "confidence": 0.0,
    "next_action": "string — exact planned next step",
    "failure_class": "string | null",
    "checkpoint_required": false,
    "tais_status": "NORMAL|THROTTLE|COOLDOWN|EMERGENCY|SENSOR_FAIL",
    "tais_halt_reason": "string | null"
  }
}
```

> **Fix applied (v2.3):** `IDLE` added to status enum. `tais_halt_reason` field added (non-nullable on TAIS state writes; null otherwise). On task terminal state (`DELIVERED`, `ESCALATED`, `FAILED`), ASD transitions to `IDLE` via one final Precision Planner write — this is the ISEC activation signal (see Section 3.6).

**Rules:**
- No natural language outside this JSON structure is permitted.
- The schema is immutable — fields cannot be added, removed, or renamed by the model.
- `task_id` and `session_id` are non-nullable and must match the active Mission Manifest.
- Any output failing JSON validation → model re-prompted once. Second failure → `STATE_FAIL` (Appendix B).

**State-Change Coalescing:**
If multiple ASD-relevant events occur within a **5-second window**, they are batched into a single Precision Planner invocation. The most recent state wins for each field. This prevents inference overhead from rapid sub-task state churn during parallel execution.

> **Fix applied (v2.3):** State-change coalescing added to prevent redundant Precision Planner invocations during high-frequency sub-task execution.

### 2.7 Confidence Score Computation

> **Fix applied (v2.4):** Confidence score computation fully defined. Resolves the gap where thresholds (0.72 escalation, 0.85 L1-L2 promotion) were referenced but the computation was unspecified.

The confidence score is a **composite heuristic** computed by the Reflection Module after each sub-task execution. It is NOT derived from model logit probabilities (inaccessible via Ollama API). It is a weighted composite of four measurable signals:

```
C = (w1 * T) + (w2 * V) + (w3 * R) + (w4 * A)

Where:
  T = Tool Success Rate       (successful_tool_calls / total_tool_calls), range 0.0-1.0
                                If no tool calls: T = 1.0
  V = Verification Pass Rate  (passed_checks / total_checks), range 0.0-1.0
                                Success criteria checks from Mission Manifest
  R = Retry Penalty           1.0 - (correction_cycles_used / max_correction_cycles)
                                0 corrections -> R = 1.0; max corrections -> R = 0.0
  A = Ambiguity Penalty       1.0 - ambiguity_score (see Section 4.3.4)
                                Fully unambiguous -> A = 1.0

Weights (sum to 1.0):
  w1 = 0.35   (tool success most predictive of task quality)
  w2 = 0.35   (verification pass rate equally important)
  w3 = 0.20   (correction cycles signal difficulty/uncertainty)
  w4 = 0.10   (residual ambiguity penalty)

Range: 0.0-1.0
```

**Threshold Reference:**

| Threshold | Used For | Section |
|---|---|---|
| `< 0.72` | Trigger human escalation | 1.2, 2.3 |
| `> 0.85` | Eligibility for L1 to L2 promotion | 3.2 |
| `> 0.80` | Benchmark suite pass criterion (Task 10) | Appendix E |

**Sub-Task vs. Mission Confidence:**
Each sub-task gets its own C score. The **mission-level confidence** is the weighted average:

```
C_mission = SUM(C_subtask_i * complexity_i) / SUM(complexity_i)
```

**Special Cases:**
- TRIVIAL tasks (no tool calls, no verification steps): `C = 1.0`
- Task that hits escalation: final `C` at escalation time reported in escalation package
- TAIS EMERGENCY halt: `C` frozen at last computed value, reported in checkpoint

### 2.8 System Prompt Definitions

> **Fix applied (v2.4):** Full system prompt text defined. These are the three critical prompts that drive all model behavior.

#### 2.8.1 Base System Prompt (Identity + Reasoning Style + Tool-Calling)

```
You are Aura-9, an Autonomous Reasoning Agent. You are NOT a chatbot. You are an
occupational intelligence system. Once assigned a task, you own it. You plan, execute,
self-correct, build tools when needed, and only surface to the user at meaningful
decision gates.

REASONING STYLE:
- Think step-by-step before acting. Show your reasoning in <think> tags.
- After reasoning, produce a structured action (tool call or response).
- After every tool result, evaluate: did this match the intended sub-task?
  Did it meet success criteria?
- If confidence drops below 0.72: pause and escalate. Do not guess when uncertain.

TOOL CALLING:
- Call tools using the exact JSON schema provided in the tools array.
- Never fabricate tool results. Never skip tools when they are required.
- After a tool call, wait for the result before proceeding.

ASD STATE UPDATES:
- When your task state changes, you will be prompted for a Precision Planner Mode update.
- In Precision Planner Mode: output ONLY the JSON schema defined in Section 2.6. No prose.

CONSTRAINTS:
- You do not have unsupervised internet write access.
- You do not execute financial transactions without explicit dual-confirmation.
- You do not modify your own Watchdog module.
- You do not persist raw user PII beyond the active session.
- You do not resume autonomous execution after TAIS EMERGENCY without user confirmation.
- You do not accept new tasks during graceful shutdown.

QUALITY STANDARD:
- Every output you produce must pass your own internal verification before delivery.
- Unverified results do not reach the user.
```

#### 2.8.2 Precision Planner Mode Override Prompt

This prompt replaces the base system prompt when an ASD state write is required:

```
PRECISION PLANNER MODE ACTIVE.

You are in JSON-only output mode. Produce EXACTLY the following JSON structure and
nothing else. No prose. No explanation. No markdown formatting. No code blocks.
Output must begin with { and end with }.

Required schema:
{
  "asd_update": {
    "task_id": "<non-nullable string>",
    "session_id": "<non-nullable string>",
    "current_objective": "<string>",
    "status": "<CREATED|PLANNED|EXECUTING|CORRECTING|VERIFYING|DELIVERED|PAUSED|BLOCKED|SUSPENDED|ESCALATED|FAILED|IDLE>",
    "active_subtasks": ["<ST-ID>"],
    "completed_subtasks": ["<ST-ID>"],
    "blocked_by": "<string or null>",
    "confidence": <0.0-1.0 float>,
    "next_action": "<exact planned next step>",
    "failure_class": "<string or null>",
    "checkpoint_required": <true|false>,
    "tais_status": "<NORMAL|THROTTLE|COOLDOWN|EMERGENCY|SENSOR_FAIL>",
    "tais_halt_reason": "<string or null>"
  }
}

Any deviation from this schema is invalid. If you cannot comply, output:
{"error": "PRECISION_PLANNER_SCHEMA_FAIL"}
```

#### 2.8.3 Pre-Processor Prompt (Mission Manifest Generation)

Used for STANDARD task classification (TRIVIAL tasks use rule-based fast path; see Section 4.3.4):

```
You are the Pre-Processor for Aura-9. Your job is to transform the user's raw input
into a structured Mission Manifest.

INSTRUCTIONS:
1. Analyze the input for: intent, complexity, required tools, ambiguity, success criteria.
2. Compute estimated_complexity (0.0-1.0) using the heuristic guide below.
3. If ambiguity_score > 0.60: output ONLY a single clarification question (no manifest).
4. Otherwise: output the Mission Manifest in the YAML schema below.
5. Output ONLY the YAML. No prose. No markdown code fences.

COMPLEXITY HEURISTIC:
  0.0-0.2: Trivial -- single lookup, no tool call, < 5 reasoning steps
  0.2-0.4: Simple -- 1-2 tool calls, straightforward logic, no branching
  0.4-0.6: Moderate -- 2-4 tool calls, some conditional logic, possible retry
  0.6-0.8: Complex -- 4+ tool calls, multi-step reasoning, state management
  0.8-1.0: Very complex -- novel problem, multi-tool orchestration, potential Skill synthesis

MANIFEST SCHEMA:
mission_manifest:
  manifest_version: "2.4"
  task_id: "<uuid-v4>"
  session_id: "{session_id}"
  created_at: "<ISO-8601>"
  task_class: "STANDARD"
  priority: "<LOW|NORMAL|HIGH|CRITICAL>"
  original_intent: "<raw user input verbatim>"
  interpreted_goal: "<expanded, KPI-tagged goal>"
  ambiguity_score: <0.0-1.0>
  sub_tasks:
    - id: ST-001
      description: "<specific sub-task>"
      success_criteria: "<measurable pass condition>"
      tools_required: ["<tool_name>"]
      estimated_complexity: <0.0-1.0>
      depends_on: []
  constraints:
    time_budget_minutes: <integer>
    escalation_threshold: 0.72
    human_gate_required: <true|false>
    max_correction_cycles: 3
```

### 2.9 Tool-Calling Format & Function Schema

> **Fix applied (v2.4):** Complete tool-calling format defined. Ollama supports OpenAI-compatible tool calling via the `/api/chat` endpoint with a `tools` array.

#### 2.9.1 Tool Presentation to Model

Tools are provided in the Ollama `/api/chat` request as a `tools` array following the OpenAI function calling schema:

```json
{
  "model": "qwen3.5:9b-instruct-q5_k_m",
  "messages": ["..."],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "qdrant_search",
        "description": "Search Aura-9 semantic memory (L2) via hybrid vector + keyword search.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string", "description": "Natural language search query"},
            "collection": {
              "type": "string",
              "enum": ["expertise", "documentation", "skill_library", "past_missions", "failure_analysis"]
            },
            "top_k": {"type": "integer", "default": 5}
          },
          "required": ["query", "collection"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "python_exec",
        "description": "Execute Python code in isolated subprocess sandbox. No network access.",
        "parameters": {
          "type": "object",
          "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout_seconds": {"type": "integer", "default": 120, "maximum": 120}
          },
          "required": ["code"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "mcp_call",
        "description": "Call an MCP server tool. Subject to tier approval and daily limits.",
        "parameters": {
          "type": "object",
          "properties": {
            "server_id": {"type": "string"},
            "tool_name": {"type": "string"},
            "arguments": {"type": "object"}
          },
          "required": ["server_id", "tool_name", "arguments"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "memory_write",
        "description": "Write a note, conclusion, or fact to session memory for later retrieval.",
        "parameters": {
          "type": "object",
          "properties": {
            "content": {"type": "string"},
            "content_type": {
              "type": "string",
              "enum": ["factual_knowledge", "reusable_insight", "entity_relationship", "ephemeral_working_data"]
            },
            "tags": {"type": "array", "items": {"type": "string"}}
          },
          "required": ["content", "content_type"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "vla_screenshot",
        "description": "Capture and analyze current screen state using VLA Bridge.",
        "parameters": {
          "type": "object",
          "properties": {
            "focus_region": {"type": "string", "description": "Optional: screen region to focus on"}
          },
          "required": []
        }
      }
    }
  ],
  "stream": true,
  "options": {"temperature": 0.3, "num_ctx": 28672}
}
```

#### 2.9.2 Model Tool-Call Output Format

When the model decides to call a tool, Ollama returns a message with `tool_calls`:

```json
{
  "model": "qwen3.5:9b-instruct-q5_k_m",
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "qdrant_search",
          "arguments": {
            "query": "API retry logic exponential backoff",
            "collection": "expertise",
            "top_k": 5
          }
        }
      }
    ]
  },
  "done": true
}
```

#### 2.9.3 Tool Result Injection

After executing the tool, the result is appended as a `tool` role message:

```json
{
  "role": "tool",
  "content": "[{\"id\": \"node-abc\", \"score\": 0.94, \"payload\": {\"text\": \"Exponential backoff: start at 1s, double each retry, cap at 60s...\"}}]"
}
```

#### 2.9.4 Complete Tool-Call Round-Trip Example

```
User: "Find the best retry strategy for our HTTP client"

 1. Request to /api/chat with tools array, temperature=0.3
 2. Model response: tool_calls: [{name: "qdrant_search", arguments: {...}}]
 3. Orchestrator executes qdrant_search
 4. Appends tool result to message history
 5. Second request to /api/chat (continuation with tool result in context)
 6. Model final response: "Based on retrieved expertise: Use exponential backoff..."
 7. Reflection Module evaluates result against success_criteria
 8. Confidence score computed
 9. If confidence > 0.72: deliver result
10. If confidence <= 0.72: enter correction cycle
```

#### 2.9.5 Skill Tool Registration

When a Skill from the `skill_library` is available for a task, it is dynamically added to the `tools` array at request time. The tool schema is stored as part of the Skill's Qdrant payload (see Section 4.4).


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
| `sess:{session_id}:turns` | Conversation turns (Redis Stream) | Session-bound (max 24h) |
| `sess:{session_id}:tool_results:{call_id}` | Raw tool outputs (full verbatim) | Session-bound (max 24h) |
| `sess:{session_id}:scratchpad` | Active CoT state | Session-bound (max 24h) |
| `sess:{session_id}:metadata` | Session metadata (`last_active`, `created_at`, `session_id`) | Session-bound (max 24h) |
| `sess:{session_id}:continuation:{turn_id}` | Output Buffer overflow continuations | Session-bound (max 24h) |
| `asd:state` | ASD live state tree (Precision Planner JSON) | No TTL |
| `asd:checkpoint:{task_id}:{ckpt_id}` | Point-in-time ASD snapshots | 7 days |
| `mcp:calls:{server_id}:{date}` | MCP call counter per server per day | 48 hours |
| `watchdog:heartbeat` | Watchdog liveness key | 90s TTL (refreshed every 30s) |
| `watchdog:buffer` | Watchdog output buffer during restart | 5 minutes |
| `isec:progress` | ISEC pass progress checkpoint | Until ISEC completion |

> **Fix applied (v2.3):** Maximum session TTL of **24 hours** added as a safety net for all session-bound keys. `sess:{session_id}:metadata` key added with `last_active` timestamp refreshed on every interaction. `watchdog:buffer`, `isec:progress`, and `sess:{session_id}:continuation:{turn_id}` keys added to namespace table.

**Session Expiry:**
The `sess:{session_id}:metadata` key carries `last_active` (refreshed on every turn) and `created_at`. The Continuity Engine health check sweep (every 5 minutes) inspects all active session metadata keys and expires any session whose `last_active` exceeds **24 hours**, cleaning up all associated session-scoped keys.

> **Fix applied (v2.3):** Session-bound TTL safety net defined. Continuity Engine health check handles stale session cleanup. Prevents Redis key leakage on crashes without graceful shutdown.

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


> **Fix applied (v2.4):** Hybrid search RRF implementation detail and division-by-zero edge case defined.

**Hybrid Search Implementation (RRF):**

Qdrant's native sparse+dense hybrid search is used with a `prefetch` + `query` fusion:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion

def hybrid_search(client: QdrantClient, collection: str, query_text: str, top_k: int = 5):
    dense_vector = embed(query_text)  # nomic-embed-text, 768-dim
    results = client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(query=dense_vector, using="text_dense", limit=20),
            Prefetch(query=sparse_encode(query_text), using="text_sparse", limit=20),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return results.points
```

**RRF Formula and Parameters:**
```
RRF_score(doc) = SUM_i [1 / (k + rank_i(doc))]

k = 60  (standard RRF constant)
top_k = 5  (default; 10 for L3 graph context injection)
pre_fetch_limit = 20  (candidates from each retriever before fusion)
```

**Qdrant Collection Configuration:**
Each collection requires both `text_dense` (768-dim cosine) and `text_sparse` (BM25) vector fields:

```python
from qdrant_client.models import VectorParams, Distance, SparseVectorParams

client.create_collection(
    collection_name="expertise",
    vectors_config={"text_dense": VectorParams(size=768, distance=Distance.COSINE)},
    sparse_vectors_config={"text_sparse": SparseVectorParams()},
)
```

**Result Formatting for Context Injection:**
Top-k results are formatted for the L2 context slot (8,192 tokens):
```
[L2-1 | score=0.94 | expertise] Exponential backoff: start at 1s, double each retry...
[L2-2 | score=0.89 | documentation] HTTP 429 handling requires...
```
Each chunk prefixed with `[L2-N | score=X.XX | collection]`. Total L2 injection truncated to 8,192 tokens.

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


> **Fix applied (v2.4):** Division-by-zero edge case resolved for cold-start and single-node scenarios.

**R Normalization Edge Cases:**
- **Cold start (zero nodes):** `R = 1.0` for the first promoted node (no normalization needed).
- **Single node with zero retrievals:** `R = 0.0` (grace period protects from archival).
- **All nodes at zero retrievals:** `max_retrievals = 0` -> `R = 0.0` for all nodes. Grace period immunity applies.

```python
def compute_R(retrievals: int, max_retrievals: int) -> float:
    if max_retrievals == 0:
        return 0.0
    return min(retrievals / max_retrievals, 1.0)
```

**Decay Behavior:**
- S recalculated after every retrieval event.
- `£ = 1.0` renders a node immortal regardless of R and F.
- Nodes below threshold are flagged — never auto-deleted. ISEC handles archival.
- `EXPERIMENTAL` tool maturity: if a Skill remains `EXPERIMENTAL` for > 30 days with zero failures, it is eligible for manual promotion review regardless of use count (see Section 4.5).

**Grace Period for Newly Promoted Nodes:**
Newly promoted L2 and L3 nodes carry a `promoted_at: ISO-8601` timestamp. During ISEC Pass 4 (Decay Audit), nodes where `now − promoted_at < 30 days` are **skipped** — they are immune from archival flagging regardless of their S score. This prevents brand-new, high-quality knowledge from being immediately flagged due to R = 0.0 (no retrievals yet).

> **Fix applied (v2.3):** `promoted_at: ISO-8601` field added to all L2 and L3 nodes at promotion time. ISEC Pass 4 skips nodes within their 30-day grace period. Resolves dead zone where newly promoted nodes with R = 0.0 would compute S = 0.0 and immediately qualify for archival.

**CLI:**
```bash
aura9 memory pin <node_id>      # Set £ = 1.0
aura9 memory unpin <node_id>    # Reset £ = 0.0
aura9 memory score <node_id>    # Inspect current S, R, F, £
```

**Qdrant Backup Strategy:**
Qdrant snapshots are managed by the Continuity Engine (see Section 5.6). A full snapshot of all collections is taken daily alongside the FalkorDB graph snapshot. Snapshots are stored in `./backups/qdrant-{date}/` and retained for **30 days** before archival to `./archive/qdrant/`.

```bash
aura9 qdrant snapshot                              # Manual snapshot (all collections)
aura9 qdrant restore --dir ./backups/qdrant-2026-04-10/   # Rollback to snapshot
```

> **Fix applied (v2.3):** Qdrant backup strategy defined. Daily snapshots via Continuity Engine. CLI commands `aura9 qdrant snapshot` and `aura9 qdrant restore` added.

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
(:Agent {id, name, version, created_at})
(:Session {id, started_at, ended_at, agent_id})
(:Project {id, name, session_id, created_at})
(:Task {id, status, session_id, created_at, significance_score})
(:Tool {id, name, version, maturity, session_id})
(:Skill {id, name, version, maturity, trust_level, session_id, significance_score, promoted_at})
(:File {id, path, type, session_id})
(:Entity {id, name, type, session_id})
(:StateNode {task_id, checkpoint_id, timestamp, status,
             redis_snapshot_key, next_action, session_id, tais_halt_reason})
(:MemoryNode {id, session_id, tier, significance_score, user_pinned, promoted_at})

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

> **Fix applied (v2.3):** `(:Agent {id, name, version, created_at})` node type added. This is a **singleton node** created at first boot and referenced by all `(Tool)-[:NATIVE_TO]->(Agent)` and `(Agent)-[:HAS_SKILL]->(Skill)` relationships. `promoted_at` field added to `(:Skill)` and `(:MemoryNode)` nodes. `tais_halt_reason` field added to `(:StateNode)` for cold-start TAIS EMERGENCY detection.

> **Fix applied (v2.2):** `(:Session)` node fully defined. `Tool` and `Skill` are now distinct node types (see Section 4.2 for duality resolution). `Skill` carries `VERSION_OF` for versioning; native `Tool` nodes are unversioned.

**Significance Score Decay (L3):**
`Task`, `MemoryNode`, and `Skill` nodes carry `significance_score` using the same `S = (R × F) + £` formula and 90-day rolling window as L2. Nodes below `S < 0.2` are flagged for archival to cold storage. Newly promoted nodes with `promoted_at` within the last 30 days are immune from decay audit (see Section 3.3 Grace Period).

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


### 3.4.1 FalkorDB Failure Handling

> **Fix applied (v2.4):** FalkorDB failure handling defined. Mirrors the Redis failure handling pattern.

**Failure Detection:**
FalkorDB connectivity checked by Continuity Engine health check (every 5 minutes) and on every write attempt. Failed write triggers `MEMORY_FAIL` (Appendix B) with `source: falkordb`.

**During ASD Shadow Writes (FalkorDB Down):**
```
Step 1: Redis write completes normally (synchronous, primary -- unaffected)
Step 2: Redis ACK returned to agent -- agent continues
Step 3: FalkorDB async write fails -> MEMORY_FAIL logged to audit trail
Step 4: Failed write queued in Redis list falkordb:retry_queue (1h TTL per item)
Step 5: Continuity Engine retry loop: attempt FalkorDB writes every 60 seconds
Step 6: On FalkorDB recovery: drain retry_queue in order (oldest first)
Step 7: Log recovery event to audit trail
```

**ASD is NOT affected** -- Redis remains authoritative. FalkorDB failure degrades graph-based cold-start recovery and L3 retrieval only.

**During ISEC Operation (FalkorDB Down):**
```
Pass 3 (L3 Enrichment): Skip entirely. Log "FalkorDB unavailable -- Pass 3 deferred."
Pass 4 (Decay Audit -- L3 portion): Skip L3 node decay recalculation.
ISEC continues Passes 1, 2, 5 normally.
On FalkorDB recovery: ISEC Pass 3 and L3 Pass 4 run on next idle cycle.
```

**During Cold-Start Resumption (FalkorDB Down):**
```
Step 2 (FalkorDB StateNode query) fails:
-> CRITICAL alert: "FalkorDB unavailable -- cannot read task checkpoint."
-> Start in interactive-only mode, no autonomous execution
-> Do not attempt to resume; data integrity cannot be guaranteed
-> On FalkorDB recovery: re-run startup sequence
```

**Degraded Mode Summary:**

| FalkorDB Failure Scenario | Impact | Degraded Behavior |
|---|---|---|
| ASD shadow write fails | None on live operation | Queue for retry; Redis authoritative |
| ISEC Pass 3 (L3 enrichment) | L3 graph not updated | Deferred to next idle cycle |
| Cold-start resumption | Cannot verify checkpoint | Interactive-only mode; user notified |
| L3 retrieval (context injection) | L3 context slot empty | L2 slots fill the gap; graceful degradation |

**`falkordb:retry_queue` Schema:**
```json
{
  "write_id": "uuid",
  "queued_at": "ISO-8601",
  "operation": "CREATE_NODE | CREATE_EDGE | UPDATE_NODE",
  "cypher": "MERGE (:StateNode {...}) ...",
  "session_id": "sess-...",
  "retry_count": 0
}
```
Maximum 3 retries per item. After 3 failures -> discard, log as `MEMORY_FAIL` with `permanent: true`.

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


> **Fix applied (v2.4):** MR-1 classification mechanism, entity extraction pipeline, and observability metrics defined.

#### 3.5.1 MR-1 Classification Mechanism

MR-1 routing decisions are made by a **rule-based + lightweight heuristic engine** -- NOT an inference call. Must complete in < 10ms.

**Classification Rules (evaluated in order):**

```python
def classify_for_routing(data: dict) -> str:
    if data.get("content_type") == "asd_state":
        return "ASD_DIRECT"
    ephemeral_types = {"tool_output_raw", "scratchpad", "continuation", "turn"}
    if data.get("content_type") in ephemeral_types:
        return "L1_EPHEMERAL"
    if data.get("content_type") in {"factual_knowledge", "reusable_insight"}:
        return "L2_SEMANTIC"
    if data.get("content_type") == "entity_relationship":
        return "L3_GRAPH"
    if data.get("source") == "reflection" and data.get("confidence", 0) > 0.85:
        return "L2_SEMANTIC"
    if data.get("source") == "task_terminal" and data.get("status") == "DELIVERED":
        return "L2_SEMANTIC"
    return "DISCARD"
```

**Key principle:** The model does NOT make routing decisions. The model uses the `memory_write` tool (Section 2.9) with an explicit `content_type` parameter. That declaration drives routing deterministically.

#### 3.5.2 Entity Extraction Pipeline

Entity extraction runs as part of MR-1's L2+L3 promotion path.

**Extraction Mechanism: Rule-Based + Optional NER Hybrid**
```
Stage 1: Regex-based NER (synchronous, < 5ms)
  Extract: dates, URLs, email addresses, code identifiers, version numbers,
  file paths, IP addresses, currency amounts, percentage values

Stage 2: spaCy NER (if installed; optional; < 50ms)
  Extract: PERSON, ORG, PRODUCT, GPE, EVENT
  Fallback: skip Stage 2 if spaCy not installed

Stage 3: Keyword extraction (TF-IDF-style heuristic, < 10ms)
  Extract top-5 noun phrases from content as CONCEPT entities

Stage 4: Relationship inference (rule-based only)
  "X uses Y" -> (X)-[:USED_TOOL]->(Y)
  "X produces Y" -> (X)-[:PRODUCED]->(Y)
  "X depends on Y" -> (X)-[:DEPENDS_ON]->(Y)
```

**Entity Type Taxonomy:**

| Type | Description | Examples |
|---|---|---|
| `PERSON` | Human name | "Alice", "Bob" |
| `ORG` | Organization | "Anthropic", "OpenAI" |
| `PRODUCT` | Software/product | "Redis", "Qdrant" |
| `GPE` | Geopolitical entity | "Singapore", "EU" |
| `EVENT` | Named event | "2026 Q2 review" |
| `DATE` | Date/time expression | "April 2026" |
| `URL` | Web address | "https://example.com" |
| `CODE_ID` | Code identifier | "task_id", "session_id" |
| `VERSION` | Version string | "v2.4", "Python 3.12" |
| `FILE_PATH` | File system path | "/etc/config.yaml" |
| `CONCEPT` | Abstract keyword | "exponential backoff" |

**Entity Extraction Example:**

Input: "Used httpx 0.27.0 to call the Stripe API at https://api.stripe.com -- rate limited to 100 req/min"

Extracted:
```json
[
  {"type": "PRODUCT", "value": "httpx"},
  {"type": "VERSION", "value": "0.27.0"},
  {"type": "PRODUCT", "value": "Stripe API"},
  {"type": "URL", "value": "https://api.stripe.com"},
  {"type": "CONCEPT", "value": "rate limiting"}
]
```

#### 3.5.3 MR-1 Observability Metrics

```
mr1_routing_decisions_total{destination="L1_EPHEMERAL|L2_SEMANTIC|L3_GRAPH|DISCARD"}
mr1_discard_rate_percent
mr1_classification_latency_ms
mr1_entity_extraction_latency_ms
mr1_bypass_events_total{reason="ASD_DIRECT"}
```

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


> **Fix applied (v2.4):** ISEC Pass 1 pattern identification mechanism defined concretely.

**Pass 1 Concrete Mechanism:**
Pattern identification uses a **rule-based heuristic pipeline** -- NOT a model inference call:

```python
from collections import Counter

def identify_promotion_candidates(session_logs):
    candidates = []
    # Heuristic 1: Tool results that led to DELIVERED sub-tasks
    for log in session_logs:
        if log["type"] == "tool_result" and log.get("subtask_status") == "DELIVERED":
            candidates.append({"source": log, "reason": "successful_tool_result",
                               "confidence": log.get("confidence", 0.0)})
    # Heuristic 2: Explicit memory_write calls from 9B model
    for log in session_logs:
        if log["type"] == "memory_write" and log.get("content_type") in ("factual_knowledge", "reusable_insight"):
            candidates.append({"source": log, "reason": "explicit_memory_write", "confidence": 1.0})
    # Heuristic 3: Recurring patterns -- same tool+query used successfully 3+ times
    tool_hashes = Counter(hash_tool_call(log) for log in session_logs if log["type"] == "tool_call")
    for h, count in tool_hashes.items():
        if count >= 3:
            candidates.append({"source": get_log_by_hash(h), "reason": "recurring_pattern", "confidence": 0.90})
    return [c for c in candidates if c["confidence"] > 0.85]
```

**ISEC Multi-Pass Pipeline:**

```
Pass 1 — L1 Review
  Read all session logs in Redis from completed sessions
  Identify: recurring patterns, established logic, reusable conclusions
  Tag candidates for promotion to L2
  Checkpoint progress to isec:progress in Redis

Pass 2 — L2 Deduplication  [requires nomic-embed-text loaded]
  Check VRAM headroom before loading embedding model
  Find Qdrant vectors with cosine similarity > 0.97
  Merge duplicates → retain highest-confidence version
  Recalculate S scores for merged nodes
  Checkpoint progress to isec:progress in Redis

Pass 3 — L3 Enrichment
  For each promoted L2 node → ensure entity nodes + edges exist in FalkorDB
  Connect new nodes to existing task/project/session graph
  Set promoted_at timestamp on all newly promoted nodes
  Checkpoint progress to isec:progress in Redis

Pass 4 — Decay Audit
  Recalculate S scores across all L2 and L3 nodes (90-day rolling window)
  SKIP nodes where now − promoted_at < 30 days (grace period)
  Flag nodes with S < 0.2 for archival
  Archive flagged nodes to cold storage → never delete originals
  Checkpoint progress to isec:progress in Redis

Pass 5 — L1 Pruning
  Purge reviewed L1 session logs confirmed promoted or discarded
  Reset session-bound TTLs are already managed by session close
```

**ISEC Interruption Behavior:**
On new task arrival during ISEC: complete the current **atomic operation** (e.g., a single vector write or a single S-score recalculation), then yield. Unload `nomic-embed-text` if it was loaded for Pass 2. ISEC resumes from where it left off on the next idle period — pass progress is checkpointed to `isec:progress` in Redis before yielding.

> **Fix applied (v2.3):** ISEC interruption behavior defined. Atomic-level yield, embedding model unloaded on yield, progress preserved in `isec:progress`. Pass 3 now sets `promoted_at` on promoted nodes. Pass 4 now skips nodes in 30-day grace period.

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

> **Fix applied (v2.2):** `llava:13b` removed from minimum tier. At 6GB with the 9B primary model loaded (~5.5GB), `llava:13b` (~8GB) is not viable. `moondream2` (~1.8GB) is the only supported vision model on 6GB hardware. VLA vision model is loaded on-demand and unloaded after the snapshot cycle when VRAM is constrained.

| Property | Value |
|---|---|
| **Snapshot Interval** | 60 seconds (24/7 monitoring mode) |
| **On-Demand** | Immediate capture on explicit tool call |
| **Output Schema** | `{elements: [], clickable_zones: [], text_content: "", session_id: ""}` |

> **Fix applied (v2.2):** `red_zones: []` removed from VLA output schema. Red Zones are defined in config and enforced as a separate lookup layer applied to VLA output — not detected by the vision model.

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

**Vision Model Keep-Alive Mode:**
If VRAM headroom remained **≥ 2.2GB for the last 3 consecutive cycles**, skip step 8 and keep the vision model loaded. Unload when headroom drops below **2.0GB** (checked before each cycle). This eliminates the 2–5 second reload overhead on hardware with sufficient VRAM margin.

> **Fix applied (v2.3):** Keep-alive mode added for VLA vision model. Model remains loaded if VRAM headroom stayed ≥ 2.2GB for 3+ consecutive cycles. Unloaded when headroom falls below 2.0GB. Reduces per-cycle overhead on 8GB+ hardware.

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
aura9 mcp stats                          # Call counts per server today
aura9 mcp set-limit <server_id> <n>      # Set daily alert threshold
aura9 mcp disable <server_id>            # Hard-block all calls to server
aura9 mcp enable <server_id>             # Re-enable a disabled server
```

**Daily Limit Enforcement:**
When daily call count for a server reaches the configured limit → emit `ALERT` to user. Calls are **not auto-blocked** — the limit is advisory only. To hard-block a server, use `aura9 mcp disable <server_id>`. A disabled server returns `TOOL_FAIL` on any call attempt, logged to the audit trail.

> **Fix applied (v2.3):** MCP call limit enforcement behavior defined. Limit is advisory (ALERT only); hard-block requires explicit `aura9 mcp disable` command. `disable` and `enable` CLI commands added.

#### 4.3.3 Code Interpreter — Subprocess Sandbox

> **Fix applied (v2.2):** Python does not run natively in WebAssembly. Pyodide (Python-in-Wasm) has significant limitations: no threading, restricted stdlib, ~10MB overhead, and no native extension support. The execution sandbox uses a **subprocess isolation model** instead.

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
- Crypto: `cryptography`
- Files: `openpyxl`, `pypdf`, `pillow`

> **Note:** Standard library modules (`re`, `json`, `hashlib`, `os`, `sys`, etc.) are always available — they are part of Python itself and do not require installation in the sandbox image.

> **Fix applied (v2.3):** `hashlib` moved from Approved Packages list to stdlib note. `hashlib` is a Python standard library module and is always available without being "installed." `re` and `json` also moved to stdlib note for consistency.

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


> **Fix applied (v2.4):** Pre-Processor circular dependency resolved. Ambiguity score and estimated_complexity computation defined.

#### Execution Model -- Resolving the Circular Dependency

The Pre-Processor uses a **two-tier classification** approach:

**Tier 1: Rule-Based Fast Path (TRIVIAL detection, no inference)**
```python
import re

def is_trivial(user_input: str) -> bool:
    if len(user_input.split()) > 25:
        return False
    tool_keywords = {"search", "find", "execute", "run", "calculate", "fetch",
        "create", "build", "generate", "analyze", "compare", "download",
        "install", "deploy", "schedule", "send", "open", "screenshot"}
    if any(kw in user_input.lower() for kw in tool_keywords):
        return False
    trivial_patterns = [
        r"^(hi|hello|hey|good morning|good evening)[\s!.?]*$",
        r"^what (time|day|date) is it",
        r"^(yes|no|ok|okay|sure|proceed|continue|stop|cancel)[\s!.?]*$",
        r"^(status|health|ping)[\s!.?]*$",
    ]
    return any(re.match(p, user_input.lower()) for p in trivial_patterns)
```

**Tier 2: 9B Model Inference (STANDARD task -- Mission Manifest generation)**
When Tier 1 determines NOT trivial, the **9B model** is invoked once using the Pre-Processor Prompt (Section 2.8.3). The model outputs the full Mission Manifest including `estimated_complexity`.

This resolves the circular dependency: the 9B model is the "who" for STANDARD tasks. The Pre-Processor is not a separate model -- it is a constrained prompt mode of the primary 9B kernel. The "bypass for trivial tasks" saves an inference call precisely for trivial tasks (detected rule-based). For STANDARD tasks, one inference call produces the entire Mission Manifest.

#### Ambiguity Score Computation

The ambiguity score is computed by the 9B model as part of Mission Manifest generation. The orchestrator applies a **rule-based override** if the model's score appears miscalibrated:

```python
def compute_ambiguity_score(user_input: str, model_score: float) -> float:
    rule_score = 0.0
    input_lower = user_input.lower()
    if any(p in input_lower for p in ["fix it", "update it", "change it", "do it"]):
        rule_score = max(rule_score, 0.70)
    if any(w in input_lower for w in ["some", "a few", "various", "certain"]):
        rule_score = max(rule_score, 0.55)
    if any(w in input_lower for w in ["better", "improve", "enhance", "optimize"]) and \
       not any(w in input_lower for w in ["by", "from", "to", "%", "ms", "seconds"]):
        rule_score = max(rule_score, 0.65)
    return max(model_score, rule_score)
```

**Threshold:** `ambiguity_score > 0.60` -> one targeted clarification question.

#### estimated_complexity Computation

`estimated_complexity` per sub-task is computed by the 9B model during Manifest generation, guided by the complexity heuristic in Section 2.8.3. **Mission-level complexity** = arithmetic mean of all sub-task values.

**TAIS Degraded Mode:**
During TAIS COOLDOWN or EMERGENCY, the Pre-Processor operates in **degraded mode**: tasks are accepted and queued but classified as `task_class: DEFERRED`. No inference is performed and no Mission Manifest is assembled until TAIS returns to NORMAL or THROTTLE. Deferred tasks are surfaced to the user with their queue position and the current TAIS status.

> **Fix applied (v2.3):** Pre-Processor TAIS fallback defined. Prevents Pre-Processor inference calls from executing during thermal protection events. Users receive acknowledgment that their task is deferred rather than a silent hang.

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
│     └─ promoted_at: ISO-8601 timestamp recorded      │
└───────────────────────────────────────────────────────┘
```


> **Fix applied (v2.4):** Skill interface contract, blueprint schema, test framework, and loading mechanism fully defined.

#### 4.4.1 Skill Interface Contract

Every synthesized Skill must conform to:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class AuraSkill(ABC):
    SKILL_ID: str
    SKILL_VERSION: str
    DESCRIPTION: str
    TAGS: list[str]
    TOOL_SCHEMA: Dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        # Returns: {"success": bool, "result": Any, "error": str | None}
        pass

    def validate_inputs(self, **kwargs) -> bool:
        return True
```

**Skills may NOT:** import unapproved modules, make network calls without Tier-2, write outside `/tmp`, call other Skills directly, or modify global state outside their return dict.

#### 4.4.2 Qdrant Skill Library Payload Schema

```json
{
  "skill_id": "fetch-with-retry",
  "version": "v1",
  "description": "HTTP fetch with exponential backoff retry logic",
  "tags": ["http", "retry", "networking"],
  "maturity": "EXPERIMENTAL",
  "trust_level": "sandbox_only",
  "created_at": "2026-04-17T10:00:00Z",
  "promoted_at": "2026-04-17T10:00:00Z",
  "successful_uses": 0,
  "failure_count": 0,
  "last_used_at": null,
  "significance_score": 0.0,
  "session_id": "sess-...",
  "source_code": "class FetchWithRetrySkill(AuraSkill): ...",
  "tool_schema": {
    "type": "function",
    "function": {
      "name": "fetch-with-retry",
      "description": "HTTP fetch with exponential backoff retry logic",
      "parameters": {
        "type": "object",
        "properties": {
          "url": {"type": "string"},
          "max_retries": {"type": "integer", "default": 3}
        },
        "required": ["url"]
      }
    }
  },
  "test_vectors": [
    {
      "input": {"url": "https://example.com", "max_retries": 2},
      "expected_keys": ["success", "status_code"],
      "expected_success": true
    }
  ]
}
```

Dense embedding vector (768-dim) computed from: `skill_id + " " + description + " " + " ".join(tags)`.

#### 4.4.3 Test Vector Schema & Validation Framework

```python
from pydantic import BaseModel
from typing import Any, Optional

class TestVector(BaseModel):
    input: dict
    expected_keys: list[str]
    expected_success: bool
    expected_output_contains: Optional[str] = None
    expected_output_type: Optional[str] = None
    max_execution_ms: int = 5000

class SkillTestSuite(BaseModel):
    skill_id: str
    skill_version: str
    vectors: list[TestVector]   # At least 3 required for promotion
    deterministic: bool
```

**What "100% Validation" Means:**
- **Deterministic skills:** ALL vectors must pass ALL assertions (keys, success flag, output type, timing).
- **Non-deterministic skills:** Structural assertions must pass. `expected_output_contains` is soft check.
- 100% validation = all **hard** assertions pass across all test vectors.

**EXPERIMENTAL vs. VALIDATED (RC-43):** Forge validation covers **synthetic test vectors only**. `VALIDATED` requires >= 10 successful production uses with 0 failures.

**Test Vector Storage:** Qdrant skill payload `test_vectors` field + `./skills/tests/{skill_id}_{version}_vectors.json`.

#### 4.4.4 Skill Loading at Runtime

```python
import types

def load_skill_from_source(source_code: str, skill_id: str):
    module_name = f"aura9_skill_{skill_id.replace('-', '_')}"
    module = types.ModuleType(module_name)
    exec(compile(source_code, module_name, "exec"), module.__dict__)
    skill_classes = [
        obj for obj in module.__dict__.values()
        if isinstance(obj, type) and issubclass(obj, AuraSkill) and obj is not AuraSkill
    ]
    if not skill_classes:
        raise ValueError(f"No AuraSkill subclass found in skill {skill_id}")
    return skill_classes[0]()
```

#### 4.4.5 Surfacing Available Skills to the Model

At inference time, the Tool/Skill Selection Engine queries `skill_library`:

```python
def get_relevant_skills(task_description: str, top_k: int = 3) -> list[dict]:
    results = hybrid_search(qdrant_client, "skill_library", task_description, top_k=top_k)
    return [
        point.payload["tool_schema"]
        for point in results
        if point.payload["maturity"] not in ("DEPRECATED", "QUARANTINED")
    ]
```

Skills above 0.75 similarity added to `tools` array. `CORE` maturity always included.

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
| `EXPERIMENTAL` | Newly synthesized, < 10 successful uses | Sandbox only |
| `VALIDATED` | > 10 successful uses, 0 failures | Standard access |
| `TRUSTED` | > 50 uses, < 2% failure rate | Full access |
| `CORE` | Manually promoted by user | Always loaded |
| `DEPRECATED` | Superseded by newer version | Read-only, 7-day retention |
| `QUARANTINED` | 3+ task failures | Suspended pending review |

> **Fix applied (v2.3):** `EXPERIMENTAL` criteria changed from `< 5 uses` to `< 10 successful uses`. This closes the gap between EXPERIMENTAL (previously `< 5 uses`) and VALIDATED (`> 10 uses`) where a Skill with 6–10 uses belonged to neither category. Skills remain EXPERIMENTAL until they accumulate ≥ 10 successful uses with 0 failures.

---

## 5. Autonomous Task Orchestration

### 5.1 Aura State Daemon (ASD)

The ASD is a persistent JSON state tree in Redis — the **absolute source of truth** for Aura-9's current mission. Updated exclusively via Precision Planner Mode (direct write, bypasses MR-1).

**Live State:** `asd:state` (Redis, no TTL)
**Shadow:** FalkorDB `StateNode` (async per write, see Section 3.4)
**Snapshots:** `asd:checkpoint:{task_id}:{ckpt_id}` (Redis, 7-day TTL)

**ASD State Schema:** See Section 2.6 Precision Planner Mode locked output schema.

**IDLE State:**
When a task reaches a terminal state (`DELIVERED`, `ESCALATED`, `FAILED`), ASD transitions to `IDLE` via one final Precision Planner write. `IDLE` is the signal for ISEC activation (see Section 3.6). The `asd:state` key is retained in Redis at `IDLE` status until a new task begins.

> **Fix applied (v2.3):** `IDLE` formally defined as a valid ASD status. Transition to `IDLE` on task terminal state documented. ISEC activation tied to `ASD status == IDLE` (as referenced in Section 3.6).

### 5.2 Task Lifecycle

```
CREATED → PLANNED → EXECUTING → [CORRECTING] → VERIFYING → DELIVERED
                                      ↑                |
                                      └────────────────┘
                                  (up to 3 cycles, class-aware)

Event (not a state): CHECKPOINT_SAVED — emitted during EXECUTING, CORRECTING
                     Does not change task status.

Terminal states:
  DELIVERED  — Successful completion → ASD transitions to IDLE
  ESCALATED  — Correction limit exceeded, handed to user → ASD transitions to IDLE
  FAILED     — Unrecoverable error, full diagnostic generated → ASD transitions to IDLE

Pause states:
  PAUSED     — Human gate encountered, awaiting input
  SUSPENDED  — Gate timeout exceeded (> 120 min), resources released
  BLOCKED    — Dependency unavailable, waiting

Special state:
  IDLE       — No active task; ISEC activation signal
```

> **Fix applied (v2.2):** `CHECKPOINTED` removed as a task status. Checkpointing is an **event** (`CHECKPOINT_SAVED`) emitted during execution — the task status remains `EXECUTING`. The ASD `checkpoint_required` flag signals that a checkpoint should be written at the next safe point.

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

**Concurrent Gate Coalescing:**
If multiple sub-tasks require human gates within a **60-second window**, they are coalesced into a single **Gate Brief** listing all decision points together. The agent enters `PAUSED` for the **entire mission** (not per sub-task) until the user responds. The coalesced Gate Brief lists each gate item with its sub-task ID, decision point, options, and recommendation.

> **Fix applied (v2.3):** Concurrent gate coalescing policy added. Prevents users from receiving multiple simultaneous Gate Briefs from parallel sub-tasks. Single `PAUSED` state for the full mission during coalesced gate review.

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
2. Query FalkorDB: MATCH (s:StateNode)
                   WHERE s.status IN ['EXECUTING', 'PAUSED', 'SUSPENDED', 'BLOCKED', 'CORRECTING']
                   RETURN s ORDER BY s.timestamp DESC LIMIT 1
3. If found:
   a. Check s.tais_halt_reason:
      - If tais_halt_reason == 'EMERGENCY':
        → Surface CRITICAL alert: "Previous session was halted by TAIS EMERGENCY.
          Confirm GPU is safe before resuming autonomous mode. [Y/N]"
        → Wait for explicit user confirmation
        → If user declines → start in interactive-only mode, no autonomous execution
      - Otherwise → proceed automatically
   b. Status-specific resumption:
      - EXECUTING / CORRECTING / VERIFYING:
        → Restore Redis ASD from FalkorDB StateNode
        → Log interruption gap (last_checkpoint → now)
        → Resume from next_action in StateNode
      - PAUSED:
        → Restore Redis ASD from FalkorDB StateNode
        → Re-surface Gate Brief to user
        → Enter PAUSED state awaiting response
      - SUSPENDED:
        → Re-acquire GPU resources before resuming
        → Restore Redis ASD from FalkorDB StateNode
        → Re-surface Gate Brief (expired) to user
        → Enter PAUSED state awaiting response
      - BLOCKED:
        → Re-check dependency availability
        → If dependency now available → restore and resume
        → If still unavailable → enter BLOCKED state, notify user
4. If not found:
   → Normal interactive startup
```

> **Fix applied (v2.3):** Cold-start resumption query expanded from `['EXECUTING','PAUSED']` to `['EXECUTING', 'PAUSED', 'SUSPENDED', 'BLOCKED', 'CORRECTING']`. Status-specific resumption logic added for each state. TAIS EMERGENCY detection added: if `tais_halt_reason == 'EMERGENCY'`, explicit user confirmation required before re-entering autonomous mode. `tais_halt_reason` persisted in `StateNode` (see Section 3.4).

### 5.6 Continuity Engine

The Continuity Engine is a **Python daemon process** (`continuity_daemon.py`) that runs alongside the main agent loop. It is responsible for all scheduled background operations that maintain 24/7 operational integrity.

**Responsibilities:**

| Interval | Action |
|---|---|
| Every 15 minutes | Write ASD checkpoint (`asd:checkpoint:{task_id}:{ckpt_id}`) |
| Every 24 hours | FalkorDB graph snapshot to `./backups/falkordb-{date}.dump` |
| Every 24 hours | Qdrant snapshot of all collections to `./backups/qdrant-{date}/` |
| Every 5 minutes | Health check sweep (all services, including stale session cleanup) |
| On TAIS EMERGENCY | Trigger immediate checkpoint before halt |
| On SUSPENDED state | Trigger resource release sequence |

> **Fix applied (v2.3):** Qdrant daily snapshot added to Continuity Engine responsibilities. Stale session cleanup (24h max TTL) added to 5-minute health check sweep.

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
aura9 shutdown --force       # Steps 3, 4 (10-second timeout), and 10 only (emergency use)
```

> **Fix applied (v2.3):** Force shutdown updated to include steps 3, 4 (with a 10-second BGSAVE timeout), and 10. If BGSAVE does not complete within 10 seconds, shutdown proceeds and potential Redis data loss is accepted — the FalkorDB shadow provides recovery for ASD state. This ensures the ASD checkpoint (step 3) is flushed to disk when possible without blocking indefinitely.

### 5.8 Session Lifecycle

> **Fix applied (v2.4):** Session lifecycle fully defined.

**Session Start:**
1. **`python main.py` (fresh start):** New `session_id` generated at boot.
2. **`python main.py --resume` (cold-start):** Session restored from most recent StateNode. No new session created.
3. **First task after 24h inactivity:** If `last_active` > 24 hours, a new session is created.

**Multiple Tasks in One Session:**
YES -- multiple tasks can run in the same session. ASD cycles IDLE -> task lifecycle -> IDLE between tasks.

**Session End:**
1. `aura9 shutdown --graceful` followed by a fresh restart.
2. Session `last_active` > 24 hours (archived by Continuity Engine, new session on next task).
3. Future: `aura9 session new` forces a new session ID.

**Session vs. Task ID:**
```
Session (sess-abc-1744320000)
  Task 1 (task-uuid-1)  [DELIVERED]
  Task 2 (task-uuid-2)  [DELIVERED]
  Task 3 (task-uuid-3)  [EXECUTING]   <- current
```

**SUSPENDED State + Session TTL (RC-44):**
When a task enters `SUSPENDED` state, the session-bound L1 Redis TTL is **extended to 7 days** (matching ASD checkpoint TTL). Prevents L1 data from expiring while waiting for user return.

**Rate Limiting Human Gates (P2 #47):**
Maximum **5 human gates per mission**. On 6th gate trigger, the entire mission is escalated with all prior gate decisions.

### 5.9 Race Condition & Consistency Resolutions

> **Fix applied (v2.4):** All identified race conditions and contradictions resolved.

#### RC-36: ASD Coalescing vs. Watchdog Race

Each ASD write carries a monotonically increasing **sequence number** (`seq: integer`).
- Watchdog evaluates individual outputs as they arrive; coalesced ASD write occurs at window close.
- **Watchdog 3-second grace period** on new task start accommodates coalescing window.
- Watchdog flags during coalescing are queued and evaluated against final coalesced state.

#### RC-37: ISEC Pass 2 + TAIS VRAM Conflict

ISEC embedding model (`nomic-embed-text`, ~274MB) and TAIS quant switch share VRAM:
- **TAIS takes priority** -- always.
- ISEC checks TAIS status before loading `nomic-embed-text`: if not NORMAL, defer.
- On TAIS THROTTLE during ISEC Pass 2: immediately unload embedding model, checkpoint, yield.
- TAIS waits up to **10 seconds** for ISEC to unload.

#### RC-38: Parallel Sub-Tasks Inference Starvation

See Section 7.7 Inference Queue -- round-robin fairness within NORMAL priority tier. Correction loops deprioritized.

#### RC-39: Watchdog Hard Kill During ASD Shadow Write

On hard-kill trigger:
1. Wait for in-flight FalkorDB shadow write (max **2-second timeout**).
2. If timeout: mark FalkorDB write as potentially inconsistent in `falkordb:retry_queue`.
3. Proceed with hard-kill checkpoint to Redis.
4. Halt inference.

#### RC-40: Gate Coalescing + Correction Loop Deadlock

Correction loops must complete current cycle before gate coalescing finalizes PAUSED. Gate window **extended by 30 seconds** if any sub-task is CORRECTING. After 30 seconds, force PAUSED.

#### RC-42: MR-1 DISCARD vs. Audit Trail PII

Resolved in Section 6.6: Audit trail logs routing decision and data hash (`SHA-256(content)`), NOT full content.

#### RC-43: Skill Forge 100% Validation vs. EXPERIMENTAL

Resolved in Section 4.4.3: Forge validation covers **synthetic test vectors only**. VALIDATED requires >= 10 production uses.


---

## 6. Security & Operational Guardrails

### 6.0 Authentication & Authorization

> **Fix applied (v2.4):** Authentication and authorization model defined.

#### 6.0.1 CLI Authorization

Single-user, local-first system. CLI commands authorized by **filesystem ownership** (`os.getuid()` comparison). No password or token required for local operation.

**CLI Command Tiers:**

| Tier | Commands | Auth |
|---|---|---|
| Tier 0 -- Read-only | `status`, `task show`, `watchdog log`, `tais status`, `vram status`, `memory score`, `mcp stats`, `skill list`, `audit` | UID match only |
| Tier 1 -- State-modifying | `checkpoint --now`, `memory pin/unpin`, `memory isec`, `mcp set-limit`, `skill rollback`, `qdrant snapshot`, `graph snapshot` | UID match + confirmation prompt |
| Tier 2 -- Destructive | `shutdown`, `mcp disable/enable`, `qdrant restore`, `graph restore`, `memory restore` | UID match + typed confirmation (type "CONFIRM") |

#### 6.0.2 Inter-Process Communication

Redis AUTH password stored in `config/aura9.config.yaml` under `redis.password` (file permissions `0600`). Redis binds to `127.0.0.1` only.

#### 6.0.3 MCP Server Credentials

MCP credentials stored in `./secrets/mcp-credentials.yaml` (permissions `0600`, `.gitignore`). Format: `{server_id: {auth_type: "api_key|oauth2", credential: "<value>"}}`.

#### 6.0.4 Watchdog Integrity Protection

Watchdog config/model files reside on a **read-only filesystem partition** (or `chattr +i`). The 9B model process has no write access.

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

**Deterministic vs. Model Checks:**

| Check Type | Method | Rationale |
|---|---|---|
| Schema validation | JSON schema validator (deterministic) | Binary pass/fail — no model judgment needed |
| Loop detection | Hash comparison of (action, inputs) tuples (deterministic) | Exact repetition detection is hash-reliable |
| Gate bypass | Rule-based state machine check (deterministic) | Verifiable against ASD state — no inference needed |
| Capability creep | Permission tier lookup (deterministic) | Static configuration comparison |
| Drift detection | 1.5B model inference | Subjective off-mission judgment requires model |
| Output toxicity | 1.5B model inference | Subjective safety assessment requires model |

> **Fix applied (v2.3):** Schema validation, loop detection, gate bypass, and capability creep checks now use **deterministic code** rather than the 1.5B model. Only drift detection and toxicity assessment — which require subjective judgment — use the model. This improves reliability for binary checks and reduces the impact of the capability mismatch between the 1.5B Watchdog and the 9B primary. Known limitation: drift detection false-negative rate is elevated by the model size gap; periodic manual review of Watchdog alert logs is recommended.


> **Fix applied (v2.4):** Watchdog inference scheduling defined for shared Ollama instance.

**Watchdog Inference Scheduling:**

| Hardware Tier | Watchdog Runtime | Ollama Port |
|---|---|---|
| 6GB VRAM | CPU -- separate Ollama instance | 11435 |
| 8GB+ VRAM | GPU -- shared Ollama instance | 11434 |

**On 6GB hardware:** Watchdog runs on CPU via its own Ollama instance on port **11435**. CPU inference (~10-30s per eval) avoids VRAM competition.

**On 8GB+ hardware:** Both models share port 11434. Watchdog queued behind primary requests.

**Model Load Conflict Prevention (during TAIS quant switch):**
```
1. TAIS sends THROTTLE signal
2. Watchdog pauses new inference submissions
3. TAIS waits for in-flight Watchdog inference (max 60s)
4. Quant switch proceeds
5. Watchdog resumes after primary model reloaded
6. If Watchdog GPU-resident: unload Watchdog -> load Q4_K_M -> reload Watchdog
```

**Watchdog Restart Buffering:**
During Watchdog restart (see Section 6.2), the main agent continues to publish inference outputs to the `ipc:watchdog:monitor` pub/sub channel AND simultaneously appends them to the `watchdog:buffer` Redis list (5-minute TTL). On Watchdog restart completion, the Watchdog drains `watchdog:buffer` before resuming normal pub/sub subscription, ensuring no outputs are missed during the restart window.

> **Fix applied (v2.3):** Watchdog restart buffer defined. Dual-write to `watchdog:buffer` Redis list during restart. Buffer drained before pub/sub resume. `watchdog:buffer` key added to Redis namespace table (Section 3.2).

### 6.2 Watchdog Liveness Check

**Mechanism:** Watchdog process refreshes Redis key `watchdog:heartbeat` (TTL: 90s) every **30 seconds**. If the key expires, the liveness check triggers.

**Response to Heartbeat Timeout:**
```
1. CRITICAL alert to user
2. Autonomous execution suspended immediately
3. Interactive-only mode (no autonomous tool execution)
4. Main agent begins dual-writing to watchdog:buffer (Redis list, 5-min TTL)
5. Auto-restart Watchdog (up to 3 attempts, 30s between)
6. On successful restart:
   a. Watchdog drains watchdog:buffer before resuming pub/sub subscription
   b. Agent stops dual-writing to watchdog:buffer
   c. Autonomous mode resumes after heartbeat confirmed
7. If 3 restarts fail → full halt, manual restart required
```

> **Fix applied (v2.3):** Buffer drain step added to restart sequence. Steps 4 and 6a/6b document the dual-write and drain pattern. Explicit: agent stops buffering after Watchdog confirms it has drained and subscribed.

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


> **Fix applied (v2.4):** Audit trail storage format defined as structured JSON Lines (JSONL).

**Storage Format:** JSONL (JSON Lines) -- one JSON object per line, newline-delimited. Enables streaming writes and `grep`/`jq` querying.

**Query Mechanism:**
```bash
jq 'select(.task_id == "task-uuid")' /mnt/c/aura9-audit/audit.log
jq 'select(.action_type == "watchdog_alert" and .timestamp > "2026-04-17")' /mnt/c/aura9-audit/audit.log
aura9 audit --task-id <uuid>
aura9 audit --action-type gate_triggered --last 24h
```

**Metrics format:** Prometheus exposition format at `localhost:9001/metrics`.

### 6.6 PII Scrubbing

> **Fix applied (v2.4):** PII detection mechanism, scrubbing rules, and pipeline placement defined.

**What Constitutes PII:**

| Category | Examples | Detection |
|---|---|---|
| Email address | `user@example.com` | Regex |
| Phone number | `+1-555-123-4567` | Regex |
| Physical address | "123 Main St, Seattle WA" | Regex + spaCy GPE |
| Financial account | Credit card, IBAN, routing number | Regex |
| Government ID | SSN, passport number | Regex |
| Full name + identifier | "John Smith, DOB 1990-05-15" | Regex + spaCy PERSON |

**Scrubbing Patterns:**
```python
PII_PATTERNS = [
    (r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', '[EMAIL_REDACTED]'),
    (r'(\+\d{1,3}[-.])?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE_REDACTED]'),
    (r'\d{3}-\d{2}-\d{4}', '[SSN_REDACTED]'),
    (r'(?:\d{4}[-\s]?){3}\d{4}', '[CARD_REDACTED]'),
    (r'(?:\d{1,3}\.){3}\d{1,3}', '[IP_REDACTED]'),
]
```

**Pipeline Placement:**

| Data Flow Point | Scrubbing Applied? | Notes |
|---|---|---|
| Active session L1 Redis write | NO | Session-scoped, 24h TTL |
| L1 to L2 promotion (MR-1) | YES | Before Qdrant write |
| Audit trail write | YES | `action_detail` field scrubbed |
| MR-1 DISCARD log | YES | Decision + data hash only, NOT content (RC-42) |
| Checkpoint write (ASD) | PARTIAL | `current_objective` and `next_action` scrubbed |

### 6.7 Dual-Confirmation Gate for Financial Transactions

> **Fix applied (v2.4):** Financial transaction detection and dual-confirmation mechanism defined.

**Detection (by Zero-Trust Sanitizer at MCP Gateway level):**
```python
FINANCIAL_INDICATORS = {
    "server_id_prefix": ["stripe", "paypal", "plaid", "banking", "payment", "finance"],
    "tool_name_keywords": ["transfer", "pay", "charge", "debit", "credit", "withdraw",
                           "deposit", "send_money", "purchase", "buy", "sell", "trade"],
    "argument_keys": ["amount", "recipient_account", "to_address", "wallet_address",
                      "iban", "routing_number"],
}
```

**Dual-Confirmation Mechanism:**
```
Step 1: IMMEDIATE HALT -- MCP call blocked before execution
Step 2: ASD status -> PAUSED
Step 3: Surface to user:
  ======================================================
  !! FINANCIAL TRANSACTION GATE -- Confirmation Required
  ======================================================
  Pending action: {tool_name} on {server_id}
  Parameters: {formatted_params}

  Step 1/2 -- Type APPROVE to proceed: ___
Step 4: On "APPROVE" -> second confirmation:
  Step 2/2 -- Type the transaction amount/identifier to confirm: ___
Step 5: On match -> execute. Log FINANCIAL_GATE_APPROVED to audit trail.
Step 6: On mismatch or any other input -> CANCEL.
```

**Non-MCP Financial Actions:** Import of known payment libraries (`stripe`, `plaid`, `coinbase`) in Code Interpreter triggers `SECURITY_FAIL` immediately.


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

### 7.6 Project Structure & Module Boundaries

> **Fix applied (v2.4):** Complete file/module tree structure, process model, and entry points defined.

```
aura-9/
+-- main.py                         # Entry point, CLI args, startup
+-- pyproject.toml                  # Python project metadata and dependencies
+-- docker-compose.yaml             # Redis, Qdrant, FalkorDB stack
+-- config/
|   +-- aura9.config.yaml           # All configurable parameters (Appendix G)
+-- secrets/
|   +-- mcp-credentials.yaml        # MCP server credentials (.gitignore)
+-- logs/
+-- backups/
+-- archive/
+-- skills/
|   +-- tests/                      # Exported test vectors
|   +-- interface.py                # AuraSkill ABC definition
+-- migrations/                     # FalkorDB schema migrations (Appendix I)
+-- src/
|   +-- __init__.py
|   +-- agent/
|   |   +-- orchestrator.py         # Main agent loop, 4-Phase CoT
|   |   +-- pre_processor.py        # Mission Manifest generation
|   |   +-- reflection.py           # Confidence score, verification
|   |   +-- context_manager.py      # Context Budget Manager
|   |   +-- session.py              # Session lifecycle
|   +-- memory/
|   |   +-- l1_redis.py             # Redis operations, key namespaces
|   |   +-- l2_qdrant.py            # Qdrant operations, hybrid search
|   |   +-- l3_falkordb.py          # FalkorDB operations, Cypher queries
|   |   +-- memory_router.py        # MR-1 classification, entity extraction
|   |   +-- isec.py                 # ISEC multi-pass daemon
|   |   +-- significance.py         # S score computation
|   +-- tais/
|   |   +-- tais.py                 # Temperature polling, quant switch
|   +-- watchdog/
|   |   +-- watchdog.py             # Output monitoring, checks
|   +-- tools/
|   |   +-- vla_bridge.py           # Screenshot, vision model
|   |   +-- mcp_gateway.py          # MCP connections, tier enforcement
|   |   +-- code_interpreter.py     # Subprocess sandbox
|   |   +-- skill_forge.py          # Skill synthesis, validation
|   +-- security/
|   |   +-- sanitizer.py            # Zero-Trust payload interceptor
|   |   +-- pii_scrubber.py         # PII detection and scrubbing
|   |   +-- red_zone.py             # Visual Red Zone enforcement
|   +-- continuity/
|   |   +-- continuity_engine.py    # Checkpoints, snapshots, health
|   +-- asd/
|   |   +-- asd.py                  # ASD state, Precision Planner Mode
|   +-- ollama/
|   |   +-- client.py               # Ollama API client, model lifecycle
|   +-- metrics/
|   |   +-- prometheus.py           # Metrics endpoint
|   +-- audit/
|   |   +-- audit_trail.py          # Immutable audit writer
|   +-- cli/
|       +-- commands.py             # All aura9 CLI commands
+-- tests/
    +-- unit/
    +-- integration/
    +-- mocks/
```

**Process Model:**

| Component | Process Type | Concurrency Model |
|---|---|---|
| Main agent (orchestrator) | **Main process** | `asyncio` event loop |
| TAIS | **Thread** in main process | `threading.Thread` |
| Memory Router (MR-1) | **Async task** in main process | `asyncio` task |
| Watchdog (CPU tier) | **Separate process** | Subprocess, Ollama port 11435 |
| Watchdog (GPU tier) | **Thread** in main process | Shares Ollama port 11434 |
| Continuity Engine | **Thread** in main process | `asyncio`-scheduled tasks |
| ISEC | **Thread** in main process | Idle-activated |
| Prometheus metrics | **Thread** in main process | `http.server.HTTPServer` daemon |
| DAG sub-tasks | **ThreadPoolExecutor** | Max 3 workers (2 during THROTTLE) |

**Entry Points:**
```bash
python main.py                    # Fresh start
python main.py --resume           # Cold-start resumption
python main.py --task "..."       # Direct task submission
python main.py --benchmark        # Run Appendix E benchmark suite
aura9 <command>                   # CLI via pyproject.toml entry point
```

### 7.7 Ollama API Interaction Patterns

> **Fix applied (v2.4):** Ollama API endpoints, streaming, model lifecycle, and error handling defined.

**Endpoints Used:**

| Endpoint | Method | Usage |
|---|---|---|
| `/api/chat` | POST | **Primary inference** -- all 9B calls with tool support |
| `/api/generate` | POST | Model load/unload lifecycle (`keep_alive`) |
| `/api/embeddings` | POST | `nomic-embed-text` embedding generation |
| `/api/tags` | GET | Check available models |
| `/api/show` | POST | Model metadata (context size, quant) |
| `/api/ps` | GET | Currently running models and VRAM usage |

**Streaming vs. Non-Streaming:**

| Use Case | Streaming | Rationale |
|---|---|---|
| Primary 9B inference (interactive) | YES | Progressive output |
| Precision Planner Mode (ASD write) | NO | JSON must be complete |
| Pre-Processor (Mission Manifest) | NO | YAML must be complete |
| Watchdog inference | NO | Verdict must be complete |
| Embeddings | NO | Single vector response |

**Inference Queue:**
```
asyncio.PriorityQueue inside the main process.
Single consumer processes all inference requests.

Priority levels:
  0 = CRITICAL (ASD/Precision Planner)
  1 = HIGH (Pre-Processor)
  2 = NORMAL (primary CoT)
  3 = LOW (Watchdog, ISEC embedding)

Fairness: Sub-task requests use round-robin within NORMAL tier.
Correction loops deprioritized: priority = (2, 9).
```

**Ollama Health Check (P2 #49):**
```python
async def ollama_health_check() -> dict:
    try:
        start = time.monotonic()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": config.primary_model,
                    "messages": [{"role": "user", "content": "Reply with exactly: HEALTH_OK"}],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_ctx": 512}
                },
                timeout=30.0
            )
            elapsed = time.monotonic() - start
            content = response.json()["message"]["content"].strip()
            return {"healthy": "HEALTH_OK" in content, "latency_ms": elapsed * 1000}
    except Exception as e:
        return {"healthy": False, "error": str(e)}
```

Failure: 1st -> ALERT, retry 30s. 2nd -> CRITICAL, suspend autonomous. 3rd -> recommend Ollama restart.

**Ollama Crash/Restart Error Handling:**
```python
async def chat_with_retry(self, **kwargs) -> dict:
    for attempt in range(3):
        try:
            return await self._raw_chat(**kwargs)
        except httpx.ConnectError:
            if attempt < 2:
                await asyncio.sleep(5 * (attempt + 1))
                continue
            raise InferenceError("INFERENCE_FAIL: Ollama unreachable after 3 attempts")
```

### 7.8 Infrastructure Latency Degradation

> **Fix applied (v2.4):** Latency thresholds for all infrastructure services defined.

| Service | Normal | Warning | Degraded |
|---|---|---|---|
| Redis | < 1ms | 1-10ms | > 10ms |
| Qdrant | < 50ms | 50-200ms | > 200ms |
| FalkorDB | < 20ms | 20-200ms | > 200ms |
| Ollama (canary) | < 2000ms | 2000-5000ms | > 5000ms |

**Warning:** Log warning, increment Prometheus counter. No operational change.

**Degraded behavior:**

| Service | Degraded Behavior |
|---|---|
| Qdrant > 200ms | Reduce `top_k` from 5 to 3. If > 1000ms: skip L2 retrieval entirely. |
| FalkorDB > 200ms | Skip L3 context injection. If > 1000ms: skip shadow write (queue for retry). |
| Ollama > 5000ms | ALERT. If 3 consecutive: CRITICAL, suspend autonomous mode. |


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
aura9 mcp set-limit <server_id> <n>
aura9 mcp disable <server_id>            # Hard-block server
aura9 mcp enable <server_id>             # Re-enable server
aura9 qdrant snapshot                    # Manual Qdrant snapshot
aura9 qdrant restore --dir <path>        # Restore Qdrant from snapshot
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
✓ Last Qdrant snapshot — 6h ago
✓ Stale sessions cleaned — 0 expired
```

### 9.4 Command Behavior During Active Tasks

> **Fix applied (v2.4):** Behavior of CLI commands during active task execution defined.

**`aura9 memory isec --run-now` During Active Task (P2 #51):**
```
Response: "ISEC scheduled: will run when current task completes and GPU is idle.
Current task: {task_id} | Status: {status} | Queue position: 1"
```
ISEC queued via `isec:pending` Redis flag. On ASD -> IDLE, Continuity Engine starts ISEC.

**`aura9 skill rollback` During Active Skill Use (P2 #52):**
```
Response: "Rollback queued: skill {skill_id} will roll back to {target_version}
after current invocation completes."
```
Active task continues with current version. Rollback executes after invocation completes.

### 9.5 Graceful Shutdown During ISEC

> **Fix applied (v2.4):** ISEC behavior during graceful shutdown defined (P2 #53).

When `aura9 shutdown --graceful` is invoked while ISEC is running:
```
1. ISEC receives shutdown signal via ipc:continuity:trigger {"trigger_type": "SHUTDOWN"}
2. ISEC has 60-second timeout to complete current atomic operation
3. After timeout (or atomic completion):
   a. Checkpoint progress to isec:progress
   b. Unload nomic-embed-text if loaded
   c. Emit completion signal
4. Shutdown proceeds (Section 5.7 step 5)
```
Data integrity: checkpointed progress ensures next ISEC run resumes from where it stopped.


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
Grace:    now − promoted_at < 30 days → immune from decay audit regardless of S
Flagged:  S < 0.2 (and outside grace period) → marked by ISEC for archival
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
- [ ] `qwen3.5:9b-instruct-q5_k_m` pulled and listed
- [ ] `qwen3.5:9b-instruct-q4_k_m` pulled (TAIS THROTTLE fallback)
- [ ] `qwen3.5:1.5b` (Watchdog) pulled and listed
- [ ] `nomic-embed-text` pulled and listed
- [ ] Docker Compose stack healthy: Redis (6379), Qdrant (6333/6334), FalkorDB (6380)
- [ ] FalkorDB graph schema applied — all node types and relationships created
- [ ] `(:Agent)` singleton node created in FalkorDB at first boot
- [ ] `(:Session)` node creation verified in FalkorDB
- [ ] Qdrant collections initialized: `expertise`, `documentation`, `skill_library`, `past_missions`, `failure_analysis`
- [ ] Python project scaffold created, venv active, dependencies installed
- [ ] `config/aura9.config.yaml` reviewed and customized
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
- [ ] Qdrant daily snapshot job scheduled via Continuity Engine

### First Run

- [ ] Full health check — all 16 checks green (including Qdrant snapshot and stale session checks)
- [ ] Simple test task — verify full lifecycle (CREATED → DELIVERED → IDLE)
- [ ] ASD state updates visible in Redis (`asd:state`) with `task_id` and `session_id`
- [ ] `session_id` stamped on L1 keys, L2 payloads, L3 nodes, and audit entries
- [ ] `CHECKPOINT_SAVED` event emitted — verify via audit trail (not a status field)
- [ ] `sess:{session_id}:metadata` key created and `last_active` updated on each turn
- [ ] Simulate reboot — cold-start recovery resumes from checkpoint without user action
- [ ] TAIS telemetry visible on metrics endpoint
- [ ] VRAM utilization visible on metrics endpoint (independent of TAIS)
- [ ] Watchdog heartbeat visible: `redis-cli get watchdog:heartbeat`
- [ ] Kill Watchdog — confirm `CRITICAL` alert within 90s, autonomous mode suspended
- [ ] Confirm `watchdog:buffer` populated during Watchdog downtime
- [ ] Restore Watchdog — confirm buffer drained, heartbeat resumes, autonomous mode restored
- [ ] `aura9 memory score` on a promoted node — S formula confirmed, `promoted_at` present
- [ ] Tool result accessible for full session duration (session-bound TTL verified)
- [ ] Graceful shutdown — all 11 steps complete, clean audit entry logged
- [ ] FalkorDB snapshot created on 24h schedule — test manual restore
- [ ] Qdrant snapshot created on 24h schedule — test `aura9 qdrant restore`

### Ongoing Operations

- [ ] Review Watchdog alert log weekly
- [ ] Review TAIS thermal log weekly — sustained patterns
- [ ] Review VRAM utilization trend weekly
- [ ] Inspect skill_library monthly — deprecated/quarantined skills
- [ ] `aura9 memory isec --run-now` monthly — review consolidation report
- [ ] Audit MCP call counters monthly — adjust limits
- [ ] FalkorDB snapshot restore test quarterly
- [ ] Qdrant snapshot restore test quarterly
- [ ] Cold storage archive restore test quarterly

---

## Appendix A — Glossary

| Term | Definition |
|---|---|
| **APM** | Autonomous Persistence Module — the Triple-Thread memory system (Redis + Qdrant + FalkorDB) |
| **ASD** | Aura State Daemon — persistent JSON state tree in Redis, authoritative source of truth for the current mission |
| **CHECKPOINT_SAVED** | Event emitted when a checkpoint is written. Not a task status — task remains `EXECUTING` |
| **CoT** | Chain-of-Thought — step-by-step reasoning style |
| **Continuity Engine** | Python asyncio daemon managing scheduled background operations (checkpoints, snapshots, health checks, stale session cleanup) |
| **DAG** | Directed Acyclic Graph — dependency graph for parallel sub-task scheduling |
| **Gate** | Deliberate pause requiring human input before autonomous execution proceeds |
| **IDLE** | ASD status indicating no active task; signal for ISEC activation |
| **ISEC** | Idle-State Epistemic Consolidation — background daemon consolidating memory during GPU idle periods |
| **MCP** | Model Context Protocol — standardized tool and API connectivity protocol |
| **Mission Manifest** | Structured YAML task plan generated by the Pre-Processor, versioned at `manifest_version: 2.4` |
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
| Ollama (Watchdog CPU) | 11435 | HTTP | Watchdog inference (6GB tier only) |

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
| **2.3** | **All 21 v2.2 audit findings resolved.** Critical: `(:Agent {id, name, version, created_at})` singleton node added to FalkorDB schema (Section 3.4) — created at first boot, referenced by `NATIVE_TO` and `HAS_SKILL` relationships; cold-start resumption query expanded to `['EXECUTING', 'PAUSED', 'SUSPENDED', 'BLOCKED', 'CORRECTING']` with status-specific resumption logic for each state (Section 5.5); `tais_halt_reason` field added to ASD schema and `StateNode` — cold-start after TAIS EMERGENCY now requires explicit user confirmation before autonomous mode (Section 5.5, 2.6); 30-day grace period added for newly promoted L2/L3 nodes (`promoted_at: ISO-8601` field), ISEC Pass 4 skips grace-period nodes, resolving S=0.0 dead zone for new knowledge (Section 3.3, 3.4, 3.6). Important: concurrent human gate coalescing policy added — multiple gates within 60s coalesced into single Gate Brief, full mission enters PAUSED (Section 5.4); `IDLE` added to ASD status enum, IDLE transition documented on task terminal state as ISEC activation signal (Section 2.6, 5.1, 5.2); Watchdog restart buffer added — dual-write to `watchdog:buffer` Redis list during restart, drained before pub/sub resume (Section 6.1, 6.2); Qdrant daily snapshots added to Continuity Engine, `aura9 qdrant snapshot/restore` CLI commands added (Section 3.3, 5.6); EXPERIMENTAL maturity criteria changed from `< 5 uses` to `< 10 successful uses`, closing the 5–10 use gap (Section 4.5); MCP call limit enforcement defined as advisory ALERT with `aura9 mcp disable` for hard-block (Section 4.3.2). Moderate: Pre-Processor TAIS fallback — tasks held as `DEFERRED` during COOLDOWN/EMERGENCY (Section 4.3.4); 24h maximum session TTL added, `sess:{session_id}:metadata` key added, Continuity Engine handles stale session cleanup (Section 3.2); Watchdog deterministic vs. model checks table added — schema validation, loop detection, gate bypass, capability creep now use deterministic code (Section 6.1); Precision Planner state-change coalescing added — 5-second window batching for ASD events (Section 2.6); force shutdown updated to include BGSAVE with 10-second timeout before step 10 (Section 5.7); ISEC interruption behavior defined — atomic-level yield, embedding model unloaded, progress saved to `isec:progress` in Redis (Section 3.6). Minor: `priority: LOW|NORMAL|HIGH|CRITICAL` field added to Mission Manifest (Section 2.3); `hashlib` moved from Approved Packages to stdlib note (Section 4.3.3); Output Buffer overflow behavior defined — continuations stored in `sess:{session_id}:continuation:{turn_id}` (Section 2.5); VLA keep-alive mode added — model stays loaded if VRAM headroom ≥ 2.2GB for 3+ cycles, unloads at < 2.0GB (Section 4.3.1); this Appendix F entry added. Document-wide: version header updated to 2.3; `manifest_version` updated to "2.3"; health check updated to 16 checks; deployment checklist updated for Qdrant snapshots, `(:Agent)` node, session metadata, Watchdog buffer; all cross-references validated. |

| **2.4** | **All 53 v2.3 audit findings resolved.** P0: Confidence score formula (S2.7), system prompts (S2.8), tool-calling format (S2.9), MR-1 classification (S3.5.1), entity extraction (S3.5.2), Pre-Processor circular dependency (S4.3.4), FalkorDB failure handling (S3.4.1), Skill interface/testing (S4.4), project structure (S7.6), Ollama API patterns (S7.7). P1: Session lifecycle (S5.8), auth model (S6.0), PII scrubbing (S6.6), dual-confirmation gate (S6.7), Watchdog scheduling (S6.1), TAIS quant switch API (S2.2), context eviction (S2.5), checkpoint content (S2.5), RRF detail (S3.3), sig score fix (S3.3), ISEC Pass 1 (S3.6), ambiguity/complexity (S4.3.4), audit format (S6.5). Race conditions RC-36 through RC-44 resolved. P2: latency degradation (S7.8), gate limiting (S5.8), command behavior (S9.4-9.5), Ollama health (S7.7), MR-1 metrics (S3.5.3), backup alignment (S2.5). New files: config/aura9.config.yaml, docker-compose.yaml, pyproject.toml. New appendices: G (Config Schema), H (Testing Strategy), I (Graph Migration). Model: Qwen 3.5 9B confirmed (available in Ollama). |

---


---

## Appendix G — Configuration Schema

> **Fix applied (v2.4):** Unified configuration schema defined. See `config/aura9.config.yaml` for complete defaults.

### G.1 Configuration Sections

| Section | Purpose |
|---|---|
| `model` | Ollama model names and inference parameters |
| `tais` | Thermal thresholds and polling intervals |
| `memory` | L1/L2/L3 capacity, TTLs, promotion thresholds |
| `context` | Token budget allocations |
| `inference` | Queue settings, concurrency limits |
| `redis` | Connection settings |
| `qdrant` | Connection settings |
| `falkordb` | Connection settings |
| `security` | Auth, Red Zones, Sanitizer |
| `observability` | Metrics, logging, audit trail |
| `paths` | File system paths |
| `session` | Session TTL, gate limits |

### G.2 Validation Rules

All numeric fields have `min`/`max` constraints. String enum fields accept only listed values. Configuration loaded and validated at startup using Pydantic.

---

## Appendix H — Testing Strategy

> **Fix applied (v2.4):** Unit test expectations, integration test boundaries, and mock strategies defined.

### H.1 Test Framework

Python `pytest` with `pytest-asyncio`. All tests in `tests/` directory (Section 7.6).

```bash
pytest tests/unit/
pytest tests/integration/
pytest tests/ --cov=src --cov-report=html
```

### H.2 Mock Strategy

| Service | Mock | Package |
|---|---|---|
| Redis | `fakeredis` | `pip install fakeredis` |
| Ollama | Custom `MockOllamaClient` | `tests/mocks/mock_ollama.py` |
| Qdrant | Custom `MockQdrantClient` | `tests/mocks/mock_qdrant.py` |
| FalkorDB | Custom `MockFalkorDB` | `tests/mocks/mock_falkordb.py` |

### H.3 Unit Test Expectations

| Module | Key Tests |
|---|---|
| `tais.py` | Temperature transitions, sensor fail, recovery |
| `memory_router.py` | All routing paths (L1/L2/L3/DISCARD), entity extraction, ASD bypass |
| `reflection.py` | Confidence formula, edge cases (no tools, max retries), mission aggregation |
| `pre_processor.py` | Trivial detection patterns, ambiguity override, TAIS degraded mode |
| `pii_scrubber.py` | All regex patterns, scrubbing placement |
| `significance.py` | S formula, div-by-zero guard, grace period, cold-start |

### H.4 Testing TAIS Without Real GPU

```python
class MockNvml:
    _temp = 70
    def nvmlDeviceGetTemperature(self, handle, sensor): return self._temp
    def set_temperature(self, temp): self._temp = temp

def test_tais_throttle(mock_nvml):
    tais = TAIS(nvml=mock_nvml)
    mock_nvml.set_temperature(76)
    assert tais.poll_and_act() == TAISStatus.THROTTLE
```

### H.5 Integration Test Boundaries

| Test | Infrastructure |
|---|---|
| `test_asd_lifecycle.py` | Mock Ollama + fakeredis + Mock FalkorDB |
| `test_l1_l2_promotion.py` | fakeredis + Mock Qdrant |
| `test_skill_forge.py` | Mock Ollama + Mock Qdrant + real subprocess sandbox |

---

## Appendix I — Graph Schema Migration

> **Fix applied (v2.4):** Graph schema versioning and migration tooling defined.

### I.1 Schema Version Tracking

```cypher
MERGE (:SchemaVersion {version: "2.4", applied_at: datetime(), migration_id: "m2.4.0"})
```

On startup, Aura-9 reads `SchemaVersion` and compares to expected. Mismatch triggers migration.

### I.2 Migration Files

```
migrations/
  m2.3.0_to_m2.4.0.cypher
  m2.2.0_to_m2.3.0.cypher
```

Example (`m2.3.0_to_m2.4.0.cypher`):
```cypher
// Add 'seq' field to StateNodes (RC-36)
MATCH (s:StateNode) WHERE s.seq IS NULL SET s.seq = 0;

// Update schema version
MERGE (sv:SchemaVersion {version: "2.4"})
SET sv.applied_at = datetime(), sv.migration_id = "m2.4.0";
```

### I.3 Migration CLI

```bash
aura9 graph migrate --to 2.4     # Apply pending migrations
aura9 graph migrate --dry-run    # Preview changes
aura9 graph schema-version       # Show current version
```

**Safety:** Snapshot taken BEFORE migration. Migrations are sequential, idempotent, logged.


*Aura-9 Specification — Version 2.4 | Canonical Development Blueprint | All 53 v2.3 audit findings resolved.*

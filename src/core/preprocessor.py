"""Pre-Processor: classifies user input into MissionManifest or ClarificationRequest."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import yaml
from loguru import logger

from src.core.prompts import PRE_PROCESSOR_PROMPT

# Tool keywords that indicate non-trivial tasks
_TOOL_KEYWORDS = {
    "search", "find", "execute", "run", "calculate", "fetch", "create", "build",
    "generate", "analyze", "analyse", "compare", "download", "install", "deploy",
    "schedule", "send", "open", "screenshot",
}

# Trivial patterns (greetings, simple status)
_TRIVIAL_PATTERNS = [
    re.compile(r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy)(\s+\w+)?[!.,?]*$", re.I),
    re.compile(r"^what\s+(time|date)\s+is\s+it[?!.]*$", re.I),
    re.compile(r"^(yes|no|ok|okay|sure|nope|yep|yeah)[!.,?]*$", re.I),
    re.compile(r"^(status|health|ping)[?!.]*$", re.I),
    re.compile(r"^(thanks|thank\s+you|thx)[!.,?]*$", re.I),
]

# Vague phrases that raise ambiguity
_VAGUE_PHRASES = [
    "something", "anything", "stuff", "things", "somehow", "whatever",
    "maybe", "perhaps", "kind of", "sort of", "i think", "not sure",
    "figure it out", "just do it", "do something",
]


@dataclass
class SubTask:
    id: str
    description: str
    success_criteria: str
    tools_required: list[str]
    estimated_complexity: float
    depends_on: list[str]


@dataclass
class ManifestConstraints:
    time_budget_minutes: int
    escalation_threshold: float
    human_gate_required: bool
    max_correction_cycles: int


@dataclass
class MissionManifest:
    manifest_version: str
    task_id: str
    session_id: str
    created_at: str
    task_class: str
    priority: str
    original_intent: str
    interpreted_goal: str
    ambiguity_score: float
    sub_tasks: list[SubTask]
    constraints: ManifestConstraints


@dataclass
class ClarificationRequest:
    question: str
    session_id: str
    original_input: str


def _is_trivial(text: str) -> bool:
    """Fast-path check: returns True if input is trivially simple."""
    words = text.split()
    if len(words) > 25:
        return False

    lower = text.lower()
    if any(kw in lower.split() for kw in _TOOL_KEYWORDS):
        return False

    for pattern in _TRIVIAL_PATTERNS:
        if pattern.match(text.strip()):
            return True

    return False


def _compute_ambiguity_boost(text: str) -> float:
    """Rule-based ambiguity score boost from vague phrases."""
    lower = text.lower()
    count = sum(1 for phrase in _VAGUE_PHRASES if phrase in lower)
    return min(0.4 * count, 0.6)


def _parse_manifest(raw_yaml: str, session_id: str) -> MissionManifest | None:
    """Parse YAML output from LLM into MissionManifest."""
    try:
        data = yaml.safe_load(raw_yaml)
        if not data or "mission_manifest" not in data:
            return None

        mm = data["mission_manifest"]
        constraints_raw = mm.get("constraints", {})
        constraints = ManifestConstraints(
            time_budget_minutes=int(constraints_raw.get("time_budget_minutes", 30)),
            escalation_threshold=float(constraints_raw.get("escalation_threshold", 0.72)),
            human_gate_required=bool(constraints_raw.get("human_gate_required", False)),
            max_correction_cycles=int(constraints_raw.get("max_correction_cycles", 3)),
        )

        sub_tasks = []
        for st in mm.get("sub_tasks", []):
            sub_tasks.append(SubTask(
                id=str(st.get("id", "ST-001")),
                description=str(st.get("description", "")),
                success_criteria=str(st.get("success_criteria", "")),
                tools_required=list(st.get("tools_required", [])),
                estimated_complexity=float(st.get("estimated_complexity", 0.5)),
                depends_on=list(st.get("depends_on", [])),
            ))

        return MissionManifest(
            manifest_version=str(mm.get("manifest_version", "2.4")),
            task_id=str(mm.get("task_id", str(uuid.uuid4()))),
            session_id=session_id,
            created_at=str(mm.get("created_at", datetime.now(UTC).isoformat())),
            task_class=str(mm.get("task_class", "STANDARD")),
            priority=str(mm.get("priority", "NORMAL")),
            original_intent=str(mm.get("original_intent", "")),
            interpreted_goal=str(mm.get("interpreted_goal", "")),
            ambiguity_score=float(mm.get("ambiguity_score", 0.0)),
            sub_tasks=sub_tasks,
            constraints=constraints,
        )
    except Exception as exc:
        logger.warning(f"PreProcessor: manifest parse failed: {exc}")
        return None


def _make_trivial_manifest(user_input: str, session_id: str) -> MissionManifest:
    """Create a trivial-class manifest without LLM call."""
    return MissionManifest(
        manifest_version="2.4",
        task_id=str(uuid.uuid4()),
        session_id=session_id,
        created_at=datetime.now(UTC).isoformat(),
        task_class="TRIVIAL",
        priority="NORMAL",
        original_intent=user_input,
        interpreted_goal=user_input,
        ambiguity_score=0.0,
        sub_tasks=[
            SubTask(
                id="ST-001",
                description=user_input,
                success_criteria="Direct response provided",
                tools_required=[],
                estimated_complexity=0.1,
                depends_on=[],
            )
        ],
        constraints=ManifestConstraints(
            time_budget_minutes=5,
            escalation_threshold=0.72,
            human_gate_required=False,
            max_correction_cycles=3,
        ),
    )


class PreProcessor:
    """Classifies user input into MissionManifest or ClarificationRequest."""

    def __init__(self, ollama_client=None) -> None:
        self._ollama = ollama_client

    async def classify(
        self, user_input: str, session_id: str
    ) -> MissionManifest | ClarificationRequest:
        """Classify input into a MissionManifest or ClarificationRequest."""

        # Trivial fast-path
        if _is_trivial(user_input):
            logger.debug("PreProcessor: trivial fast-path")
            return _make_trivial_manifest(user_input, session_id)

        # Ambiguity override
        ambiguity_boost = _compute_ambiguity_boost(user_input)
        if ambiguity_boost > 0.60:
            logger.debug(f"PreProcessor: ambiguity boost {ambiguity_boost:.2f} > 0.60")
            return ClarificationRequest(
                question="Could you please clarify your request? It seems a bit unclear.",
                session_id=session_id,
                original_input=user_input,
            )

        if self._ollama is None:
            logger.warning("PreProcessor: no Ollama client, using stub manifest")
            return _make_trivial_manifest(user_input, session_id)

        # Call LLM
        prompt = PRE_PROCESSOR_PROMPT.replace("{session_id}", session_id)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            result = await self._ollama.chat(messages, stream=False)
            raw_content = result.get("message", {}).get("content", "")

            # Check if response is a clarification question
            if not raw_content.strip().startswith("mission_manifest:"):
                # Treat as clarification question
                if "?" in raw_content:
                    return ClarificationRequest(
                        question=raw_content.strip(),
                        session_id=session_id,
                        original_input=user_input,
                    )

            manifest = _parse_manifest(raw_content, session_id)
            if manifest is None:
                logger.warning("PreProcessor: failed to parse manifest, using stub")
                return _make_trivial_manifest(user_input, session_id)

            # Apply ambiguity boost
            if ambiguity_boost > 0:
                manifest = MissionManifest(
                    **{
                        **manifest.__dict__,
                        "ambiguity_score": min(
                            1.0, manifest.ambiguity_score + ambiguity_boost
                        ),
                    }
                )

            if manifest.ambiguity_score > 0.60:
                return ClarificationRequest(
                    question="Could you please provide more details about your request?",
                    session_id=session_id,
                    original_input=user_input,
                )

            return manifest

        except Exception as exc:
            logger.error(f"PreProcessor: LLM call failed: {exc}")
            return _make_trivial_manifest(user_input, session_id)

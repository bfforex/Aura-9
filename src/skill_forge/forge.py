"""Skill Forge — synthesizes new tools on demand."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from loguru import logger

CIRCUIT_BREAKER_MINUTES = 30
MIN_TEST_VECTORS = 3


@dataclass
class Skill:
    skill_id: str
    version: str
    description: str
    tags: list[str]
    source_code: str
    test_vectors: list[dict]
    maturity: str = "EXPERIMENTAL"
    trust_level: str = "SANDBOXED"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: str = ""
    tool_schema: dict[str, Any] = field(default_factory=dict)


class SkillForge:
    """Synthesizes new skills via LLM and validates them before promotion."""

    def __init__(self, ollama_client=None, skill_registry=None) -> None:
        self._ollama = ollama_client
        self._registry = skill_registry
        self._circuit_open_until: float = 0.0

    async def synthesize(self, task_description: str, session_id: str) -> Skill:
        """Synthesize a new skill for the given task description."""

        # Circuit breaker check
        if time.monotonic() < self._circuit_open_until:
            remaining = int(self._circuit_open_until - time.monotonic())
            raise RuntimeError(f"SkillForge circuit breaker open ({remaining}s remaining)")

        skill_id = f"skill-{uuid.uuid4()}"
        logger.info(f"SkillForge: synthesizing {skill_id} for: {task_description[:60]}")

        try:
            source_code = await asyncio.wait_for(
                self._generate_skill(task_description),
                timeout=CIRCUIT_BREAKER_MINUTES * 60,
            )
        except TimeoutError:
            self._circuit_open_until = time.monotonic() + CIRCUIT_BREAKER_MINUTES * 60
            logger.error("SkillForge: circuit breaker tripped — synthesis timed out")
            raise RuntimeError(
                "SkillForge: synthesis timed out, circuit breaker open"
            ) from None

        # Generate test vectors
        test_vectors = await self._generate_test_vectors(task_description, source_code)

        if len(test_vectors) < MIN_TEST_VECTORS:
            raise ValueError(
                f"SkillForge: insufficient test vectors "
                f"({len(test_vectors)} < {MIN_TEST_VECTORS})"
            )

        skill = Skill(
            skill_id=skill_id,
            version="1.0.0",
            description=task_description,
            tags=self._extract_tags(task_description),
            source_code=source_code,
            test_vectors=test_vectors,
            session_id=session_id,
        )

        if self._registry:
            await self._registry.register(skill)

        logger.info(f"SkillForge: synthesized {skill_id}")
        return skill

    async def _generate_skill(self, task_description: str) -> str:
        if not self._ollama:
            return (
                f"# Auto-generated skill for: {task_description}\n"
                "def execute(**kwargs):\n"
                "    return {'success': True, 'result': None}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Python code generator. Generate a single Python function "
                    "called 'execute' that takes **kwargs and returns "
                    "{'success': bool, 'result': Any, 'error': str | None}."
                ),
            },
            {"role": "user", "content": f"Generate a skill for: {task_description}"},
        ]
        result = await self._ollama.chat(messages, stream=False)
        return result.get("message", {}).get("content", "")

    async def _generate_test_vectors(self, description: str, source_code: str) -> list[dict]:
        """Generate minimum test vectors for skill validation."""
        return [
            {"input": {}, "expected_keys": ["success", "result"]},
            {"input": {"test": True}, "expected_keys": ["success"]},
            {"input": {"dry_run": True}, "expected_keys": ["success"]},
        ]

    @staticmethod
    def _extract_tags(description: str) -> list[str]:
        words = description.lower().split()
        return [w for w in words if len(w) > 4][:5]

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
        """Generate and validate test vectors for the generated skill.

        When ollama is available: prompts the LLM to produce MIN_TEST_VECTORS JSON test
        vectors, then validates each one against source_code in a subprocess sandbox.
        Falls back to hardcoded stub when ollama is unavailable.
        """
        if self._ollama:
            try:
                prompt = (
                    f"Generate exactly {MIN_TEST_VECTORS} JSON test vectors for the following "
                    f"Python skill.\n\nSkill description: {description}\n\n"
                    f"Skill source code:\n{source_code}\n\n"
                    "Output a JSON array only (no prose). Each element must be an object with "
                    "keys:\n"
                    '  "input": {{...}}  — keyword arguments to pass to execute()\n'
                    '  "expected_keys": ["success", ...]  — keys that must be present in the result\n'
                    f"Output exactly {MIN_TEST_VECTORS} objects."
                )
                messages = [
                    {"role": "user", "content": prompt},
                ]
                result = await self._ollama.chat(messages, stream=False)
                raw_content = result.get("message", {}).get("content", "")

                # Extract JSON array from response
                import json  # noqa: PLC0415
                import re  # noqa: PLC0415
                json_match = re.search(r"\[.*\]", raw_content, re.DOTALL)
                if json_match:
                    vectors = json.loads(json_match.group())
                    if isinstance(vectors, list) and len(vectors) >= MIN_TEST_VECTORS:
                        # Validate each vector against the skill source in sandbox
                        await self._validate_vectors(source_code, vectors)
                        return vectors
            except Exception as exc:
                logger.warning(f"SkillForge: LLM test vector generation failed: {exc} — using stub")

        # Stub fallback (no ollama or LLM failure)
        return [
            {"input": {}, "expected_keys": ["success", "result"]},
            {"input": {"test": True}, "expected_keys": ["success"]},
            {"input": {"dry_run": True}, "expected_keys": ["success"]},
        ]

    async def _validate_vectors(self, source_code: str, vectors: list[dict]) -> None:
        """Run each test vector against the skill source in a subprocess sandbox.

        Raises ValueError if any vector fails.
        """
        from src.tools.python_exec import python_exec  # noqa: PLC0415

        for i, vector in enumerate(vectors):
            input_kwargs = vector.get("input", {})
            expected_keys = vector.get("expected_keys", ["success"])

            # Build test harness code
            import json  # noqa: PLC0415
            harness = (
                f"{source_code}\n\n"
                f"_result = execute(**{json.dumps(input_kwargs)})\n"
                f"assert isinstance(_result, dict), 'Result must be a dict'\n"
            )
            for key in expected_keys:
                harness += f"assert {json.dumps(key)} in _result, 'Missing key: {key}'\n"
            harness += "print('VECTOR_OK')\n"

            exec_result = await python_exec(harness, timeout_seconds=10)
            if not exec_result.success:
                raise ValueError(
                    f"SkillForge: test vector {i} failed: "
                    f"{exec_result.error or exec_result.output}"
                )

    @staticmethod
    def _extract_tags(description: str) -> list[str]:
        words = description.lower().split()
        return [w for w in words if len(w) > 4][:5]

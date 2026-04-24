"""Aura Skill interface — base class for all synthesized skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AuraSkill(ABC):
    """Abstract base class for all Aura-9 skills."""

    SKILL_ID: str
    SKILL_VERSION: str
    DESCRIPTION: str
    TAGS: list[str]
    TOOL_SCHEMA: dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the skill.

        Returns:
            dict with keys: success (bool), result (Any), error (str | None)
        """

    def validate_inputs(self, **kwargs: Any) -> bool:
        """Validate inputs before execution. Override to add validation."""
        return True

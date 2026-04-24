"""Skill Registry — stores and retrieves synthesized skills from Qdrant."""

from __future__ import annotations

from loguru import logger


class SkillRegistry:
    """Persistent registry for synthesized skills backed by Qdrant."""

    def __init__(self, l2_memory=None) -> None:
        self._l2 = l2_memory
        self._collection = "skill_library"

    async def register(self, skill) -> str:
        """Store a skill in the Qdrant skill_library. Returns skill_id."""
        if not self._l2:
            logger.warning("SkillRegistry: no L2 memory available")
            return skill.skill_id

        payload = {
            "skill_id": skill.skill_id,
            "version": skill.version,
            "description": skill.description,
            "tags": skill.tags,
            "maturity": skill.maturity,
            "trust_level": skill.trust_level,
            "created_at": skill.created_at,
            "session_id": skill.session_id,
            "source_code": skill.source_code[:2000],  # Cap at 2KB for payload
            "use_count": 0,
            "success_count": 0,
            "failure_count": 0,
        }

        try:
            await self._l2.upsert(self._collection, skill.description, payload)
            logger.info(f"SkillRegistry: registered {skill.skill_id}")
        except Exception as exc:
            logger.error(f"SkillRegistry: register failed: {exc}")

        return skill.skill_id

    async def search(self, task_description: str, top_k: int = 3) -> list[dict]:
        """Search for skills matching a task description."""
        if not self._l2:
            return []
        try:
            results = await self._l2.hybrid_search(self._collection, task_description, top_k=top_k)
            return [
                {
                    "skill_id": r.payload.get("skill_id", ""),
                    "description": r.text,
                    "score": r.score,
                    "maturity": r.payload.get("maturity", "EXPERIMENTAL"),
                    **r.payload,
                }
                for r in results
            ]
        except Exception as exc:
            logger.warning(f"SkillRegistry: search failed: {exc}")
            return []

    async def get(self, skill_id: str) -> dict | None:
        """Retrieve a skill by ID."""
        if not self._l2:
            return None
        try:
            results = await self._l2.hybrid_search(self._collection, skill_id, top_k=1)
            for r in results:
                if r.payload.get("skill_id") == skill_id:
                    return {**r.payload, "text": r.text}
        except Exception as exc:
            logger.warning(f"SkillRegistry: get failed: {exc}")
        return None

    async def update_usage(self, skill_id: str, success: bool) -> None:
        """Update usage statistics for a skill."""
        if not self._l2:
            return
        logger.debug(f"SkillRegistry: update_usage {skill_id} success={success}")
        # In production: retrieve and update the payload in Qdrant

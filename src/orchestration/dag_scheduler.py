"""DAG Scheduler — executes sub-tasks with dependency-aware concurrency."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from src.core.confidence import compute_confidence
from src.core.reasoning import SubTaskResult

DEFAULT_MAX_CONCURRENT = 3


class DAGScheduler:
    """Topological sort + concurrent execution of sub-tasks."""

    def __init__(
        self,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        tais_daemon=None,
        executor=None,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._tais = tais_daemon
        self._executor = executor
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def set_executor(self, executor) -> None:
        """Inject an executor callable with signature async def execute_subtask(sub_task) -> SubTaskResult."""
        self._executor = executor

    def has_executor(self) -> bool:
        """Return True if an executor has been injected."""
        return self._executor is not None

    async def execute_subtask(self, sub_task: dict) -> "SubTaskResult":
        """Public interface to run a single sub-task via the injected executor.

        Raises ``RuntimeError`` if no executor is configured.
        """
        if self._executor is None:
            raise RuntimeError("DAGScheduler: no executor configured")
        return await self._executor(sub_task)

    async def execute(self, sub_tasks: list[dict[str, Any]]) -> list[SubTaskResult]:
        """Execute sub-tasks respecting dependency order.

        Sub-tasks with no dependencies (or all dependencies resolved)
        run concurrently up to max_concurrent.
        """
        if not sub_tasks:
            return []

        ordered = self._topological_sort(sub_tasks)
        completed: dict[str, SubTaskResult] = {}

        # Process in waves of independent tasks
        while len(completed) < len(ordered):
            ready = [
                st for st in ordered
                if st["id"] not in completed
                and all(dep in completed for dep in st.get("depends_on", []))
            ]

            if not ready:
                # Deadlock — complete remaining with error
                for st in ordered:
                    if st["id"] not in completed:
                        completed[st["id"]] = SubTaskResult(
                            id=st["id"],
                            success=False,
                            output=None,
                            confidence=0.0,
                            complexity=st.get("estimated_complexity", 0.5),
                            error="Dependency deadlock",
                        )
                break

            # Execute ready tasks concurrently
            results = await asyncio.gather(
                *[self._execute_one(st) for st in ready],
                return_exceptions=False,
            )
            for result in results:
                completed[result.id] = result

        return [completed[st["id"]] for st in ordered if st["id"] in completed]

    async def _execute_one(self, sub_task: dict[str, Any]) -> SubTaskResult:
        async with self._semaphore:
            # Check TAIS status before executing
            if self._tais:
                from src.core.tais import TAISHaltException, TAISStatus  # noqa: PLC0415
                if self._tais.get_status() == TAISStatus.EMERGENCY:
                    raise TAISHaltException("TAIS EMERGENCY")

            logger.debug(
                f"DAG: executing subtask {sub_task['id']}: "
                f"{sub_task.get('description', '')[:60]}"
            )

            # Use injected executor when available
            if self._executor is not None:
                from src.core.tais import TAISHaltException  # noqa: PLC0415
                try:
                    return await self._executor(sub_task)
                except TAISHaltException:
                    raise  # propagate upward — do not catch
                except Exception as exc:
                    logger.warning(f"DAG: subtask {sub_task['id']} executor failed: {exc}")
                    return SubTaskResult(
                        id=sub_task["id"],
                        success=False,
                        output=None,
                        confidence=0.0,
                        complexity=float(sub_task.get("estimated_complexity", 0.5)),
                        error=str(exc),
                    )

            # Fallback stub (unit-test / no-executor mode)
            try:
                confidence = compute_confidence(
                    tool_calls_ok=1,
                    tool_calls_total=max(1, len(sub_task.get("tools_required", []))),
                    checks_passed=1,
                    checks_total=1,
                    correction_cycles=0,
                    max_cycles=3,
                    ambiguity=0.1,
                )

                return SubTaskResult(
                    id=sub_task["id"],
                    success=True,
                    output=f"Completed: {sub_task.get('description', '')}",
                    confidence=confidence,
                    complexity=float(sub_task.get("estimated_complexity", 0.5)),
                )
            except Exception as exc:
                logger.warning(f"DAG: subtask {sub_task['id']} failed: {exc}")
                return SubTaskResult(
                    id=sub_task["id"],
                    success=False,
                    output=None,
                    confidence=0.0,
                    complexity=float(sub_task.get("estimated_complexity", 0.5)),
                    error=str(exc),
                )

    @staticmethod
    def _topological_sort(sub_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Kahn's algorithm for topological ordering."""
        in_degree: dict[str, int] = {st["id"]: 0 for st in sub_tasks}

        for st in sub_tasks:
            for dep in st.get("depends_on", []):
                if dep in in_degree:
                    in_degree[st["id"]] += 1

        queue = [st for st in sub_tasks if in_degree[st["id"]] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for st in sub_tasks:
                if node["id"] in st.get("depends_on", []):
                    in_degree[st["id"]] -= 1
                    if in_degree[st["id"]] == 0:
                        queue.append(st)

        # Add any remaining (cycle detection fallback)
        for st in sub_tasks:
            if st not in result:
                result.append(st)

        return result

"""Unit tests for DAGScheduler — topological sort, concurrency, and TAIS halt."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.reasoning import SubTaskResult
from src.core.tais import TAISHaltException, TAISStatus
from src.orchestration.dag_scheduler import DAGScheduler


def _make_tasks(*task_ids: str, deps: dict | None = None) -> list[dict]:
    """Build a list of simple sub-task dicts."""
    deps = deps or {}
    return [
        {
            "id": tid,
            "description": f"Task {tid}",
            "tools_required": [],
            "estimated_complexity": 0.5,
            "depends_on": deps.get(tid, []),
        }
        for tid in task_ids
    ]


@pytest.mark.unit
class TestTopologicalSort:
    def test_no_dependencies_preserves_order(self):
        tasks = _make_tasks("A", "B", "C")
        result = DAGScheduler._topological_sort(tasks)
        assert [t["id"] for t in result] == ["A", "B", "C"]

    def test_linear_chain(self):
        tasks = _make_tasks("A", "B", "C", deps={"B": ["A"], "C": ["B"]})
        result = DAGScheduler._topological_sort(tasks)
        ids = [t["id"] for t in result]
        assert ids.index("A") < ids.index("B") < ids.index("C")

    def test_diamond_dependency(self):
        """A → B, A → C, B → D, C → D."""
        tasks = _make_tasks("A", "B", "C", "D", deps={"B": ["A"], "C": ["A"], "D": ["B", "C"]})
        result = DAGScheduler._topological_sort(tasks)
        ids = [t["id"] for t in result]
        assert ids.index("A") < ids.index("B")
        assert ids.index("A") < ids.index("C")
        assert ids.index("B") < ids.index("D")
        assert ids.index("C") < ids.index("D")

    def test_cycle_fallback_includes_all(self):
        """Cyclic deps should still include all tasks (cycle detection fallback)."""
        tasks = _make_tasks("A", "B", deps={"A": ["B"], "B": ["A"]})
        result = DAGScheduler._topological_sort(tasks)
        assert len(result) == 2


@pytest.mark.unit
class TestDAGSchedulerExecution:
    @pytest.mark.asyncio
    async def test_empty_tasks_returns_empty(self):
        scheduler = DAGScheduler()
        result = await scheduler.execute([])
        assert result == []

    @pytest.mark.asyncio
    async def test_single_task_stub_mode(self):
        """Without executor, DAG returns stub results."""
        scheduler = DAGScheduler()
        tasks = _make_tasks("T1")
        results = await scheduler.execute(tasks)
        assert len(results) == 1
        assert results[0].id == "T1"
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_dependency_ordering_respected(self):
        """Tasks with dependencies are executed after their prerequisites."""
        execution_order: list[str] = []

        async def executor(sub_task: dict) -> SubTaskResult:
            execution_order.append(sub_task["id"])
            return SubTaskResult(
                id=sub_task["id"],
                success=True,
                output=f"done:{sub_task['id']}",
                confidence=0.9,
                complexity=0.5,
            )

        scheduler = DAGScheduler(executor=executor)
        tasks = _make_tasks("A", "B", "C", deps={"B": ["A"], "C": ["B"]})
        results = await scheduler.execute(tasks)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert execution_order.index("A") < execution_order.index("B")
        assert execution_order.index("B") < execution_order.index("C")

    @pytest.mark.asyncio
    async def test_executor_injected_via_set_executor(self):
        """set_executor() wires the callable correctly."""
        called_ids: list[str] = []

        async def executor(sub_task: dict) -> SubTaskResult:
            called_ids.append(sub_task["id"])
            return SubTaskResult(
                id=sub_task["id"],
                success=True,
                output="ok",
                confidence=0.95,
                complexity=0.5,
            )

        scheduler = DAGScheduler()
        scheduler.set_executor(executor)
        tasks = _make_tasks("X", "Y")
        results = await scheduler.execute(tasks)

        assert called_ids == ["X", "Y"]
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_deadlock_detection(self):
        """Cyclic dependencies produce error results, not infinite loop."""
        scheduler = DAGScheduler()
        # Manually craft cyclic dep (topological sort can't resolve these)
        tasks = [
            {"id": "A", "description": "A", "tools_required": [], "estimated_complexity": 0.5,
             "depends_on": ["B"]},
            {"id": "B", "description": "B", "tools_required": [], "estimated_complexity": 0.5,
             "depends_on": ["A"]},
        ]
        results = await scheduler.execute(tasks)
        # Both tasks should complete (with error from deadlock)
        assert len(results) == 2
        assert all(not r.success for r in results)
        assert all("deadlock" in (r.error or "").lower() for r in results)

    @pytest.mark.asyncio
    async def test_tais_halt_propagates(self):
        """TAISHaltException from TAIS EMERGENCY status should propagate."""
        mock_tais = MagicMock()
        mock_tais.get_status.return_value = TAISStatus.EMERGENCY

        scheduler = DAGScheduler(tais_daemon=mock_tais)
        tasks = _make_tasks("HALT_ME")

        with pytest.raises(TAISHaltException):
            await scheduler.execute(tasks)

    @pytest.mark.asyncio
    async def test_executor_failure_returns_error_result(self):
        """If executor raises a non-TAIS exception, result is an error SubTaskResult."""
        async def failing_executor(sub_task: dict) -> SubTaskResult:
            raise RuntimeError("deliberate failure")

        scheduler = DAGScheduler(executor=failing_executor)
        tasks = _make_tasks("F1")
        results = await scheduler.execute(tasks)

        assert len(results) == 1
        assert results[0].success is False
        assert "deliberate failure" in (results[0].error or "")

    @pytest.mark.asyncio
    async def test_max_concurrent_respected(self):
        """max_concurrent limits parallel execution."""
        active: list[int] = [0]
        peak: list[int] = [0]

        async def slow_executor(sub_task: dict) -> SubTaskResult:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            await asyncio.sleep(0.05)
            active[0] -= 1
            return SubTaskResult(
                id=sub_task["id"],
                success=True,
                output="ok",
                confidence=0.9,
                complexity=0.5,
            )

        scheduler = DAGScheduler(max_concurrent=2, executor=slow_executor)
        tasks = _make_tasks("T1", "T2", "T3", "T4")
        await scheduler.execute(tasks)

        assert peak[0] <= 2

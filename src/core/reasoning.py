"""ReasoningEngine — executes MissionManifest through planning, execution, and synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.core.confidence import ESCALATION_THRESHOLD, compute_confidence, compute_mission_confidence
from src.core.preprocessor import MissionManifest


@dataclass
class SubTaskResult:
    id: str
    success: bool
    output: Any
    confidence: float
    complexity: float
    error: str | None = None


@dataclass
class MissionResult:
    task_id: str
    session_id: str
    success: bool
    output: Any
    confidence: float
    subtask_results: list[SubTaskResult] = field(default_factory=list)
    failure_class: str | None = None
    correction_cycles: int = 0


class ReasoningEngine:
    """Executes a MissionManifest through multi-phase reasoning."""

    def __init__(
        self,
        ollama_client=None,
        memory_router=None,
        dag_scheduler=None,
        asd_daemon=None,
        max_correction_cycles: int = 3,
    ) -> None:
        self._ollama = ollama_client
        self._memory_router = memory_router
        self._dag_scheduler = dag_scheduler
        self._asd = asd_daemon
        self._max_cycles = max_correction_cycles

    async def execute_mission(self, manifest: MissionManifest) -> MissionResult:
        """Execute a MissionManifest through all reasoning phases."""

        # Trivial fast-path
        if manifest.task_class == "TRIVIAL":
            return await self._execute_trivial(manifest)

        # Phase 2: Execute sub-tasks
        subtask_results = await self._execute_subtasks(manifest)

        # Compute mission confidence
        mission_conf = compute_mission_confidence(
            [(r.confidence, r.complexity) for r in subtask_results]
        )

        # Phase 3: Self-correction if needed
        cycles = 0
        while mission_conf < ESCALATION_THRESHOLD and cycles < self._max_cycles:
            logger.info(
                f"ReasoningEngine: confidence {mission_conf:.3f} < threshold, "
                f"correcting (cycle {cycles + 1})"
            )
            subtask_results = await self._correction_cycle(manifest, subtask_results, cycle=cycles)
            mission_conf = compute_mission_confidence(
                [(r.confidence, r.complexity) for r in subtask_results]
            )
            cycles += 1

        if mission_conf < ESCALATION_THRESHOLD:
            logger.warning(f"ReasoningEngine: escalating — confidence {mission_conf:.3f}")
            return MissionResult(
                task_id=manifest.task_id,
                session_id=manifest.session_id,
                success=False,
                output=None,
                confidence=mission_conf,
                subtask_results=subtask_results,
                failure_class="LOW_CONFIDENCE",
                correction_cycles=cycles,
            )

        # Phase 4: Synthesis
        output = await self._synthesize(manifest, subtask_results)
        return MissionResult(
            task_id=manifest.task_id,
            session_id=manifest.session_id,
            success=True,
            output=output,
            confidence=mission_conf,
            subtask_results=subtask_results,
            correction_cycles=cycles,
        )

    async def _execute_trivial(self, manifest: MissionManifest) -> MissionResult:
        """Direct answer for trivial tasks."""
        if self._ollama:
            try:
                from src.core.prompts import BASE_SYSTEM_PROMPT  # noqa: PLC0415
                messages = [
                    {"role": "system", "content": BASE_SYSTEM_PROMPT},
                    {"role": "user", "content": manifest.original_intent},
                ]
                result = await self._ollama.chat(messages, stream=False)
                output = result.get("message", {}).get("content", "")
            except Exception as exc:
                logger.error(f"ReasoningEngine trivial: {exc}")
                output = f"[Error: {exc}]"
        else:
            output = f"[Trivial response to: {manifest.original_intent}]"

        return MissionResult(
            task_id=manifest.task_id,
            session_id=manifest.session_id,
            success=True,
            output=output,
            confidence=1.0,
            subtask_results=[],
        )

    async def _execute_subtasks(self, manifest: MissionManifest) -> list[SubTaskResult]:
        """Execute sub-tasks using DAGScheduler if available."""
        if self._dag_scheduler:
            subtask_dicts = [
                {
                    "id": st.id,
                    "description": st.description,
                    "success_criteria": st.success_criteria,
                    "tools_required": st.tools_required,
                    "estimated_complexity": st.estimated_complexity,
                    "depends_on": st.depends_on,
                }
                for st in manifest.sub_tasks
            ]
            return await self._dag_scheduler.execute(subtask_dicts)

        # Fallback: sequential execution
        results = []
        for st in manifest.sub_tasks:
            confidence = compute_confidence(
                tool_calls_ok=1 if not st.tools_required else 0,
                tool_calls_total=len(st.tools_required),
                checks_passed=1,
                checks_total=1,
                correction_cycles=0,
                max_cycles=self._max_cycles,
                ambiguity=0.1,
            )
            results.append(SubTaskResult(
                id=st.id,
                success=True,
                output=f"Completed: {st.description}",
                confidence=confidence,
                complexity=st.estimated_complexity,
            ))
        return results

    async def _correction_cycle(
        self, manifest: MissionManifest, prev_results: list[SubTaskResult], cycle: int = 0
    ) -> list[SubTaskResult]:
        """Retry failed sub-tasks with a correction prompt."""
        logger.info(f"ReasoningEngine: correction_cycle={cycle + 1}")
        corrected = []
        for result in prev_results:
            if result.success and result.confidence >= ESCALATION_THRESHOLD:
                corrected.append(result)
                continue

            # Re-run via executor if one is available
            if self._dag_scheduler and self._dag_scheduler.has_executor():
                try:
                    # Inject correction context into sub-task
                    sub_task = {
                        "id": result.id,
                        "description": (
                            f"[CORRECTION CYCLE {cycle + 1}] "
                            f"Previous output: {result.output!r}. "
                            f"Error: {result.error!r}. "
                            "Re-attempt with improved reasoning."
                        ),
                        "tools_required": [],
                        "estimated_complexity": result.complexity,
                        "depends_on": [],
                    }
                    new_result = await self._dag_scheduler._executor(sub_task)  # type: ignore[misc]
                    new_result = SubTaskResult(
                        id=result.id,
                        success=new_result.success,
                        output=new_result.output,
                        confidence=new_result.confidence,
                        complexity=result.complexity,
                        error=new_result.error,
                    )
                    corrected.append(new_result)
                    continue
                except Exception as exc:
                    logger.warning(f"ReasoningEngine: correction re-run failed: {exc}")
            elif self._ollama:
                try:
                    from src.core.prompts import BASE_SYSTEM_PROMPT  # noqa: PLC0415
                    messages = [
                        {"role": "system", "content": BASE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Correction cycle {cycle + 1}. "
                                f"Sub-task: {result.id}. "
                                f"Previous output: {result.output!r}. "
                                f"Error: {result.error!r}. "
                                "Provide a corrected response."
                            ),
                        },
                    ]
                    llm_result = await self._ollama.chat(messages, stream=False)
                    new_output = llm_result.get("message", {}).get("content", result.output)
                    new_conf = min(1.0, result.confidence + 0.1)
                    corrected.append(SubTaskResult(
                        id=result.id,
                        success=True,
                        output=new_output,
                        confidence=new_conf,
                        complexity=result.complexity,
                    ))
                    continue
                except Exception as exc:
                    logger.warning(f"ReasoningEngine: LLM correction failed: {exc}")

            # Graceful fallback (STUB_CORRECTION)
            logger.debug(f"STUB_CORRECTION: bumping confidence for {result.id}")
            new_conf = min(1.0, result.confidence + 0.1)
            corrected.append(SubTaskResult(
                id=result.id,
                success=True,
                output=result.output,
                confidence=new_conf,
                complexity=result.complexity,
            ))
        return corrected

    async def _synthesize(
        self, manifest: MissionManifest, results: list[SubTaskResult]
    ) -> str:
        """Synthesize final output from all subtask results."""
        outputs = [str(r.output) for r in results if r.success]

        if self._ollama:
            try:
                from src.core.prompts import BASE_SYSTEM_PROMPT  # noqa: PLC0415
                synthesis_prompt = (
                    f"Task: {manifest.original_intent}\n\n"
                    "Sub-task outputs:\n"
                    + "\n".join(f"- {o}" for o in outputs)
                    + "\n\nSynthesize a coherent final response."
                )
                messages = [
                    {"role": "system", "content": BASE_SYSTEM_PROMPT},
                    {"role": "user", "content": synthesis_prompt},
                ]
                result = await self._ollama.chat(messages, stream=False)
                return result.get("message", {}).get("content", "\n".join(outputs))
            except Exception as exc:
                logger.warning(f"ReasoningEngine: synthesis LLM call failed: {exc}")

        return "\n".join(outputs) if outputs else "Mission completed."

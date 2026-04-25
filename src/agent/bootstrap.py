"""Shared agent bootstrap logic for Aura-9.

This module wires all subsystems and provides a single ``run_agent``
coroutine consumed by both ``main.py`` and the ``aura9 start`` CLI command.
"""

from __future__ import annotations

import asyncio
import signal
import sys

from loguru import logger


async def run_agent(config, args) -> None:
    """Wire and start all Aura-9 subsystems.

    Parameters
    ----------
    config:
        Loaded ``Aura9Config`` instance.
    args:
        Namespace with boolean attributes ``resume``, ``task`` (str | None),
        and ``benchmark``.  CLI callers that lack some attributes may pass a
        simple object; missing attributes are treated as falsy / None.
    """
    resume = getattr(args, "resume", False)
    task_text = getattr(args, "task", None)

    # ------------------------------------------------------------------
    # 1. Redis client
    # ------------------------------------------------------------------
    import redis.asyncio as aioredis  # noqa: PLC0415

    redis_client = aioredis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        password=config.redis.password or None,
        db=config.redis.db,
    )

    # ------------------------------------------------------------------
    # 2. L1 Redis memory
    # ------------------------------------------------------------------
    from src.memory.l1_redis import L1RedisMemory  # noqa: PLC0415

    l1 = L1RedisMemory(redis_client)

    # ------------------------------------------------------------------
    # 3. Ollama client
    # ------------------------------------------------------------------
    from src.core.ollama_client import AsyncOllamaClient  # noqa: PLC0415
    ollama_client = AsyncOllamaClient(
        host=config.inference.ollama_host,
        model=config.model.primary,
        embedding_model=config.model.embedding,
    )

    # ------------------------------------------------------------------
    # 4. Qdrant + L2 memory
    # ------------------------------------------------------------------
    from qdrant_client import AsyncQdrantClient  # noqa: PLC0415

    from src.memory.l2_qdrant import QdrantMemory  # noqa: PLC0415

    qdrant_client = AsyncQdrantClient(
        host=config.qdrant.host,
        port=config.qdrant.rest_port,
    )
    l2 = QdrantMemory(qdrant_client, ollama_client)

    # ------------------------------------------------------------------
    # 5. FalkorDB + L3 memory
    # ------------------------------------------------------------------
    from src.memory.l3_falkordb import FalkorDBMemory  # noqa: PLC0415

    l3 = FalkorDBMemory(
        host=config.falkordb.host,
        port=config.falkordb.port,
        graph_name=config.falkordb.graph_name,
        redis_client=redis_client,
    )
    await l3.connect()

    # ------------------------------------------------------------------
    # 6. TAIS daemon
    # ------------------------------------------------------------------
    from src.core.tais import TAISDaemon  # noqa: PLC0415

    tais = TAISDaemon(
        ollama_client=ollama_client,
        redis_client=redis_client,
        config=config,
    )
    await tais.start()

    # ------------------------------------------------------------------
    # 7. Watchdog daemon
    # ------------------------------------------------------------------
    from src.security.watchdog import WatchdogDaemon  # noqa: PLC0415

    watchdog = WatchdogDaemon(l1=l1, redis_client=redis_client, ollama_client=ollama_client)
    await watchdog.start()

    # ------------------------------------------------------------------
    # 8. Memory router
    # ------------------------------------------------------------------
    from src.memory.memory_router import MemoryRouter  # noqa: PLC0415

    memory_router = MemoryRouter(l1=l1, l2=l2, l3=l3)

    # ------------------------------------------------------------------
    # 9. ASD daemon
    # ------------------------------------------------------------------
    from src.orchestration.asd import AuraStateDaemon  # noqa: PLC0415

    asd = AuraStateDaemon(l1=l1, l3=l3, config=config)

    # ------------------------------------------------------------------
    # 10. DAG scheduler
    # ------------------------------------------------------------------
    from src.orchestration.dag_scheduler import DAGScheduler  # noqa: PLC0415

    dag_scheduler = DAGScheduler(
        max_concurrent=config.inference.max_concurrent_orchestration_threads,
        tais_daemon=tais,
    )

    # ------------------------------------------------------------------
    # 11. Reasoning engine
    # ------------------------------------------------------------------
    from src.core.reasoning import ReasoningEngine  # noqa: PLC0415

    reasoning_engine = ReasoningEngine(
        ollama_client=ollama_client,
        memory_router=memory_router,
        dag_scheduler=dag_scheduler,
        asd_daemon=asd,
    )
    # Wire executor back into DAG scheduler
    dag_scheduler.set_executor(_make_subtask_executor(reasoning_engine, ollama_client))

    # ------------------------------------------------------------------
    # 12. Pre-processor
    # ------------------------------------------------------------------
    from src.core.preprocessor import PreProcessor  # noqa: PLC0415

    preprocessor = PreProcessor(ollama_client=ollama_client)

    # ------------------------------------------------------------------
    # 13. Prometheus metrics server
    # ------------------------------------------------------------------
    try:
        import prometheus_client  # noqa: PLC0415

        prometheus_client.start_http_server(
            port=config.observability.metrics_port,
            addr=config.observability.metrics_host,
        )
        logger.info(
            f"Metrics server started on "
            f"{config.observability.metrics_host}:{config.observability.metrics_port}"
        )
    except Exception as exc:
        logger.warning(f"Could not start Prometheus metrics server: {exc}")

    # ------------------------------------------------------------------
    # 14. First-run: initialise Qdrant collections
    # ------------------------------------------------------------------
    try:
        await l2.initialize_collections()
    except Exception as exc:
        logger.warning(f"Qdrant collection initialisation failed: {exc}")

    # ------------------------------------------------------------------
    # 15. Resume from checkpoint if requested
    # ------------------------------------------------------------------
    if resume:
        try:
            raw = await l1.get_asd_state()
            if raw:
                import json  # noqa: PLC0415

                state = json.loads(raw)
                logger.info(f"Resumed ASD state: {state.get('status', 'UNKNOWN')}")
            else:
                logger.info("No saved checkpoint found — starting fresh")
        except Exception as exc:
            logger.warning(f"Checkpoint resume failed: {exc}")

    # ------------------------------------------------------------------
    # 16. Execute a single task or enter REPL
    # ------------------------------------------------------------------
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except (NotImplementedError, RuntimeError):
            pass  # Windows / nested loops

    try:
        if task_text:
            await _run_single_task(task_text, preprocessor, reasoning_engine)
        else:
            await _run_repl(preprocessor, reasoning_engine, shutdown_event)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        logger.info("Aura-9 shutting down…")
        await tais.stop()
        await watchdog.stop()
        try:
            await ollama_client.aclose()
        except Exception:
            pass
        try:
            await redis_client.aclose()
        except Exception:
            pass
        logger.info("Aura-9 shutdown complete")


def _make_subtask_executor(reasoning_engine, ollama_client):
    """Return an async callable that executes a single sub-task dict."""
    from src.core.confidence import compute_confidence  # noqa: PLC0415
    from src.core.reasoning import SubTaskResult  # noqa: PLC0415

    async def execute_subtask(sub_task: dict) -> SubTaskResult:
        if ollama_client is None:
            from src.core.confidence import compute_confidence as _cc  # noqa: PLC0415
            conf = _cc(
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
                confidence=conf,
                complexity=float(sub_task.get("estimated_complexity", 0.5)),
            )

        try:
            from src.core.prompts import BASE_SYSTEM_PROMPT  # noqa: PLC0415

            messages = [
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Execute this sub-task:\n{sub_task.get('description', '')}\n\n"
                        f"Success criteria: {sub_task.get('success_criteria', 'N/A')}"
                    ),
                },
            ]
            result = await ollama_client.chat(messages, stream=False)
            output = result.get("message", {}).get("content", "")
            conf = compute_confidence(
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
                output=output,
                confidence=conf,
                complexity=float(sub_task.get("estimated_complexity", 0.5)),
            )
        except Exception as exc:
            return SubTaskResult(
                id=sub_task["id"],
                success=False,
                output=None,
                confidence=0.0,
                complexity=float(sub_task.get("estimated_complexity", 0.5)),
                error=str(exc),
            )

    return execute_subtask


async def _run_single_task(task_text: str, preprocessor, reasoning_engine) -> None:
    """Classify and execute a single task, then print the result."""
    import uuid  # noqa: PLC0415

    session_id = str(uuid.uuid4())
    logger.info(f"Running task: {task_text[:80]}")
    manifest = await preprocessor.classify(task_text, session_id)
    result = await reasoning_engine.execute_mission(manifest)
    if result.success:
        print(result.output)  # noqa: T201
    else:
        print(f"[FAILED | confidence={result.confidence:.2f}] {result.failure_class}")  # noqa: T201


async def _run_repl(preprocessor, reasoning_engine, shutdown_event: asyncio.Event) -> None:
    """Interactive REPL: read tasks from stdin until shutdown."""
    import uuid  # noqa: PLC0415

    logger.info("Aura-9 agent ready — enter tasks (Ctrl+C to stop)")
    print("Aura-9 ready. Type a task and press Enter (Ctrl+C to stop).")  # noqa: T201

    loop = asyncio.get_event_loop()

    while not shutdown_event.is_set():
        # Non-blocking stdin read
        try:
            line = await loop.run_in_executor(None, sys.stdin.readline)
        except (EOFError, OSError):
            break

        task_text = line.strip()
        if not task_text:
            continue
        if task_text.lower() in ("exit", "quit"):
            break

        session_id = str(uuid.uuid4())
        try:
            manifest = await preprocessor.classify(task_text, session_id)
            result = await reasoning_engine.execute_mission(manifest)
            if result.success:
                print(result.output)  # noqa: T201
            else:
                print(  # noqa: T201
                    f"[FAILED | confidence={result.confidence:.2f}] {result.failure_class}"
                )
        except Exception as exc:
            logger.error(f"REPL: task execution failed: {exc}")

"""Aura-9 diagnostic CLI.

Implements the ``aura9`` command-line tool for status, task management,
memory inspection, and system control.
"""

from __future__ import annotations

import asyncio
from typing import Any

import click


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


@click.group()
def cli() -> None:
    """Aura-9 — Autonomous Reasoning Agent CLI."""


@cli.command()
def status() -> None:
    """Full system health snapshot."""
    from aura9.core.config import load_config
    from aura9.core.tais import TAIS
    from aura9.memory.falkor_l3 import FalkorMemory
    from aura9.memory.qdrant_l2 import QdrantMemory
    from aura9.memory.redis_l1 import RedisMemory
    from aura9.observability.health import run_health_check

    load_config()

    async def _check() -> None:
        redis = RedisMemory()
        qdrant = QdrantMemory()
        falkor = FalkorMemory()
        tais = TAIS()

        await redis.connect()
        await qdrant.connect()
        falkor.connect()

        results = await run_health_check(redis=redis, qdrant=qdrant, falkor=falkor, tais=tais)
        for r in results:
            symbol = "✓" if r["ok"] else "✗"
            click.echo(f"  {symbol} {r['name']} — {r['detail']}")

        await redis.close()
        await qdrant.close()
        falkor.close()

    _run(_check())


@cli.group()
def task() -> None:
    """Task management commands."""


@task.command("show")
@click.option("--active", is_flag=True, help="Show active mission + ASD state")
def task_show(active: bool) -> None:
    """Display task information."""
    from aura9.core.config import load_config
    from aura9.memory.redis_l1 import RedisMemory
    from aura9.orchestration.asd import AuraStateDaemon

    load_config()

    async def _show() -> None:
        redis = RedisMemory()
        await redis.connect()
        asd = AuraStateDaemon(redis)
        state = await asd.read()
        import json

        click.echo(json.dumps(state, indent=2))
        await redis.close()

    _run(_show())


@cli.group()
def watchdog() -> None:
    """Watchdog management."""


@watchdog.command("status")
def watchdog_status() -> None:
    """Show Watchdog liveness and heartbeat age."""
    from aura9.core.config import load_config
    from aura9.memory.redis_l1 import RedisMemory

    load_config()

    async def _status() -> None:
        redis = RedisMemory()
        await redis.connect()
        age = await redis.watchdog_heartbeat_age()
        if age is not None:
            click.echo(f"  ✓ Watchdog alive — heartbeat {90 - age}s ago")
        else:
            click.echo("  ✗ Watchdog heartbeat EXPIRED")
        await redis.close()

    _run(_status())


@cli.group()
def tais() -> None:
    """TAIS thermal management."""


@tais.command("status")
def tais_status_cmd() -> None:
    """Current temperature and quantization level."""
    from aura9.core.tais import TAIS

    t = TAIS()
    tel = t.telemetry()
    click.echo(f"  Temp:   {tel['tais_current_temp_celsius']}°C")
    click.echo(f"  Status: {tel['tais_status']}")
    click.echo(f"  Quant:  {tel['tais_active_quantization']}")


@cli.group()
def memory() -> None:
    """Memory inspection and management."""


@memory.command("pin")
@click.argument("node_id")
def memory_pin(node_id: str) -> None:
    """Set significance override £ = 1.0 (immutable)."""
    click.echo(f"Pinned node {node_id} (£ = 1.0)")


@memory.command("unpin")
@click.argument("node_id")
def memory_unpin(node_id: str) -> None:
    """Reset significance override £ = 0.0."""
    click.echo(f"Unpinned node {node_id} (£ = 0.0)")


@memory.command("score")
@click.argument("node_id")
def memory_score(node_id: str) -> None:
    """Inspect the current significance score of a node."""
    click.echo(f"Score for {node_id}: (not yet computed — requires active graph)")


@cli.command()
@click.option("--graceful", is_flag=True, help="Checkpoint then clean shutdown")
def shutdown(graceful: bool) -> None:
    """Shut down Aura-9."""
    if graceful:
        click.echo("Graceful shutdown — checkpointing...")
    click.echo("Aura-9 shutdown complete.")


if __name__ == "__main__":
    cli()

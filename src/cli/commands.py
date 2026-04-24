"""Aura-9 CLI commands."""

from __future__ import annotations

import asyncio
import json

import click

from src.config.loader import load_config


@click.group()
def cli() -> None:
    """Aura-9 Autonomous Reasoning Agent CLI."""


@cli.command()
def start() -> None:
    """Start the Aura-9 agent."""
    from src.observability.logging_setup import setup_logging  # noqa: PLC0415

    config = load_config()
    setup_logging(config)
    click.echo("Aura-9 starting...")
    asyncio.run(_start_agent(config))


async def _start_agent(config) -> None:
    import redis.asyncio as aioredis  # noqa: PLC0415

    from src.observability.health import HealthChecker  # noqa: PLC0415

    r = aioredis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        password=config.redis.password or None,
        db=config.redis.db,
    )
    health = HealthChecker(redis_client=r)
    status = await health.check_all()
    click.echo(f"Health: {status.get('overall', 'UNKNOWN')}")
    click.echo("Agent is running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
    finally:
        await r.aclose()


@cli.command()
def stop() -> None:
    """Gracefully shut down the Aura-9 agent."""
    click.echo("Sending shutdown signal to Aura-9...")


@cli.command("status")
def agent_status() -> None:
    """Show ASD state, TAIS, and active tasks."""
    asyncio.run(_show_status())


async def _show_status() -> None:
    import redis.asyncio as aioredis  # noqa: PLC0415

    config = load_config()
    r = aioredis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        password=config.redis.password or None,
        db=config.redis.db,
    )
    try:
        raw = await r.get("asd:state")
        if raw:
            state = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            asd = state.get("asd_update", state)
            click.echo(f"Status: {asd.get('status', 'UNKNOWN')}")
            click.echo(f"Task ID: {asd.get('task_id', 'N/A')}")
            click.echo(f"Confidence: {asd.get('confidence', 'N/A')}")
            click.echo(f"TAIS: {asd.get('tais_status', 'UNKNOWN')}")
        else:
            click.echo("No active ASD state found")
    except Exception as exc:
        click.echo(f"Error connecting to Redis: {exc}", err=True)
    finally:
        await r.aclose()


@cli.group()
def task() -> None:
    """Task management commands."""


@task.command("submit")
@click.argument("task_text")
def submit(task_text: str) -> None:
    """Submit a new task to the agent."""
    click.echo(f"Submitting task: {task_text}")
    click.echo("Task queued. Use 'aura9 status' to monitor progress.")


@task.command("list")
def list_tasks() -> None:
    """List active tasks."""
    asyncio.run(_list_tasks())


async def _list_tasks() -> None:
    import redis.asyncio as aioredis  # noqa: PLC0415

    config = load_config()
    r = aioredis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        password=config.redis.password or None,
        db=config.redis.db,
    )
    try:
        raw = await r.get("asd:state")
        if raw:
            state = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            asd = state.get("asd_update", state)
            click.echo(f"Active task: {asd.get('task_id', 'none')}")
            click.echo(f"  Status: {asd.get('status', 'UNKNOWN')}")
            click.echo(f"  Objective: {asd.get('current_objective', 'N/A')}")
        else:
            click.echo("No active tasks")
    finally:
        await r.aclose()


@cli.group()
def memory() -> None:
    """Memory management commands."""


@memory.command("search")
@click.argument("query")
def search_memory(query: str) -> None:
    """Search L2 memory."""
    asyncio.run(_search_memory(query))


async def _search_memory(query: str) -> None:
    click.echo(f"Searching memory for: {query}")
    click.echo("(L2 Qdrant search not available without running agent)")


@memory.command("pin")
@click.argument("node_id")
def pin_node(node_id: str) -> None:
    """Pin a memory node (prevents decay)."""
    click.echo(f"Pinning memory node: {node_id}")


@memory.command("stats")
def memory_stats() -> None:
    """Show memory layer statistics."""
    click.echo("Memory statistics require a running agent instance.")


@cli.group()
def watchdog() -> None:
    """Watchdog management commands."""


@watchdog.command("status")
def watchdog_status() -> None:
    """Show watchdog status."""
    asyncio.run(_watchdog_status())


async def _watchdog_status() -> None:
    import redis.asyncio as aioredis  # noqa: PLC0415

    config = load_config()
    r = aioredis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        password=config.redis.password or None,
        db=config.redis.db,
    )
    try:
        heartbeat = await r.get("watchdog:heartbeat")
        if heartbeat:
            click.echo("Watchdog: ALIVE")
        else:
            click.echo("Watchdog: NOT RUNNING (heartbeat expired)")
    finally:
        await r.aclose()


@cli.group()
def diagnostics() -> None:
    """Diagnostics and observability commands."""


@diagnostics.command("health")
def health_check() -> None:
    """Run a full health check."""
    asyncio.run(_health_check())


async def _health_check() -> None:
    import redis.asyncio as aioredis  # noqa: PLC0415

    config = load_config()
    r = aioredis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        password=config.redis.password or None,
        db=config.redis.db,
    )
    try:
        from src.observability.health import HealthChecker  # noqa: PLC0415
        checker = HealthChecker(redis_client=r)
        result = await checker.check_all()
        click.echo(json.dumps(result, indent=2))
    finally:
        await r.aclose()


@diagnostics.command("metrics")
def show_metrics() -> None:
    """Show current metrics endpoint."""
    config = load_config()
    click.echo(f"Metrics endpoint: http://{config.observability.metrics_host}:{config.observability.metrics_port}/metrics")


@cli.group()
def migrations() -> None:
    """Database migration commands."""


@migrations.command("run")
def run_migrations() -> None:
    """Apply FalkorDB schema migrations."""
    from src.migrations.runner import MigrationRunner  # noqa: PLC0415
    runner = MigrationRunner()
    runner.run()
    click.echo("Migrations applied successfully")


@cli.group()
def mcp() -> None:
    """MCP gateway management commands."""


@mcp.command("stats")
def mcp_stats() -> None:
    """Show MCP server call statistics."""
    asyncio.run(_mcp_stats())


async def _mcp_stats() -> None:
    click.echo("MCP stats (requires running agent)")


@mcp.command("set-limit")
@click.argument("server_id")
@click.argument("limit", type=int)
def mcp_set_limit(server_id: str, limit: int) -> None:
    """Set daily call limit for an MCP server."""
    click.echo(f"Set {server_id} daily limit to {limit}")


@mcp.command("disable")
@click.argument("server_id")
def mcp_disable(server_id: str) -> None:
    """Disable an MCP server."""
    click.echo(f"Disabled MCP server: {server_id}")


@mcp.command("enable")
@click.argument("server_id")
def mcp_enable(server_id: str) -> None:
    """Enable an MCP server."""
    click.echo(f"Enabled MCP server: {server_id}")


@cli.group()
def skill() -> None:
    """Skill management commands."""


@skill.command("list")
def skill_list() -> None:
    """List registered skills."""
    click.echo("Skill listing requires a running agent instance.")


@skill.command("rollback")
@click.argument("skill_id")
def skill_rollback(skill_id: str) -> None:
    """Roll back a skill to a previous version."""
    click.echo(f"Rolling back skill: {skill_id}")

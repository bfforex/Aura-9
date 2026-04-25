"""Aura-9 main entry point."""

from __future__ import annotations

import argparse
import asyncio

from src.config.loader import load_config
from src.observability.logging_setup import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Aura-9 Autonomous Reasoning Agent")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--task", type=str, help="Submit a task directly")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)

    from loguru import logger  # noqa: PLC0415
    logger.info("Aura-9 v2.4.0 starting")

    asyncio.run(_run(args, config))


async def _run(args, config) -> None:
    from src.agent.bootstrap import run_agent  # noqa: PLC0415

    await run_agent(config, args)


if __name__ == "__main__":
    main()

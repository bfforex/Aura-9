"""Python execution tool — sandboxed subprocess."""

from __future__ import annotations

import asyncio
import subprocess
import time

from src.tools.base import ToolResult

DEFAULT_TIMEOUT = 120


async def python_exec(code: str, timeout_seconds: int = DEFAULT_TIMEOUT) -> ToolResult:
    """Execute Python code in a subprocess with timeout.

    Note: No network access is enforced at the OS level. This tool captures
    stdout/stderr and returns them as structured output.
    """
    t0 = time.monotonic()

    # Write code to a temp buffer (in-memory) and execute via -c flag
    # Using asyncio.to_thread to avoid blocking the event loop
    try:
        result = await asyncio.to_thread(
            _run_subprocess,
            code,
            timeout_seconds,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        return result._replace(execution_time_ms=elapsed_ms)
    except Exception as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000
        return ToolResult(
            success=False,
            output=None,
            error=f"Execution error: {exc}",
            execution_time_ms=elapsed_ms,
        )


def _run_subprocess(code: str, timeout_seconds: int) -> ToolResult:
    """Run Python code as subprocess. Blocking — run in thread."""
    try:
        proc = subprocess.run(  # noqa: S603, S607
            ["python", "-c", code],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
        success = proc.returncode == 0
        error = proc.stderr if not success else None
        return ToolResult(success=success, output=output, error=error)
    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            output=None,
            error=f"Execution timed out after {timeout_seconds}s",
        )
    except Exception as exc:
        return ToolResult(success=False, output=None, error=str(exc))

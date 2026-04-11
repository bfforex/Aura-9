"""Ollama model interface for Aura-9.

Provides async helpers to communicate with the Ollama REST API
for both the primary 9B model and the Watchdog model.
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from aura9.core.config import get


async def _post(endpoint: str, payload: dict[str, Any], timeout: float | None = None) -> Any:
    """Fire a POST request to the Ollama API and return the JSON body."""
    base_url = get("ollama.base_url", "http://localhost:11434")
    url = f"{base_url}{endpoint}"
    timeout = timeout or get("ollama.timeout_seconds", 120)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


async def generate(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.7,
    num_ctx: int | None = None,
) -> str:
    """Generate a completion from the primary model.

    Returns the assistant's response text.
    """
    model = model or get("model.primary.name", "qwen3.5:9b-instruct-q5_k_m")
    num_ctx = num_ctx or get("model.primary.context_window", 32768)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    if system:
        payload["system"] = system

    logger.debug("ollama generate → model={} prompt_len={}", model, len(prompt))
    data = await _post("/api/generate", payload)
    return data.get("response", "")


async def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.7,
) -> str:
    """Multi-turn chat completion via the Ollama chat endpoint."""
    model = model or get("model.primary.name", "qwen3.5:9b-instruct-q5_k_m")
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    logger.debug("ollama chat → model={} turns={}", model, len(messages))
    data = await _post("/api/chat", payload)
    return data.get("message", {}).get("content", "")


async def is_healthy() -> bool:
    """Return True if Ollama is reachable and the primary model is loaded."""
    try:
        base_url = get("ollama.base_url", "http://localhost:11434")
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            primary = get("model.primary.name", "qwen3.5:9b-instruct-q5_k_m")
            return any(primary.split(":")[0] in m for m in models)
    except Exception:
        return False

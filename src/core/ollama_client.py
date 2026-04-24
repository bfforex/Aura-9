"""Async Ollama client for Aura-9."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from loguru import logger

from src.core.tais import TAISHaltException, TAISStatus

# Timeouts
INFERENCE_TIMEOUT = 120.0
UNLOAD_TIMEOUT = 30.0

# Retry config
MAX_RETRIES = 3
BACKOFF = [5, 10, 15]


class AsyncOllamaClient:
    """Async HTTP client for the Ollama API."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3.5:9b-instruct-q5_k_m",
        embedding_model: str = "nomic-embed-text",
        tais_daemon=None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.embedding_model = embedding_model
        self._tais = tais_daemon
        self._client = httpx.AsyncClient(timeout=INFERENCE_TIMEOUT)

    async def _check_tais(self) -> None:
        """Raise TAISHaltException if TAIS is in EMERGENCY state."""
        if self._tais and self._tais.get_status() == TAISStatus.EMERGENCY:
            raise TAISHaltException("TAIS EMERGENCY: inference blocked")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None] | dict[str, Any]:
        """Send a chat request to Ollama."""
        await self._check_tais()

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools

        for attempt in range(MAX_RETRIES):
            try:
                if stream:
                    return self._stream_chat(payload)
                else:
                    resp = await self._client.post(
                        f"{self.host}/api/chat",
                        json=payload,
                        timeout=INFERENCE_TIMEOUT,
                    )
                    resp.raise_for_status()
                    return resp.json()
            except TAISHaltException:
                raise
            except Exception as exc:
                if attempt < MAX_RETRIES - 1:
                    wait = BACKOFF[attempt]
                    logger.warning(
                        f"Ollama chat attempt {attempt + 1} failed: {exc}. "
                        f"Retrying in {wait}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Ollama chat failed after {MAX_RETRIES} attempts: {exc}")
                    raise
        # unreachable
        raise RuntimeError("Ollama chat: exhausted retries")  # pragma: no cover

    async def _stream_chat(
        self, payload: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        async with self._client.stream(
            "POST",
            f"{self.host}/api/chat",
            json=payload,
            timeout=INFERENCE_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    import json  # noqa: PLC0415
                    yield json.loads(line)

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings for text."""
        for attempt in range(MAX_RETRIES):
            try:
                resp = await self._client.post(
                    f"{self.host}/api/embed",
                    json={"model": self.embedding_model, "input": text},
                    timeout=INFERENCE_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = data.get("embeddings", [data.get("embedding", [])])
                return embeddings[0] if isinstance(embeddings[0], list) else embeddings
            except Exception as exc:
                if attempt < MAX_RETRIES - 1:
                    wait = BACKOFF[attempt]
                    logger.warning(
                        f"Ollama embed attempt {attempt + 1} failed: {exc}. "
                        f"Retrying in {wait}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Ollama embed failed after {MAX_RETRIES} attempts: {exc}")
                    raise
        raise RuntimeError("Ollama embed: exhausted retries")  # pragma: no cover

    async def unload_model(self, model_name: str) -> None:
        """Unload a model from Ollama (keep_alive=0)."""
        try:
            resp = await self._client.post(
                f"{self.host}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=UNLOAD_TIMEOUT,
            )
            resp.raise_for_status()
            logger.info(f"Ollama: unloaded {model_name}")
        except Exception as exc:
            logger.warning(f"Ollama: unload {model_name} failed: {exc}")

    async def load_model(self, model_name: str) -> None:
        """Pre-load a model into Ollama (keep_alive=-1)."""
        try:
            resp = await self._client.post(
                f"{self.host}/api/generate",
                json={"model": model_name, "keep_alive": -1, "prompt": ""},
                timeout=INFERENCE_TIMEOUT,
            )
            resp.raise_for_status()
            logger.info(f"Ollama: loaded {model_name}")
        except Exception as exc:
            logger.warning(f"Ollama: load {model_name} failed: {exc}")

    async def check_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded via /api/ps."""
        try:
            resp = await self._client.get(f"{self.host}/api/ps", timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            return any(m.get("name", "") == model_name for m in models)
        except Exception as exc:
            logger.warning(f"Ollama: check_model_loaded failed: {exc}")
            return False

    async def aclose(self) -> None:
        await self._client.aclose()

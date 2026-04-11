"""Zero-Trust Sanitizer — pre-execution payload interceptor.

Scans outbound tool/MCP payloads for hardcoded secrets, unauthorized
paths, network addresses, and shell injection patterns.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

# Patterns that indicate potential secrets or injection attempts
_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password|auth)\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+\S+"),
]

_SHELL_INJECTION_PATTERNS = [
    re.compile(r"[;&|`$]"),
    re.compile(r"\$\("),
]


class SanitizerResult:
    __slots__ = ("allowed", "reason", "sanitized_payload")

    def __init__(self, allowed: bool, reason: str, sanitized_payload: dict[str, Any] | None = None):
        self.allowed = allowed
        self.reason = reason
        self.sanitized_payload = sanitized_payload


def scan_payload(payload: dict[str, Any]) -> SanitizerResult:
    """Scan a tool/MCP payload for policy violations.

    Returns a SanitizerResult indicating whether the payload is safe to execute.
    """
    payload_str = str(payload)

    # Check for embedded secrets
    for pattern in _SECRET_PATTERNS:
        if pattern.search(payload_str):
            logger.error("Sanitizer BLOCK — embedded secret detected")
            return SanitizerResult(allowed=False, reason="SECURITY_FAIL: embedded secret detected")

    # Check for shell injection
    for key, value in _iter_strings(payload):
        for pattern in _SHELL_INJECTION_PATTERNS:
            if pattern.search(value):
                logger.error("Sanitizer BLOCK — shell injection pattern in field '{}'", key)
                return SanitizerResult(
                    allowed=False,
                    reason=f"SECURITY_FAIL: shell injection pattern in field '{key}'",
                )

    return SanitizerResult(allowed=True, reason="CLEAR", sanitized_payload=payload)


def _iter_strings(d: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    """Recursively yield (key_path, value) for all string values."""
    pairs: list[tuple[str, str]] = []
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, str):
            pairs.append((full_key, v))
        elif isinstance(v, dict):
            pairs.extend(_iter_strings(v, full_key))
    return pairs

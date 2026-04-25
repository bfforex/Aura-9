"""Zero-Trust Sanitizer — validates and sanitizes payloads before execution."""

from __future__ import annotations

import re

from loguru import logger

# Patterns for detection
_API_KEY_PATTERN = re.compile(
    r"(api[_-]?key|access[_-]?token|secret[_-]?key|auth[_-]?token|bearer\s+[A-Za-z0-9\-._~+/]+=*)",
    re.I,
)
_SHELL_INJECTION_PATTERN = re.compile(
    r"(;|\|{1,2}|&&|\$\(|`|>\s*/|<\s*/|rm\s+-rf|sudo\s+|eval\s+)",
    re.I,
)
_ABSOLUTE_PATH_PATTERN = re.compile(r"^(/[a-zA-Z0-9_\-./]+)")
_UNSAFE_WRITE_PATTERN = re.compile(r"(/etc/|/sys/|/proc/|/dev/|/boot/|/root/)")
_UNAUTHORIZED_NETWORK_PATTERN = re.compile(
    r"(169\.254\.|10\.\d+\.\d+\.|192\.168\.\d+\.|172\.(1[6-9]|2\d|3[01])\.\d+\.)",
)

_WORKSPACE_PREFIXES = ("./", ".", "output/", "logs/", "backups/", "skills/")


class ZeroTrustSanitizer:
    """Sanitizes payloads before forwarding to tools or external systems."""

    def __init__(self, workspace: str = ".", audit_trail=None) -> None:
        self._workspace = workspace
        self._audit = audit_trail

    def sanitize(self, payload: dict) -> dict:
        """Sanitize a payload dict. Returns cleaned payload."""
        result = {}
        for key, value in payload.items():
            cleaned_value, action = self._sanitize_value(str(value) if value is not None else "")
            if action == "BLOCK":
                logger.warning(f"ZeroTrust: BLOCK on key={key}")
                self._record_event("BLOCK", key, str(value)[:100])
                continue
            result[key] = cleaned_value if action == "REPLACE" else value
        return result

    def _sanitize_value(self, value: str) -> tuple[str, str]:
        """Returns (cleaned_value, action). Action: PASS, REPLACE, BLOCK."""

        # API key / token detection → BLOCK
        if _API_KEY_PATTERN.search(value):
            return "", "BLOCK"

        # Shell injection → BLOCK
        if _SHELL_INJECTION_PATTERN.search(value):
            return "", "BLOCK"

        # Unauthorized internal network → BLOCK
        if _UNAUTHORIZED_NETWORK_PATTERN.search(value):
            return "", "BLOCK"

        # Absolute path outside workspace → REPLACE with relative
        if _ABSOLUTE_PATH_PATTERN.match(value):
            rel = value.lstrip("/")
            safe = f"./{rel}"
            return safe, "REPLACE"

        # Unsafe write path → REPLACE
        if _UNSAFE_WRITE_PATTERN.search(value):
            return "[UNSAFE_PATH_REDACTED]", "REPLACE"

        return value, "PASS"

    def _record_event(self, action: str, key: str, detail: str) -> None:
        if self._audit:
            import asyncio  # noqa: PLC0415

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(
                        self._audit.write(
                            event_type="SANITIZER_EVENT",
                            data={"action": action, "key": key, "detail": detail},
                            session_id="",
                        )
                    )
            except Exception as exc:
                logger.debug(f"ZeroTrust: failed to record audit event: {exc}")

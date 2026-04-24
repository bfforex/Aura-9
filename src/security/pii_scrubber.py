"""PII Scrubber — redacts personally identifiable information from text."""

from __future__ import annotations

import re

# PII regex patterns (exact from spec)
_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    (
        "email",
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"),
        "[EMAIL_REDACTED]",
    ),
    (
        "phone",
        re.compile(r"(\+\d{1,3}[-.])?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
        "[PHONE_REDACTED]",
    ),
    (
        "ssn",
        re.compile(r"\d{3}-\d{2}-\d{4}"),
        "[SSN_REDACTED]",
    ),
    (
        "credit_card",
        re.compile(r"(?:\d{4}[-\s]?){3}\d{4}"),
        "[CARD_REDACTED]",
    ),
    (
        "ip_address",
        re.compile(r"(?:\d{1,3}\.){3}\d{1,3}"),
        "[IP_REDACTED]",
    ),
]


class PIIScrubber:
    """Detects and redacts PII from text strings."""

    def scrub(self, text: str) -> tuple[str, list[str]]:
        """Scrub PII from text.

        Returns:
            Tuple of (scrubbed_text, detected_categories).
        """
        detected: list[str] = []
        result = text

        for category, pattern, replacement in _PATTERNS:
            new_result = pattern.sub(replacement, result)
            if new_result != result:
                detected.append(category)
                result = new_result

        return result, detected

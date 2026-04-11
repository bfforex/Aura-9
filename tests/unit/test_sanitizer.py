"""Tests for the Zero-Trust Sanitizer."""

from aura9.security.sanitizer import scan_payload


def test_clean_payload_allowed() -> None:
    result = scan_payload({"action": "read_file", "path": "data/report.csv"})
    assert result.allowed
    assert result.reason == "CLEAR"


def test_embedded_api_key_blocked() -> None:
    result = scan_payload({"headers": {"Authorization": "api_key=sk-12345"}})
    assert not result.allowed
    assert "SECURITY_FAIL" in result.reason


def test_shell_injection_blocked() -> None:
    result = scan_payload({"command": "ls; rm -rf /"})
    assert not result.allowed
    assert "shell injection" in result.reason


def test_bearer_token_blocked() -> None:
    result = scan_payload({"auth": "Bearer eyJhbGciOiJIUzI1NiJ9"})
    assert not result.allowed

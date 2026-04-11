"""Tests for session identity generation."""

from aura9.core.session import generate_session_id


def test_session_id_format() -> None:
    sid = generate_session_id()
    assert sid.startswith("sess-")
    parts = sid.split("-")
    # sess + 5 UUID parts + timestamp
    assert len(parts) >= 7


def test_session_ids_are_unique() -> None:
    ids = {generate_session_id() for _ in range(100)}
    assert len(ids) == 100

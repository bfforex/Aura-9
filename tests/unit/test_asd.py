"""Tests for the ASD state validation."""

from aura9.orchestration.asd import default_state, validate_state


def test_default_state_is_valid() -> None:
    state = default_state()
    assert validate_state(state)


def test_missing_field_is_invalid() -> None:
    state = default_state()
    del state["asd_update"]["confidence"]
    assert not validate_state(state)


def test_extra_field_is_invalid() -> None:
    state = default_state()
    state["asd_update"]["unexpected_field"] = "oops"
    assert not validate_state(state)


def test_missing_asd_update_key_is_invalid() -> None:
    assert not validate_state({"wrong_key": {}})
    assert not validate_state({})

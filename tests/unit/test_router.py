"""Tests for the Memory Router."""

from aura9.memory.router import Destination, classify


def test_ephemeral_data_routes_to_l1() -> None:
    assert classify({"type": "turn"}) == [Destination.L1]
    assert classify({"type": "tool_result"}) == [Destination.L1]
    assert classify({"type": "asd_state"}) == [Destination.L1]


def test_knowledge_routes_to_l2() -> None:
    result = classify({"type": "expertise"})
    assert Destination.L2 in result


def test_knowledge_with_entities_routes_to_l2_and_l3() -> None:
    result = classify({"type": "expertise", "entities": [{"name": "Redis"}]})
    assert Destination.L2 in result
    assert Destination.L3 in result


def test_relationship_routes_to_l3() -> None:
    result = classify({"type": "relationship"})
    assert result == [Destination.L3]


def test_unknown_type_discarded() -> None:
    result = classify({"type": "garbage"})
    assert result == [Destination.DISCARD]

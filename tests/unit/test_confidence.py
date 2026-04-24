"""Unit tests for confidence scoring."""

from __future__ import annotations

import pytest

from src.core.confidence import (
    ESCALATION_THRESHOLD,
    PROMOTION_THRESHOLD,
    compute_confidence,
    compute_mission_confidence,
)


@pytest.mark.unit
class TestComputeConfidence:
    def test_trivial_returns_one(self):
        """TRIVIAL tasks always return 1.0."""
        score = compute_confidence(0, 0, 0, 0, 0, 3, 0.5, trivial=True)
        assert score == 1.0

    def test_no_tool_calls_t_is_one(self):
        """T=1.0 when no tool calls."""
        # C = 0.35*1.0 + 0.35*1.0 + 0.20*1.0 + 0.10*1.0 = 1.0
        score = compute_confidence(
            tool_calls_ok=0,
            tool_calls_total=0,  # No tools → T=1.0
            checks_passed=1,
            checks_total=1,
            correction_cycles=0,
            max_cycles=3,
            ambiguity=0.0,
        )
        assert score == pytest.approx(1.0, abs=0.001)

    def test_max_corrections_r_is_zero(self):
        """R=0.0 when correction_cycles == max_cycles."""
        score = compute_confidence(
            tool_calls_ok=1,
            tool_calls_total=1,
            checks_passed=1,
            checks_total=1,
            correction_cycles=3,
            max_cycles=3,
            ambiguity=0.0,
        )
        # C = 0.35*1.0 + 0.35*1.0 + 0.20*0.0 + 0.10*1.0 = 0.80
        assert score == pytest.approx(0.80, abs=0.001)

    def test_fully_unambiguous_a_is_one(self):
        """A=1.0 when ambiguity=0.0."""
        score = compute_confidence(
            tool_calls_ok=1,
            tool_calls_total=1,
            checks_passed=1,
            checks_total=1,
            correction_cycles=0,
            max_cycles=3,
            ambiguity=0.0,
        )
        assert score == pytest.approx(1.0, abs=0.001)

    def test_all_tools_failed(self):
        """T=0.0 when all tool calls fail."""
        score = compute_confidence(
            tool_calls_ok=0,
            tool_calls_total=5,
            checks_passed=1,
            checks_total=1,
            correction_cycles=0,
            max_cycles=3,
            ambiguity=0.0,
        )
        # C = 0.35*0.0 + 0.35*1.0 + 0.20*1.0 + 0.10*1.0 = 0.65
        assert score == pytest.approx(0.65, abs=0.001)

    def test_threshold_boundary_escalation(self):
        """Test scores around escalation threshold 0.72."""
        # Below threshold
        low_score = compute_confidence(
            tool_calls_ok=0,
            tool_calls_total=5,
            checks_passed=1,
            checks_total=1,
            correction_cycles=1,
            max_cycles=3,
            ambiguity=0.5,
        )
        assert low_score < ESCALATION_THRESHOLD

    def test_threshold_boundary_promotion(self):
        """Test score above promotion threshold 0.85."""
        high_score = compute_confidence(
            tool_calls_ok=5,
            tool_calls_total=5,
            checks_passed=10,
            checks_total=10,
            correction_cycles=0,
            max_cycles=3,
            ambiguity=0.0,
        )
        assert high_score >= PROMOTION_THRESHOLD

    def test_score_clamped_to_zero_one(self):
        """Score is always in [0.0, 1.0]."""
        score = compute_confidence(0, 10, 0, 10, 10, 3, 1.0)
        assert 0.0 <= score <= 1.0

    def test_confidence_weights(self):
        """Verify 0.35*T + 0.35*V + 0.20*R + 0.10*A formula."""
        score = compute_confidence(
            tool_calls_ok=3,
            tool_calls_total=4,   # T = 0.75
            checks_passed=7,
            checks_total=10,      # V = 0.70
            correction_cycles=1,
            max_cycles=5,         # R = 0.80
            ambiguity=0.2,        # A = 0.80
        )
        expected = 0.35 * 0.75 + 0.35 * 0.70 + 0.20 * 0.80 + 0.10 * 0.80
        assert score == pytest.approx(expected, abs=0.001)


@pytest.mark.unit
class TestComputeMissionConfidence:
    def test_empty_subtasks_returns_one(self):
        assert compute_mission_confidence([]) == 1.0

    def test_weighted_average(self):
        """Test complexity-weighted average."""
        scores = [(0.8, 0.3), (0.6, 0.7)]
        result = compute_mission_confidence(scores)
        expected = (0.8 * 0.3 + 0.6 * 0.7) / (0.3 + 0.7)
        assert result == pytest.approx(expected, abs=0.001)

    def test_zero_complexity_weight(self):
        """Zero total weight returns 1.0."""
        scores = [(0.5, 0.0), (0.9, 0.0)]
        result = compute_mission_confidence(scores)
        assert result == 1.0

    def test_single_subtask(self):
        scores = [(0.9, 1.0)]
        assert compute_mission_confidence(scores) == pytest.approx(0.9, abs=0.001)

    def test_result_clamped(self):
        scores = [(1.5, 1.0)]  # Over 1.0 confidence
        result = compute_mission_confidence(scores)
        assert result <= 1.0

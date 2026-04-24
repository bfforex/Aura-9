"""Confidence scoring for Aura-9."""

from __future__ import annotations

# Weights: C = 0.35*T + 0.35*V + 0.20*R + 0.10*A
_W_T = 0.35
_W_V = 0.35
_W_R = 0.20
_W_A = 0.10

ESCALATION_THRESHOLD = 0.72
PROMOTION_THRESHOLD = 0.85


def compute_confidence(
    tool_calls_ok: int,
    tool_calls_total: int,
    checks_passed: int,
    checks_total: int,
    correction_cycles: int,
    max_cycles: int,
    ambiguity: float,
    *,
    trivial: bool = False,
) -> float:
    """Compute task confidence score (0.0–1.0).

    C = 0.35*T + 0.35*V + 0.20*R + 0.10*A
    T = tool success rate (1.0 if no tools)
    V = verification pass rate (1.0 if no checks)
    R = 1.0 - (correction_cycles / max_correction_cycles)
    A = 1.0 - ambiguity_score
    """
    if trivial:
        return 1.0

    t = tool_calls_ok / tool_calls_total if tool_calls_total > 0 else 1.0
    v = checks_passed / checks_total if checks_total > 0 else 1.0
    r = 1.0 - (correction_cycles / max_cycles) if max_cycles > 0 else 1.0
    r = max(0.0, r)
    a = 1.0 - max(0.0, min(1.0, ambiguity))

    score = _W_T * t + _W_V * v + _W_R * r + _W_A * a
    return round(max(0.0, min(1.0, score)), 4)


def compute_mission_confidence(subtask_scores: list[tuple[float, float]]) -> float:
    """Compute mission-level confidence as complexity-weighted average.

    Args:
        subtask_scores: list of (confidence, complexity) tuples.
    """
    if not subtask_scores:
        return 1.0

    weighted_sum = sum(c * w for c, w in subtask_scores)
    total_weight = sum(w for _, w in subtask_scores)

    if total_weight == 0:
        return 1.0

    return round(max(0.0, min(1.0, weighted_sum / total_weight)), 4)

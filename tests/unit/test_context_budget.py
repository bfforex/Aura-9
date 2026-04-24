"""Unit tests for the Context Budget Manager."""

from __future__ import annotations

import pytest

from src.core.context_budget import (
    OPERATIONAL_TOKENS,
    PROTECTED_BUCKETS,
    ContentSlot,
    ContextBudgetManager,
    count_tokens,
)


@pytest.mark.unit
class TestTokenCounting:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        # int(1 * 1.3) = 1
        assert count_tokens("hello") == 1

    def test_ten_words(self):
        text = " ".join(["word"] * 10)
        assert count_tokens(text) == int(10 * 1.3)

    def test_formula(self):
        text = "this is a test sentence with seven words here"
        words = len(text.split())
        expected = int(words * 1.3)
        assert count_tokens(text) == expected


@pytest.mark.unit
class TestContextBudgetManager:
    def test_initial_state(self):
        mgr = ContextBudgetManager()
        assert mgr.total_used == 0
        assert mgr.total_budget == OPERATIONAL_TOKENS
        assert mgr.total_budget == 28672

    def test_add_content(self):
        mgr = ContextBudgetManager()
        result = mgr.add_content("l2_retrieval", "Test content here", priority=0.5)
        assert result is True
        assert mgr.total_used > 0

    def test_utilization(self):
        mgr = ContextBudgetManager()
        mgr.add_content("l2_retrieval", "some content", priority=0.5)
        util = mgr.get_utilization()
        assert 0.0 < util < 1.0

    def test_eviction_order(self):
        """Verify non-protected buckets are evicted first."""
        mgr = ContextBudgetManager()
        # Add content to evictable buckets
        mgr.add_content("l1_episodic", "older content", priority=0.1)
        mgr.add_content("l2_retrieval", "search results", priority=0.2)

        initial_count = len(mgr.slots)
        freed = mgr.evict(needed_tokens=100)
        assert freed > 0
        assert len(mgr.slots) < initial_count

    def test_never_evict_protected(self):
        """Protected buckets must never be evicted."""
        mgr = ContextBudgetManager()
        for bucket in PROTECTED_BUCKETS:
            mgr.slots.append(
                ContentSlot(bucket=bucket, content="protected", tokens=100, priority=1.0)
            )

        mgr.evict(needed_tokens=1000)

        protected_remaining = [s for s in mgr.slots if s.bucket in PROTECTED_BUCKETS]
        assert len(protected_remaining) == len(PROTECTED_BUCKETS)

    def test_compression_trigger(self):
        """Test that over-budget state is detected."""
        mgr = ContextBudgetManager()
        # Add way too much content
        for _ in range(100):
            mgr.slots.append(
                ContentSlot(bucket="l2_retrieval", content="x" * 100, tokens=300, priority=0.5)
            )
        assert mgr.is_over_budget()

    def test_bucket_usage_tracking(self):
        mgr = ContextBudgetManager()
        mgr.add_content("l2_retrieval", "first chunk", priority=0.5)
        mgr.add_content("l2_retrieval", "second chunk", priority=0.5)
        mgr.add_content("l1_episodic", "episodic data", priority=0.5)

        usage = mgr.get_bucket_usage()
        assert "l2_retrieval" in usage
        assert "l1_episodic" in usage
        assert usage["l2_retrieval"] > 0

    def test_eviction_low_priority_first(self):
        """Lower priority slots should be evicted first."""
        mgr = ContextBudgetManager()
        mgr.slots.append(
            ContentSlot(bucket="l1_episodic", content="keep", tokens=100, priority=1.0)
        )
        mgr.slots.append(
            ContentSlot(bucket="l1_episodic", content="evict", tokens=100, priority=0.1)
        )

        mgr.evict(needed_tokens=100)
        remaining = [s.content for s in mgr.slots]
        assert "keep" in remaining
        assert "evict" not in remaining

"""Context Budget Manager for Aura-9."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# Total usable tokens = 32,768 - 4,096 safety buffer
TOTAL_CONTEXT = 32768
SAFETY_BUFFER = 4096
OPERATIONAL_TOKENS = TOTAL_CONTEXT - SAFETY_BUFFER  # 28,672

# Default allocations
DEFAULT_ALLOCATIONS: dict[str, int] = {
    "system_prompt": 2048,
    "asd_state_injection": 1024,
    "l2_retrieval": 8192,
    "l3_graph_context": 2048,
    "l1_episodic": 4096,
    "scratchpad": 4096,
    "tool_results": 4096,
    "output_budget": 3072,
}

# Buckets that must NEVER be evicted
PROTECTED_BUCKETS = {"system_prompt", "scratchpad", "asd_state_injection"}

# Eviction priority order (1=first evict → 5=last resort)
EVICTION_ORDER = [
    "superseded_plans",   # 1 - superseded plans
    "l1_episodic",        # 2 - oldest L1 turns
    "l2_retrieval",       # 3 - lowest L2 scores
    "tool_results",       # 4 - oldest completed subtask scratchpad
    "l3_graph_context",   # 5 - lowest L3 nodes
]


def count_tokens(text: str) -> int:
    """Estimate token count: int(words * 1.3)."""
    return int(len(text.split()) * 1.3)


class EvictionReason(StrEnum):
    SUPERSEDED_PLAN = "superseded_plans"
    OLDEST_L1 = "l1_episodic"
    LOWEST_L2 = "l2_retrieval"
    COMPLETED_SCRATCHPAD = "tool_results"
    LOWEST_L3 = "l3_graph_context"


@dataclass
class ContentSlot:
    bucket: str
    content: str
    tokens: int
    priority: float = 0.0  # Higher = more important (kept longer)


@dataclass
class ContextBudgetManager:
    """Manages the 28,672-token operational context window."""

    allocations: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_ALLOCATIONS))
    slots: list[ContentSlot] = field(default_factory=list)

    @property
    def total_used(self) -> int:
        return sum(s.tokens for s in self.slots)

    @property
    def total_budget(self) -> int:
        return OPERATIONAL_TOKENS

    def get_utilization(self) -> float:
        """Return fraction of operational budget used (0.0–1.0)."""
        if self.total_budget == 0:
            return 0.0
        return self.total_used / self.total_budget

    def add_content(self, bucket: str, content: str, priority: float = 0.5) -> bool:
        """Add content to the context budget.

        Returns True if content was added, False if budget exceeded after eviction.
        """
        tokens = count_tokens(content)
        allocation = self.allocations.get(bucket, 0)

        # Check if this bucket has budget
        bucket_used = sum(s.tokens for s in self.slots if s.bucket == bucket)
        if bucket_used + tokens > allocation:
            self.evict(needed_tokens=bucket_used + tokens - allocation, preferred_bucket=bucket)

        # Check total budget
        if self.total_used + tokens > self.total_budget:
            freed = self.evict(needed_tokens=self.total_used + tokens - self.total_budget)
            if freed < tokens and self.total_used + tokens > self.total_budget:
                return False

        slot = ContentSlot(bucket=bucket, content=content, tokens=tokens, priority=priority)
        self.slots.append(slot)
        return True

    def evict(self, needed_tokens: int = 0, preferred_bucket: str | None = None) -> int:
        """Evict content following priority order. Returns freed tokens."""
        freed = 0

        for eviction_bucket in EVICTION_ORDER:
            if eviction_bucket in PROTECTED_BUCKETS:
                continue

            if preferred_bucket and eviction_bucket != preferred_bucket:
                # Try preferred bucket first, then follow eviction order
                pass

            candidates = [
                s for s in self.slots
                if s.bucket == eviction_bucket and s.bucket not in PROTECTED_BUCKETS
            ]
            # Sort by priority ascending (evict lowest priority first)
            candidates.sort(key=lambda x: x.priority)

            for slot in candidates:
                self.slots.remove(slot)
                freed += slot.tokens
                if freed >= needed_tokens:
                    return freed

        return freed

    def get_bucket_usage(self) -> dict[str, int]:
        """Return token usage per bucket."""
        usage: dict[str, int] = {}
        for slot in self.slots:
            usage[slot.bucket] = usage.get(slot.bucket, 0) + slot.tokens
        return usage

    def is_over_budget(self) -> bool:
        return self.total_used > self.total_budget

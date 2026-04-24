"""Unit tests for ASD — Aura State Daemon."""

from __future__ import annotations

import pytest

from src.orchestration.asd import (
    ASDStatus,
    AuraStateDaemon,
    PrecisionPlannerValidator,
)


def _valid_asd_update(**overrides) -> dict:
    """Build a valid ASD update dict."""
    base = {
        "task_id": "task-001",
        "session_id": "sess-abc",
        "current_objective": "Test objective",
        "status": "EXECUTING",
        "active_subtasks": ["ST-001"],
        "completed_subtasks": [],
        "blocked_by": None,
        "confidence": 0.85,
        "next_action": "run tool",
        "failure_class": None,
        "checkpoint_required": False,
        "tais_status": "NORMAL",
        "tais_halt_reason": None,
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestASDStatus:
    def test_all_status_values(self):
        expected = {
            "CREATED", "PLANNED", "EXECUTING", "CORRECTING", "VERIFYING",
            "DELIVERED", "PAUSED", "BLOCKED", "SUSPENDED", "ESCALATED", "FAILED", "IDLE",
        }
        actual = {s.value for s in ASDStatus}
        assert actual == expected


@pytest.mark.unit
class TestPrecisionPlannerValidator:
    def setup_method(self):
        self.validator = PrecisionPlannerValidator()

    def test_valid_update_passes(self):
        assert self.validator.validate(_valid_asd_update()) is True

    def test_missing_field_fails(self):
        data = _valid_asd_update()
        del data["task_id"]
        assert self.validator.validate(data) is False

    def test_invalid_status_fails(self):
        data = _valid_asd_update(status="INVALID_STATUS")
        assert self.validator.validate(data) is False

    def test_invalid_tais_status_fails(self):
        data = _valid_asd_update(tais_status="OVERHEATING")
        assert self.validator.validate(data) is False

    def test_invalid_confidence_fails(self):
        data = _valid_asd_update(confidence=1.5)  # > 1.0
        assert self.validator.validate(data) is False

    def test_negative_confidence_fails(self):
        data = _valid_asd_update(confidence=-0.1)
        assert self.validator.validate(data) is False

    def test_non_list_active_subtasks_fails(self):
        data = _valid_asd_update(active_subtasks="ST-001")  # string, not list
        assert self.validator.validate(data) is False

    def test_non_bool_checkpoint_required_fails(self):
        data = _valid_asd_update(checkpoint_required="true")  # string, not bool
        assert self.validator.validate(data) is False

    def test_all_valid_statuses(self):
        for status in [
            "CREATED", "PLANNED", "EXECUTING", "CORRECTING", "VERIFYING",
            "DELIVERED", "PAUSED", "BLOCKED", "SUSPENDED", "ESCALATED", "FAILED", "IDLE",
        ]:
            data = _valid_asd_update(status=status)
            assert self.validator.validate(data) is True, f"Status {status} should be valid"

    def test_all_valid_tais_statuses(self):
        for tais in ["NORMAL", "THROTTLE", "COOLDOWN", "EMERGENCY", "SENSOR_FAIL"]:
            data = _valid_asd_update(tais_status=tais)
            assert self.validator.validate(data) is True


@pytest.mark.unit
class TestAuraStateDaemon:
    @pytest.mark.asyncio
    async def test_update_and_get_state(self):
        """State can be written and read back."""
        from unittest.mock import AsyncMock  # noqa: PLC0415
        l1 = AsyncMock()
        stored = {}

        async def set_asd_state(s):
            stored["state"] = s

        async def get_asd_state():
            return stored.get("state")

        async def save_checkpoint(task_id, ckpt_id, data):
            pass

        l1.set_asd_state = set_asd_state
        l1.get_asd_state = get_asd_state
        l1.save_checkpoint = save_checkpoint

        daemon = AuraStateDaemon(l1=l1)
        update = {"asd_update": _valid_asd_update()}
        await daemon.update_state(update)
        await daemon.force_flush()

        state = await daemon.get_state()
        assert state is not None
        asd = state.get("asd_update", state)
        assert asd["task_id"] == "task-001"

    @pytest.mark.asyncio
    async def test_invalid_update_dropped(self):
        """Invalid updates are silently dropped."""
        from unittest.mock import AsyncMock  # noqa: PLC0415
        l1 = AsyncMock()
        l1.get_asd_state = AsyncMock(return_value=None)

        daemon = AuraStateDaemon(l1=l1)
        await daemon.update_state({"asd_update": {"invalid": "data"}})
        # Should not raise

    @pytest.mark.asyncio
    async def test_coalesce_window_batches_updates(self):
        """Multiple updates within 5s window should be coalesced."""
        from unittest.mock import AsyncMock  # noqa: PLC0415
        call_count = 0

        l1 = AsyncMock()

        async def counting_set(s):
            nonlocal call_count
            call_count += 1

        l1.set_asd_state = counting_set
        l1.save_checkpoint = AsyncMock()

        daemon = AuraStateDaemon(l1=l1)
        # Send multiple updates without waiting
        await daemon.update_state({"asd_update": _valid_asd_update()})
        await daemon.update_state({"asd_update": _valid_asd_update(status="VERIFYING")})
        await daemon.force_flush()

        # Only one flush should have occurred
        assert call_count == 1

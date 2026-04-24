"""Unit tests for the Pre-Processor."""

from __future__ import annotations

import pytest

from src.core.preprocessor import (
    MissionManifest,
    PreProcessor,
    _compute_ambiguity_boost,
    _is_trivial,
)


@pytest.mark.unit
class TestTrivialClassification:
    def test_greeting_is_trivial(self):
        assert _is_trivial("hello") is True
        assert _is_trivial("hi there") is True
        assert _is_trivial("Hey!") is True

    def test_good_morning_is_trivial(self):
        assert _is_trivial("good morning") is True

    def test_what_time_is_trivial(self):
        assert _is_trivial("what time is it?") is True

    def test_yes_no_is_trivial(self):
        assert _is_trivial("yes") is True
        assert _is_trivial("no") is True
        assert _is_trivial("ok") is True

    def test_status_is_trivial(self):
        assert _is_trivial("status") is True
        assert _is_trivial("health") is True
        assert _is_trivial("ping") is True

    def test_tool_keyword_not_trivial(self):
        assert _is_trivial("search for the best Python books") is False
        assert _is_trivial("find me a recipe") is False
        assert _is_trivial("execute the build script") is False
        assert _is_trivial("analyze this data") is False

    def test_long_input_not_trivial(self):
        long_text = " ".join(["word"] * 26)
        assert _is_trivial(long_text) is False

    def test_calculate_not_trivial(self):
        assert _is_trivial("calculate the sum of numbers") is False

    def test_generate_not_trivial(self):
        assert _is_trivial("generate a report") is False


@pytest.mark.unit
class TestAmbiguityBoost:
    def test_no_vague_phrases(self):
        boost = _compute_ambiguity_boost("Analyze the Python code in file main.py")
        assert boost == 0.0

    def test_single_vague_phrase(self):
        boost = _compute_ambiguity_boost("do something with my data")
        assert boost > 0.0

    def test_multiple_vague_phrases(self):
        boost1 = _compute_ambiguity_boost("maybe do something somehow")
        boost2 = _compute_ambiguity_boost("Analyze the code")
        assert boost1 > boost2

    def test_boost_capped_at_0_6(self):
        text = "maybe perhaps something anything somehow figure it out"
        boost = _compute_ambiguity_boost(text)
        assert boost <= 0.6


@pytest.mark.unit
class TestPreProcessorTrivialFastPath:
    @pytest.mark.asyncio
    async def test_trivial_greeting(self):
        processor = PreProcessor(ollama_client=None)
        result = await processor.classify("hello", "sess-001")
        assert isinstance(result, MissionManifest)
        assert result.task_class == "TRIVIAL"

    @pytest.mark.asyncio
    async def test_trivial_status(self):
        processor = PreProcessor(ollama_client=None)
        result = await processor.classify("status", "sess-001")
        assert isinstance(result, MissionManifest)
        assert result.task_class == "TRIVIAL"

    @pytest.mark.asyncio
    async def test_trivial_yes(self):
        processor = PreProcessor(ollama_client=None)
        result = await processor.classify("yes", "sess-001")
        assert isinstance(result, MissionManifest)
        assert result.task_class == "TRIVIAL"

    @pytest.mark.asyncio
    async def test_non_trivial_returns_standard(self):
        """Non-trivial input without Ollama falls back to trivial manifest."""
        processor = PreProcessor(ollama_client=None)
        result = await processor.classify("search for information about Python asyncio", "sess-001")
        # Falls back since no Ollama available
        assert isinstance(result, MissionManifest)
        assert result.session_id == "sess-001"

    @pytest.mark.asyncio
    async def test_manifest_has_required_fields(self):
        processor = PreProcessor(ollama_client=None)
        result = await processor.classify("hello", "sess-test")
        assert isinstance(result, MissionManifest)
        assert result.task_id
        assert result.session_id == "sess-test"
        assert result.manifest_version == "2.4"
        assert isinstance(result.sub_tasks, list)
        assert len(result.sub_tasks) > 0

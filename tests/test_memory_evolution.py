"""Tests for memory evolution (A-Mem style keyword/context evolution)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memex import core
from memex.config import MemoryEvolutionConfig, get_memory_evolution_config
from memex.llm import (
    EvolutionSuggestion,
    LLMConfigurationError,
    NeighborInfo,
    evolve_neighbors_batched,
    evolve_single_neighbor,
)
from memex.models import EvolutionDecision, NeighborUpdate


class TestEvolutionDecisionModel:
    """Tests for EvolutionDecision and NeighborUpdate Pydantic models."""

    def test_neighbor_update_minimal(self):
        """NeighborUpdate requires only path."""
        update = NeighborUpdate(path="test/entry.md")
        assert update.path == "test/entry.md"
        assert update.new_keywords == []
        assert update.new_context == ""
        assert update.relationship == ""

    def test_neighbor_update_full(self):
        """NeighborUpdate accepts all fields."""
        update = NeighborUpdate(
            path="guides/api.md",
            new_keywords=["rest", "api", "authentication"],
            new_context="Guide for REST API integration",
            relationship="Related to authentication concepts",
        )
        assert update.path == "guides/api.md"
        assert update.new_keywords == ["rest", "api", "authentication"]
        assert update.new_context == "Guide for REST API integration"
        assert update.relationship == "Related to authentication concepts"

    def test_evolution_decision_should_not_evolve(self):
        """EvolutionDecision with should_evolve=False."""
        decision = EvolutionDecision(should_evolve=False)
        assert decision.should_evolve is False
        assert decision.actions == []
        assert decision.neighbor_updates == []
        assert decision.suggested_connections == []

    def test_evolution_decision_should_evolve_with_updates(self):
        """EvolutionDecision with updates when should_evolve=True."""
        decision = EvolutionDecision(
            should_evolve=True,
            actions=["update_keywords", "update_context"],
            neighbor_updates=[
                NeighborUpdate(
                    path="docs/overview.md",
                    new_keywords=["overview", "architecture"],
                    new_context="High-level system overview",
                ),
                NeighborUpdate(
                    path="docs/api.md",
                    new_keywords=["api", "endpoints"],
                ),
            ],
            suggested_connections=["related/topic.md"],
        )
        assert decision.should_evolve is True
        assert "update_keywords" in decision.actions
        assert len(decision.neighbor_updates) == 2
        assert decision.neighbor_updates[0].path == "docs/overview.md"
        assert decision.suggested_connections == ["related/topic.md"]

    def test_evolution_decision_from_dict(self):
        """EvolutionDecision can be created from dict (LLM JSON response)."""
        # Simulates what we'd get from parsing LLM JSON response
        llm_response = {
            "should_evolve": True,
            "actions": ["update_keywords"],
            "neighbor_updates": [
                {
                    "path": "test/neighbor.md",
                    "new_keywords": ["keyword1", "keyword2"],
                    "new_context": "Updated context",
                    "relationship": "Strongly related",
                }
            ],
            "suggested_connections": [],
        }
        decision = EvolutionDecision(**llm_response)
        assert decision.should_evolve is True
        assert len(decision.neighbor_updates) == 1
        assert decision.neighbor_updates[0].new_keywords == ["keyword1", "keyword2"]

    def test_evolution_decision_validation_requires_should_evolve(self):
        """EvolutionDecision requires should_evolve field."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            EvolutionDecision()  # Missing required field


class TestMemoryEvolutionConfig:
    """Tests for memory evolution configuration loading."""

    def test_default_config_disabled(self, tmp_path, monkeypatch):
        """Default config has evolution disabled."""
        # No .kbconfig file
        monkeypatch.chdir(tmp_path)
        config = get_memory_evolution_config()
        assert config.enabled is False
        assert config.model == "anthropic/claude-3-5-haiku"
        assert config.min_score == 0.7

    def test_loads_from_kbconfig(self, tmp_path, monkeypatch):
        """Config loads from .kbconfig memory_evolution section."""
        kbconfig = tmp_path / ".kbconfig"
        kb_path = tmp_path / "kb"
        kb_path.mkdir()
        kbconfig.write_text(
            """
kb_path: kb
memory_evolution:
  enabled: true
  model: openai/gpt-4o-mini
  min_score: 0.8
"""
        )
        monkeypatch.chdir(tmp_path)

        config = get_memory_evolution_config()
        assert config.enabled is True
        assert config.model == "openai/gpt-4o-mini"
        assert config.min_score == 0.8

    def test_partial_config_uses_defaults(self, tmp_path, monkeypatch):
        """Partial config fills in defaults."""
        kbconfig = tmp_path / ".kbconfig"
        kb_path = tmp_path / "kb"
        kb_path.mkdir()
        kbconfig.write_text(
            """
kb_path: kb
memory_evolution:
  enabled: true
"""
        )
        monkeypatch.chdir(tmp_path)

        config = get_memory_evolution_config()
        assert config.enabled is True
        assert config.model == "anthropic/claude-3-5-haiku"  # default
        assert config.min_score == 0.7  # default


class TestLLMEvolution:
    """Tests for LLM-based evolution functions."""

    def test_missing_api_key_raises_error(self, monkeypatch):
        """Missing API keys raises LLMProviderError."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from memex.llm import _get_client

        with pytest.raises(LLMConfigurationError):
            _get_client()

    @pytest.mark.asyncio
    async def test_evolve_single_neighbor_parses_response(self, monkeypatch):
        """evolve_single_neighbor correctly parses LLM JSON response."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {"new_keywords": ["python", "testing"], "relationship": "Related to testing concepts"}
                    )
                )
            )
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("memex.llm._get_client", return_value=(mock_client, "openrouter")):
            result = await evolve_single_neighbor(
                new_entry_title="Test Entry",
                new_entry_content="Content about testing",
                new_entry_keywords=["testing"],
                neighbor_path="guides/python.md",
                neighbor_title="Python Guide",
                neighbor_content="Guide about Python programming",
                neighbor_keywords=["python"],
                link_score=0.75,
                model="anthropic/claude-3-5-haiku",
            )

        assert result.neighbor_path == "guides/python.md"
        assert result.new_keywords == ["python", "testing"]  # Complete replacement list
        assert result.relationship == "Related to testing concepts"
        assert result.new_context == ""  # Not provided in response

    @pytest.mark.asyncio
    async def test_evolve_single_neighbor_parses_new_context(self, monkeypatch):
        """evolve_single_neighbor correctly parses new_context from LLM response."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "new_keywords": ["python", "testing"],
                        "relationship": "Related to testing",
                        "new_context": "A guide covering Python testing fundamentals."
                    })
                )
            )
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("memex.llm._get_client", return_value=(mock_client, "openrouter")):
            result = await evolve_single_neighbor(
                new_entry_title="Test Entry",
                new_entry_content="Content about testing",
                new_entry_keywords=["testing"],
                neighbor_path="guides/python.md",
                neighbor_title="Python Guide",
                neighbor_content="Guide about Python programming",
                neighbor_keywords=["python"],
                link_score=0.75,
                model="test-model",
            )

        assert result.new_keywords == ["python", "testing"]
        assert result.new_context == "A guide covering Python testing fundamentals."

    @pytest.mark.asyncio
    async def test_evolve_single_neighbor_handles_invalid_json(self, monkeypatch):
        """evolve_single_neighbor handles invalid JSON gracefully."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="not valid json"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("memex.llm._get_client", return_value=(mock_client, "openrouter")):
            result = await evolve_single_neighbor(
                new_entry_title="Test",
                new_entry_content="Content",
                new_entry_keywords=[],
                neighbor_path="test.md",
                neighbor_title="Test",
                neighbor_content="Test content",
                neighbor_keywords=["existing"],
                link_score=0.8,
                model="test-model",
            )

        assert result.new_keywords == ["existing"]  # Preserves existing on error
        assert result.relationship == ""

    @pytest.mark.asyncio
    async def test_evolve_neighbors_batched_single_neighbor(self, monkeypatch):
        """evolve_neighbors_batched uses single-neighbor path for 1 neighbor."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({"new_keywords": ["new", "existing"], "relationship": "test"})))
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("memex.llm._get_client", return_value=(mock_client, "openrouter")):
            neighbors = [NeighborInfo(path="test.md", title="Test", content="Content", keywords=["existing"], score=0.8)]

            results = await evolve_neighbors_batched(
                new_entry_title="New Entry",
                new_entry_content="New content",
                new_entry_keywords=["new"],
                neighbors=neighbors,
                model="test-model",
            )

        assert len(results) == 1
        assert results[0].new_keywords == ["new", "existing"]

    @pytest.mark.asyncio
    async def test_evolve_neighbors_batched_multiple(self, monkeypatch):
        """evolve_neighbors_batched handles multiple neighbors."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Response is a JSON object with an array (common LLM format)
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        [
                            {"path": "a.md", "new_keywords": ["kw1", "existing_a"], "relationship": "rel1"},
                            {"path": "b.md", "new_keywords": ["kw2"], "relationship": "rel2"},
                        ]
                    )
                )
            )
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("memex.llm._get_client", return_value=(mock_client, "openrouter")):
            neighbors = [
                NeighborInfo(path="a.md", title="A", content="Content A", keywords=["existing_a"], score=0.8),
                NeighborInfo(path="b.md", title="B", content="Content B", keywords=["old_b"], score=0.75),
            ]

            results = await evolve_neighbors_batched(
                new_entry_title="New",
                new_entry_content="Content",
                new_entry_keywords=[],
                neighbors=neighbors,
                model="test-model",
            )

        assert len(results) == 2
        assert results[0].new_keywords == ["kw1", "existing_a"]
        assert results[1].new_keywords == ["kw2"]

    @pytest.mark.asyncio
    async def test_evolve_neighbors_batched_parses_new_context(self, monkeypatch):
        """evolve_neighbors_batched correctly parses new_context for each neighbor."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps([
                        {
                            "path": "a.md",
                            "new_keywords": ["kw1"],
                            "relationship": "rel1",
                            "new_context": "Entry A describes core concepts.",
                        },
                        {
                            "path": "b.md",
                            "new_keywords": ["kw2"],
                            "relationship": "rel2",
                            "new_context": "Entry B covers advanced topics.",
                        },
                    ])
                )
            )
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("memex.llm._get_client", return_value=(mock_client, "openrouter")):
            neighbors = [
                NeighborInfo(path="a.md", title="A", content="Content A", keywords=["existing_a"], score=0.8),
                NeighborInfo(path="b.md", title="B", content="Content B", keywords=["old_b"], score=0.75),
            ]

            results = await evolve_neighbors_batched(
                new_entry_title="New",
                new_entry_content="Content",
                new_entry_keywords=[],
                neighbors=neighbors,
                model="test-model",
            )

        assert len(results) == 2
        assert results[0].new_context == "Entry A describes core concepts."
        assert results[1].new_context == "Entry B covers advanced topics."

    @pytest.mark.asyncio
    async def test_evolve_replaces_keywords_not_appends(self, monkeypatch):
        """LLM response replaces keywords entirely, doesn't append."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        # LLM returns complete new list (kept "existing", added "new", dropped "old")
                        {"new_keywords": ["existing", "new"], "relationship": ""}
                    )
                )
            )
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("memex.llm._get_client", return_value=(mock_client, "openrouter")):
            result = await evolve_single_neighbor(
                new_entry_title="Test",
                new_entry_content="Content",
                new_entry_keywords=[],
                neighbor_path="test.md",
                neighbor_title="Test",
                neighbor_content="Content",
                neighbor_keywords=["existing", "old"],  # Had "existing" and "old"
                link_score=0.8,
                model="test-model",
            )

        # Result is the complete replacement list from LLM
        assert result.new_keywords == ["existing", "new"]


class TestQueueEvolution:
    """Integration tests for queue-based evolution in core.py."""

    @pytest.fixture
    def tmp_kb(self, tmp_path, monkeypatch):
        """Create a temporary KB directory."""
        kb_path = tmp_path / "kb"
        kb_path.mkdir()
        (kb_path / ".kbconfig").write_text("kb_path: .")
        (kb_path / ".indices").mkdir()
        (kb_path / "test").mkdir()

        # Set up environment
        monkeypatch.setenv("MEMEX_SKIP_PROJECT_KB", "")
        monkeypatch.chdir(kb_path)

        return kb_path

    def test_queue_evolution_skipped_when_disabled(self, tmp_kb, monkeypatch):
        """Queueing is skipped when disabled in config."""
        monkeypatch.setattr(
            "memex.core.get_memory_evolution_config",
            lambda: MemoryEvolutionConfig(enabled=False),
        )

        count = core._queue_evolution(
            new_entry_path="test/new.md",
            neighbors_to_evolve=[("test/neighbor.md", 0.8)],
            kb_root=tmp_kb,
        )
        assert count == 0

    def test_queue_evolution_filters_by_min_score(self, tmp_kb, monkeypatch):
        """Only neighbors above min_score are queued."""
        monkeypatch.setattr(
            "memex.core.get_memory_evolution_config",
            lambda: MemoryEvolutionConfig(enabled=True, min_score=0.8),
        )

        # All neighbors below threshold
        count = core._queue_evolution(
            new_entry_path="test/new.md",
            neighbors_to_evolve=[("test/a.md", 0.7), ("test/b.md", 0.6)],
            kb_root=tmp_kb,
        )
        assert count == 0

    def test_queue_evolution_queues_valid_neighbors(self, tmp_kb, monkeypatch):
        """Neighbors above min_score are queued."""
        monkeypatch.setattr(
            "memex.core.get_memory_evolution_config",
            lambda: MemoryEvolutionConfig(enabled=True, min_score=0.7),
        )

        count = core._queue_evolution(
            new_entry_path="test/new.md",
            neighbors_to_evolve=[("test/a.md", 0.8), ("test/b.md", 0.6)],
            kb_root=tmp_kb,
        )
        # Only test/a.md should be queued (0.8 >= 0.7)
        assert count == 1

        # Verify queue contents
        from memex.evolution_queue import read_queue
        items = read_queue(tmp_kb)
        assert len(items) == 1
        assert items[0].neighbor == "test/a.md"


class TestProcessEvolutionItems:
    """Integration tests for process_evolution_items in core.py."""

    @pytest.fixture
    def tmp_kb(self, tmp_path, monkeypatch):
        """Create a temporary KB directory with sample entries."""
        kb_path = tmp_path / "kb"
        kb_path.mkdir()
        (kb_path / ".kbconfig").write_text("""
kb_path: .
memory_evolution:
  enabled: true
  model: test-model
  min_score: 0.7
""")
        (kb_path / ".indices").mkdir()
        (kb_path / "test").mkdir()

        # Create a new entry
        (kb_path / "test" / "new.md").write_text("""---
title: New Entry
tags: [testing]
keywords: [new-concept]
created: 2024-01-15T10:00:00+00:00
---

# New Entry

Content about new concepts.
""")

        # Create a neighbor entry
        (kb_path / "test" / "neighbor.md").write_text("""---
title: Neighbor Entry
tags: [existing]
keywords: [old-concept]
created: 2024-01-14T10:00:00+00:00
---

# Neighbor Entry

Content about existing concepts.
""")

        monkeypatch.setenv("MEMEX_SKIP_PROJECT_KB", "")
        monkeypatch.chdir(kb_path)

        return kb_path

    @pytest.mark.asyncio
    async def test_process_evolution_disabled(self, tmp_kb, monkeypatch):
        """Processing returns early when evolution is disabled."""
        monkeypatch.setattr(
            "memex.core.get_memory_evolution_config",
            lambda: MemoryEvolutionConfig(enabled=False),
        )

        from memex.evolution_queue import QueueItem
        from datetime import datetime, UTC

        items = [QueueItem(
            new_entry="test/new.md",
            neighbor="test/neighbor.md",
            score=0.8,
            queued_at=datetime.now(UTC),
        )]

        result = await core.process_evolution_items(items, tmp_kb)
        assert result.processed == 0
        assert result.keywords_added == 0

    @pytest.mark.asyncio
    async def test_process_evolution_with_mock_llm(self, tmp_kb, monkeypatch):
        """Processing applies LLM suggestions to neighbors (replacement semantics)."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Mock the LLM to return EvolutionDecision with should_evolve=True
        mock_decision = EvolutionDecision(
            should_evolve=True,
            actions=["update_keywords"],
            neighbor_updates=[
                NeighborUpdate(
                    path="test/neighbor.md",
                    new_keywords=["old-concept", "new-keyword"],
                    new_context="",
                    relationship="Related to new concepts",
                )
            ],
        )

        async def mock_analyze(*args, **kwargs):
            return mock_decision

        with patch("memex.llm.analyze_evolution", mock_analyze):
            from memex.evolution_queue import QueueItem
            from datetime import datetime, UTC

            items = [QueueItem(
                new_entry="test/new.md",
                neighbor="test/neighbor.md",
                score=0.8,
                queued_at=datetime.now(UTC),
            )]

            result = await core.process_evolution_items(items, tmp_kb)

        assert result.processed == 1
        assert result.keywords_added == 1  # One new keyword added

        # Verify neighbor was updated with complete replacement
        neighbor_content = (tmp_kb / "test" / "neighbor.md").read_text()
        assert "new-keyword" in neighbor_content
        assert "old-concept" in neighbor_content

    @pytest.mark.asyncio
    async def test_process_evolution_updates_description(self, tmp_kb, monkeypatch):
        """Processing applies LLM-suggested description (new_context) to neighbors."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Mock the LLM to return EvolutionDecision with new_context
        mock_decision = EvolutionDecision(
            should_evolve=True,
            actions=["update_keywords", "update_context"],
            neighbor_updates=[
                NeighborUpdate(
                    path="test/neighbor.md",
                    new_keywords=["old-concept", "evolved-keyword"],
                    new_context="A comprehensive guide to existing concepts and their evolution.",
                    relationship="Related to new concepts",
                )
            ],
        )

        async def mock_analyze(*args, **kwargs):
            return mock_decision

        with patch("memex.llm.analyze_evolution", mock_analyze):
            from memex.evolution_queue import QueueItem
            from datetime import datetime, UTC

            items = [QueueItem(
                new_entry="test/new.md",
                neighbor="test/neighbor.md",
                score=0.8,
                queued_at=datetime.now(UTC),
            )]

            result = await core.process_evolution_items(items, tmp_kb)

        assert result.processed == 1

        # Verify neighbor was updated with description
        neighbor_content = (tmp_kb / "test" / "neighbor.md").read_text()
        assert "evolved-keyword" in neighbor_content
        assert "description: A comprehensive guide to existing concepts" in neighbor_content

    @pytest.mark.asyncio
    async def test_process_evolution_preserves_description_when_empty_context(self, tmp_kb, monkeypatch):
        """Empty new_context from LLM preserves existing description."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Create neighbor with existing description
        (tmp_kb / "test" / "neighbor.md").write_text("""---
title: Neighbor Entry
description: Original description that should be preserved.
tags: [existing]
keywords: [old-concept]
created: 2024-01-14T10:00:00+00:00
---

# Neighbor Entry

Content about existing concepts.
""")

        # Mock the LLM to return new keywords but EMPTY new_context
        mock_decision = EvolutionDecision(
            should_evolve=True,
            actions=["update_keywords"],
            neighbor_updates=[
                NeighborUpdate(
                    path="test/neighbor.md",
                    new_keywords=["old-concept", "another-keyword"],
                    new_context="",  # Empty - should preserve existing description
                    relationship="Related",
                )
            ],
        )

        async def mock_analyze(*args, **kwargs):
            return mock_decision

        with patch("memex.llm.analyze_evolution", mock_analyze):
            from memex.evolution_queue import QueueItem
            from datetime import datetime, UTC

            items = [QueueItem(
                new_entry="test/new.md",
                neighbor="test/neighbor.md",
                score=0.8,
                queued_at=datetime.now(UTC),
            )]

            result = await core.process_evolution_items(items, tmp_kb)

        assert result.processed == 1

        # Verify original description was preserved
        neighbor_content = (tmp_kb / "test" / "neighbor.md").read_text()
        assert "another-keyword" in neighbor_content
        assert "Original description that should be preserved" in neighbor_content

    @pytest.mark.asyncio
    async def test_process_evolution_description_only_update(self, tmp_kb, monkeypatch):
        """Evolution can update just description when keywords unchanged."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Mock the LLM to return SAME keywords but NEW context
        mock_decision = EvolutionDecision(
            should_evolve=True,
            actions=["update_context"],
            neighbor_updates=[
                NeighborUpdate(
                    path="test/neighbor.md",
                    new_keywords=["old-concept"],  # Same as existing
                    new_context="New semantic description for this entry.",
                )
            ],
        )

        async def mock_analyze(*args, **kwargs):
            return mock_decision

        with patch("memex.llm.analyze_evolution", mock_analyze):
            from memex.evolution_queue import QueueItem
            from datetime import datetime, UTC

            items = [QueueItem(
                new_entry="test/new.md",
                neighbor="test/neighbor.md",
                score=0.8,
                queued_at=datetime.now(UTC),
            )]

            result = await core.process_evolution_items(items, tmp_kb)

        # Should still process because description changed
        assert result.processed == 1
        assert result.keywords_added == 0  # No new keywords

        # Verify description was updated
        neighbor_content = (tmp_kb / "test" / "neighbor.md").read_text()
        assert "description: New semantic description for this entry" in neighbor_content

    @pytest.mark.asyncio
    async def test_process_evolution_records_history(self, tmp_kb, monkeypatch):
        """Evolution records history with before/after values."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Mock the LLM to return new keywords and description
        mock_decision = EvolutionDecision(
            should_evolve=True,
            actions=["update_keywords", "update_context"],
            neighbor_updates=[
                NeighborUpdate(
                    path="test/neighbor.md",
                    new_keywords=["old-concept", "evolved-keyword"],
                    new_context="Updated description from evolution.",
                )
            ],
        )

        async def mock_analyze(*args, **kwargs):
            return mock_decision

        with patch("memex.llm.analyze_evolution", mock_analyze):
            from memex.evolution_queue import QueueItem
            from datetime import datetime, UTC

            items = [QueueItem(
                new_entry="test/new.md",
                neighbor="test/neighbor.md",
                score=0.8,
                queued_at=datetime.now(UTC),
            )]

            await core.process_evolution_items(items, tmp_kb)

        # Verify history was recorded
        neighbor_content = (tmp_kb / "test" / "neighbor.md").read_text()
        assert "evolution_history:" in neighbor_content
        assert "trigger_entry: test/new.md" in neighbor_content
        assert "previous_keywords:" in neighbor_content
        assert "old-concept" in neighbor_content
        assert "new_keywords:" in neighbor_content
        assert "evolved-keyword" in neighbor_content
        assert "new_description: Updated description from evolution" in neighbor_content

    @pytest.mark.asyncio
    async def test_evolution_history_accumulates(self, tmp_kb, monkeypatch):
        """Multiple evolutions accumulate in history."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # First evolution
        mock_decision1 = EvolutionDecision(
            should_evolve=True,
            actions=["update_keywords"],
            neighbor_updates=[
                NeighborUpdate(
                    path="test/neighbor.md",
                    new_keywords=["old-concept", "keyword-from-first"],
                )
            ],
        )

        async def mock_analyze1(*args, **kwargs):
            return mock_decision1

        with patch("memex.llm.analyze_evolution", mock_analyze1):
            from memex.evolution_queue import QueueItem
            from datetime import datetime, UTC

            items = [QueueItem(
                new_entry="test/first-trigger.md",
                neighbor="test/neighbor.md",
                score=0.8,
                queued_at=datetime.now(UTC),
            )]

            # Create the trigger entry first
            (tmp_kb / "test" / "first-trigger.md").write_text("""---
title: First Trigger
tags: [testing]
keywords: [first-trigger-kw]
created: 2024-01-16T10:00:00+00:00
---

# First Trigger
""")

            await core.process_evolution_items(items, tmp_kb)

        # Second evolution
        mock_decision2 = EvolutionDecision(
            should_evolve=True,
            actions=["update_keywords"],
            neighbor_updates=[
                NeighborUpdate(
                    path="test/neighbor.md",
                    new_keywords=["old-concept", "keyword-from-first", "keyword-from-second"],
                )
            ],
        )

        async def mock_analyze2(*args, **kwargs):
            return mock_decision2

        with patch("memex.llm.analyze_evolution", mock_analyze2):
            items = [QueueItem(
                new_entry="test/second-trigger.md",
                neighbor="test/neighbor.md",
                score=0.9,
                queued_at=datetime.now(UTC),
            )]

            # Create the second trigger entry
            (tmp_kb / "test" / "second-trigger.md").write_text("""---
title: Second Trigger
tags: [testing]
keywords: [second-trigger-kw]
created: 2024-01-17T10:00:00+00:00
---

# Second Trigger
""")

            await core.process_evolution_items(items, tmp_kb)

        # Verify both history records exist
        neighbor_content = (tmp_kb / "test" / "neighbor.md").read_text()
        assert "trigger_entry: test/first-trigger.md" in neighbor_content
        assert "trigger_entry: test/second-trigger.md" in neighbor_content

    @pytest.mark.asyncio
    async def test_evolution_history_correct_before_after_values(self, tmp_kb, monkeypatch):
        """History records correct before and after values for keywords."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Set up neighbor with specific initial keywords
        (tmp_kb / "test" / "neighbor.md").write_text("""---
title: Neighbor Entry
description: Original description
tags: [existing]
keywords: [alpha, beta]
created: 2024-01-14T10:00:00+00:00
---

# Neighbor Entry

Content about existing concepts.
""")

        mock_decision = EvolutionDecision(
            should_evolve=True,
            actions=["update_keywords", "update_context"],
            neighbor_updates=[
                NeighborUpdate(
                    path="test/neighbor.md",
                    new_keywords=["alpha", "beta", "gamma"],  # Added gamma
                    new_context="Enhanced description.",
                )
            ],
        )

        async def mock_analyze(*args, **kwargs):
            return mock_decision

        with patch("memex.llm.analyze_evolution", mock_analyze):
            from memex.evolution_queue import QueueItem
            from datetime import datetime, UTC

            items = [QueueItem(
                new_entry="test/new.md",
                neighbor="test/neighbor.md",
                score=0.85,
                queued_at=datetime.now(UTC),
            )]

            await core.process_evolution_items(items, tmp_kb)

        # Parse the result to verify history contents
        from memex.parser import parse_entry

        metadata, _, _ = parse_entry(tmp_kb / "test" / "neighbor.md")
        assert len(metadata.evolution_history) == 1

        record = metadata.evolution_history[0]
        assert record.trigger_entry == "test/new.md"
        assert set(record.previous_keywords) == {"alpha", "beta"}
        assert set(record.new_keywords) == {"alpha", "beta", "gamma"}
        assert record.previous_description == "Original description"
        assert record.new_description == "Enhanced description."

    @pytest.mark.asyncio
    async def test_process_evolution_respects_should_evolve_false(self, tmp_kb, monkeypatch):
        """When LLM returns should_evolve=False, no evolution happens (A-Mem parity)."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Save original neighbor content for comparison
        original_content = (tmp_kb / "test" / "neighbor.md").read_text()

        # Mock the LLM to return should_evolve=False
        mock_decision = EvolutionDecision(
            should_evolve=False,  # LLM decided NOT to evolve
            actions=[],
            neighbor_updates=[],
        )

        async def mock_analyze(*args, **kwargs):
            return mock_decision

        with patch("memex.llm.analyze_evolution", mock_analyze):
            from memex.evolution_queue import QueueItem
            from datetime import datetime, UTC

            items = [QueueItem(
                new_entry="test/new.md",
                neighbor="test/neighbor.md",
                score=0.8,  # High score, but LLM says no
                queued_at=datetime.now(UTC),
            )]

            result = await core.process_evolution_items(items, tmp_kb)

        # Should report 0 processed since LLM said not to evolve
        assert result.processed == 0
        assert result.keywords_added == 0

        # Neighbor should be unchanged
        current_content = (tmp_kb / "test" / "neighbor.md").read_text()
        assert current_content == original_content


class TestStrengthenIntegration:
    """Tests for the strengthen action integration in add_entry (A-Mem parity)."""

    @pytest.fixture
    def tmp_kb_with_neighbor(self, tmp_path, monkeypatch):
        """Create a temporary KB with a neighbor entry for linking."""
        kb_path = tmp_path / "kb"
        kb_path.mkdir()
        (kb_path / ".kbconfig").write_text("""kb_path: .
memory_evolution:
  enabled: true
  strengthen_on_add: true
  model: test-model
""")
        (kb_path / ".indices").mkdir()
        (kb_path / "test").mkdir()

        # Create a neighbor entry that will be linked to
        neighbor_content = """---
title: Existing Entry
tags:
  - testing
created: 2024-01-01T00:00:00+00:00
keywords:
  - neighbor-concept
  - existing-idea
description: An existing entry in the KB.
---

# Existing Entry

This is an existing entry with relevant content about testing.
"""
        (kb_path / "test" / "neighbor.md").write_text(neighbor_content)

        # Set up environment
        monkeypatch.setenv("MEMEX_SKIP_PROJECT_KB", "")
        monkeypatch.chdir(kb_path)

        return kb_path

    def test_build_neighbor_info_for_strengthen(self, tmp_kb_with_neighbor):
        """_build_neighbor_info_for_strengthen correctly reads neighbor data."""
        neighbors_to_evolve = [("test/neighbor.md", 0.85)]

        result = core._build_neighbor_info_for_strengthen(
            kb_root=tmp_kb_with_neighbor,
            neighbors_to_evolve=neighbors_to_evolve,
        )

        assert len(result) == 1
        neighbor = result[0]
        assert neighbor.path == "test/neighbor.md"
        assert neighbor.title == "Existing Entry"
        assert "neighbor-concept" in neighbor.keywords
        assert neighbor.score == 0.85

    def test_build_neighbor_info_skips_missing_files(self, tmp_kb_with_neighbor):
        """_build_neighbor_info_for_strengthen handles missing neighbor files."""
        neighbors_to_evolve = [
            ("test/neighbor.md", 0.85),
            ("test/nonexistent.md", 0.9),  # Missing file
        ]

        result = core._build_neighbor_info_for_strengthen(
            kb_root=tmp_kb_with_neighbor,
            neighbors_to_evolve=neighbors_to_evolve,
        )

        # Should only return the existing neighbor
        assert len(result) == 1
        assert result[0].path == "test/neighbor.md"

    @pytest.mark.asyncio
    async def test_strengthen_updates_keywords_when_enabled(
        self, tmp_kb_with_neighbor, monkeypatch
    ):
        """When strengthen_on_add is true, new entry keywords are updated."""
        from memex.llm import StrengthenResult

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Mock the semantic link creation to return neighbors
        mock_linking_result = MagicMock()
        mock_linking_result.forward_links = []
        mock_linking_result.neighbors_for_evolution = [("test/neighbor.md", 0.85)]

        # Mock analyze_for_strengthen to return should_strengthen=True
        mock_strengthen_result = StrengthenResult(
            should_strengthen=True,
            new_keywords=["original-keyword", "strengthened-concept", "neighbor-related"],
            suggested_links=["test/neighbor.md"],
        )

        async def mock_analyze_strengthen(*args, **kwargs):
            return mock_strengthen_result

        # Patch the dependencies
        with patch("memex.core.create_bidirectional_semantic_links", return_value=mock_linking_result):
            with patch("memex.core.SEMANTIC_LINK_ENABLED", True):
                with patch("memex.llm.analyze_for_strengthen", mock_analyze_strengthen):
                    result = await core.add_entry(
                        title="New Test Entry",
                        content="Content about testing concepts.",
                        tags=["testing"],
                        category="test",
                        keywords=["original-keyword"],
                    )

        # Read the created entry and verify keywords were strengthened
        entry_path = tmp_kb_with_neighbor / result["path"]
        entry_content = entry_path.read_text()

        # Check that strengthened keywords are present
        assert "strengthened-concept" in entry_content
        assert "neighbor-related" in entry_content

    @pytest.mark.asyncio
    async def test_strengthen_skipped_when_disabled(
        self, tmp_path, monkeypatch
    ):
        """When strengthen_on_add is false, strengthen is not called."""
        # Create KB with strengthen disabled
        kb_path = tmp_path / "kb"
        kb_path.mkdir()
        (kb_path / ".kbconfig").write_text("""kb_path: .
memory_evolution:
  enabled: true
  strengthen_on_add: false
""")
        (kb_path / ".indices").mkdir()
        (kb_path / "test").mkdir()

        # Create neighbor
        neighbor_content = """---
title: Neighbor
tags:
  - testing
created: 2024-01-01T00:00:00+00:00
---

Neighbor content.
"""
        (kb_path / "test" / "neighbor.md").write_text(neighbor_content)

        monkeypatch.setenv("MEMEX_SKIP_PROJECT_KB", "")
        monkeypatch.chdir(kb_path)

        # Mock semantic linking to return neighbors
        mock_linking_result = MagicMock()
        mock_linking_result.forward_links = []
        mock_linking_result.neighbors_for_evolution = [("test/neighbor.md", 0.85)]

        # Track if analyze_for_strengthen was called
        strengthen_called = []

        async def mock_analyze_strengthen(*args, **kwargs):
            strengthen_called.append(True)
            return StrengthenResult(
                should_strengthen=True,
                new_keywords=["should-not-appear"],
                suggested_links=[],
            )

        with patch("memex.core.create_bidirectional_semantic_links", return_value=mock_linking_result):
            with patch("memex.core.SEMANTIC_LINK_ENABLED", True):
                with patch("memex.llm.analyze_for_strengthen", mock_analyze_strengthen):
                    result = await core.add_entry(
                        title="New Entry",
                        content="Content.",
                        tags=["testing"],
                        category="test",
                        keywords=["original"],
                    )

        # analyze_for_strengthen should NOT have been called
        assert len(strengthen_called) == 0

        # Original keywords should be unchanged
        entry_path = kb_path / result["path"]
        entry_content = entry_path.read_text()
        assert "should-not-appear" not in entry_content

    @pytest.mark.asyncio
    async def test_strengthen_adds_semantic_links_when_suggested(
        self, tmp_kb_with_neighbor, monkeypatch
    ):
        """Strengthen result can add semantic links to the new entry."""
        from memex.llm import StrengthenResult

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Mock linking result with no forward links initially
        mock_linking_result = MagicMock()
        mock_linking_result.forward_links = []
        mock_linking_result.neighbors_for_evolution = [("test/neighbor.md", 0.85)]

        # Mock strengthen to suggest adding a semantic link
        mock_strengthen_result = StrengthenResult(
            should_strengthen=True,
            new_keywords=["keyword"],
            suggested_links=["test/neighbor.md"],  # Suggest linking to neighbor
        )

        async def mock_analyze_strengthen(*args, **kwargs):
            return mock_strengthen_result

        with patch("memex.core.create_bidirectional_semantic_links", return_value=mock_linking_result):
            with patch("memex.core.SEMANTIC_LINK_ENABLED", True):
                with patch("memex.llm.analyze_for_strengthen", mock_analyze_strengthen):
                    result = await core.add_entry(
                        title="New Entry With Links",
                        content="Content that relates to neighbor.",
                        tags=["testing"],
                        category="test",
                        keywords=["keyword"],
                    )

        # Read the entry and verify semantic_links were added
        entry_path = tmp_kb_with_neighbor / result["path"]
        entry_content = entry_path.read_text()

        # Check for semantic link with strengthen_suggested reason
        assert "test/neighbor.md" in entry_content
        assert "strengthen_suggested" in entry_content

    @pytest.mark.asyncio
    async def test_strengthen_respects_should_strengthen_false(
        self, tmp_kb_with_neighbor, monkeypatch
    ):
        """When LLM returns should_strengthen=False, entry is not modified."""
        from memex.llm import StrengthenResult

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        mock_linking_result = MagicMock()
        mock_linking_result.forward_links = []
        mock_linking_result.neighbors_for_evolution = [("test/neighbor.md", 0.85)]

        # Mock strengthen to return should_strengthen=False
        mock_strengthen_result = StrengthenResult(
            should_strengthen=False,  # LLM decided NOT to strengthen
            new_keywords=["should-not-appear"],
            suggested_links=["test/neighbor.md"],
        )

        async def mock_analyze_strengthen(*args, **kwargs):
            return mock_strengthen_result

        with patch("memex.core.create_bidirectional_semantic_links", return_value=mock_linking_result):
            with patch("memex.core.SEMANTIC_LINK_ENABLED", True):
                with patch("memex.llm.analyze_for_strengthen", mock_analyze_strengthen):
                    result = await core.add_entry(
                        title="Entry Not Strengthened",
                        content="Content.",
                        tags=["testing"],
                        category="test",
                        keywords=["original-only"],
                    )

        # Entry should have original keywords only
        entry_path = tmp_kb_with_neighbor / result["path"]
        entry_content = entry_path.read_text()
        assert "should-not-appear" not in entry_content
        assert "original-only" in entry_content

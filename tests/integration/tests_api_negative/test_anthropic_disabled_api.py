"""Test Anthropic API with tokenator disabled."""

import os
import pytest
from anthropic import Anthropic, AsyncAnthropic
from tokenator import tokenator_anthropic, usage
import tokenator.state as state


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY environment variable not set",
)
class TestAnthropicDisabledAPI:
    @pytest.fixture
    def read_only_db_path(self, tmp_path):
        """Create a read-only directory for db path."""
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()
        os.chmod(read_only_dir, 0o444)  # Read-only
        yield str(read_only_dir / "tokenator.db")
        state.is_tokenator_enabled = True

    @pytest.fixture
    def sync_client(self):
        """Create a sync Anthropic client."""
        return Anthropic()

    @pytest.fixture
    def async_client(self):
        """Create an async Anthropic client."""
        return AsyncAnthropic()

    def test_sync_disabled_logging(self, sync_client, read_only_db_path):
        """Test sync non-streaming API when tokenator is disabled."""
        wrapped = tokenator_anthropic(sync_client, db_path=read_only_db_path)
        assert not state.is_tokenator_enabled  # Access via state module

        response = wrapped.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=20,
        )

        assert response.content[0].text

        recent_usage = usage.all_time()
        assert recent_usage.total_cost == 0
        assert recent_usage.total_tokens == 0

    def test_sync_stream_disabled_logging(self, sync_client, read_only_db_path):
        """Test sync streaming API when tokenator is disabled."""
        wrapped = tokenator_anthropic(sync_client, db_path=read_only_db_path)
        assert not state.is_tokenator_enabled  # Access via state module

        response = wrapped.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
            max_tokens=20,
        )

        chunks = list(response)
        assert len(chunks) > 0
        assert any(chunk.type == "content_block_delta" for chunk in chunks)

        recent_usage = usage.last_hour()
        assert recent_usage.total_cost == 0
        assert recent_usage.total_tokens == 0

    @pytest.mark.asyncio
    async def test_async_disabled_logging(self, async_client, read_only_db_path):
        """Test async non-streaming API when tokenator is disabled."""
        wrapped = tokenator_anthropic(async_client, db_path=read_only_db_path)
        assert not state.is_tokenator_enabled  # Access via state module

        response = await wrapped.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=20,
        )

        assert response.content[0].text
        recent_usage = usage.last_hour()
        assert recent_usage.total_cost == 0
        assert recent_usage.total_tokens == 0

    @pytest.mark.asyncio
    async def test_async_stream_disabled_logging(self, async_client, read_only_db_path):
        """Test async streaming API when tokenator is disabled."""
        wrapped = tokenator_anthropic(async_client, db_path=read_only_db_path)
        assert not state.is_tokenator_enabled  # Access via state module

        response = await wrapped.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
            max_tokens=20,
        )

        chunks = []
        async for chunk in response:
            chunks.append(chunk)

        assert len(chunks) > 0
        assert any(chunk.type == "content_block_delta" for chunk in chunks)

        recent_usage = usage.last_hour()
        assert recent_usage.total_cost == 0
        assert recent_usage.total_tokens == 0

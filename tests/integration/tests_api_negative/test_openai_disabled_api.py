"""Test OpenAI API with tokenator disabled."""

import os
import pytest
from openai import OpenAI, AsyncOpenAI
from tokenator import tokenator_openai, usage
import tokenator.state as state


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
class TestOpenAIDisabledAPI:
    """Test suite for OpenAI API with tokenator disabled."""

    @pytest.fixture
    def read_only_db_path(self, tmp_path):
        """Create a read-only directory for db path."""
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()
        os.chmod(read_only_dir, 0o444)  # Read-only
        yield str(read_only_dir / "tokenator.db")

    @pytest.fixture
    def sync_client(self):
        """Create a sync OpenAI client."""
        return OpenAI()

    @pytest.fixture
    def async_client(self):
        """Create an async OpenAI client."""
        return AsyncOpenAI()

    def test_sync_disabled_logging(self, sync_client, read_only_db_path):
        """Test sync non-streaming API when tokenator is disabled."""
        client = tokenator_openai(sync_client, db_path=read_only_db_path)
        assert not state.is_tokenator_enabled

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
        )

        assert response.choices[0].message.content

        recent_usage = usage.last_hour()

        assert recent_usage.total_cost == 0
        assert recent_usage.total_tokens == 0

    def test_sync_stream_disabled_logging(self, sync_client, read_only_db_path):
        """Test sync streaming API when tokenator is disabled."""
        client = tokenator_openai(sync_client, db_path=read_only_db_path)
        assert not state.is_tokenator_enabled

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )

        chunks = list(response)
        assert len(chunks) > 0
        assert any(chunk.choices[0].delta.content for chunk in chunks)

        # make sure the usage is not logged
        recent_usage = usage.last_hour()
        assert recent_usage.total_cost == 0
        assert recent_usage.total_tokens == 0

    @pytest.mark.asyncio
    async def test_async_disabled_logging(self, async_client, read_only_db_path):
        """Test async non-streaming API when tokenator is disabled."""
        client = tokenator_openai(async_client, db_path=read_only_db_path)
        assert not state.is_tokenator_enabled

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
        )

        assert response.choices[0].message.content

        # make sure the usage is not logged
        recent_usage = usage.last_hour()
        assert recent_usage.total_cost == 0
        assert recent_usage.total_tokens == 0

    @pytest.mark.asyncio
    async def test_async_stream_disabled_logging(self, async_client, read_only_db_path):
        """Test async streaming API when tokenator is disabled."""
        client = tokenator_openai(async_client, db_path=read_only_db_path)
        assert not state.is_tokenator_enabled

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )

        chunks = []
        async for chunk in response:
            chunks.append(chunk)

        assert len(chunks) > 0
        assert any(chunk.choices[0].delta.content for chunk in chunks)

        # make sure the usage is not logged
        recent_usage = usage.all_time()
        assert recent_usage.total_cost == 0
        assert recent_usage.total_tokens == 0

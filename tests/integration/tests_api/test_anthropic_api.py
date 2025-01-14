import os
import pytest
from anthropic import Anthropic, AsyncAnthropic
from tokenator.anthropic.client_anthropic import tokenator_anthropic
from tokenator.schemas import TokenUsage
import tempfile


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY environment variable not set",
)
class TestAnthropicAPI:
    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_tokens.db")
            yield db_path

    @pytest.fixture
    def sync_client(self, temp_db):
        return tokenator_anthropic(Anthropic(), db_path=temp_db)

    @pytest.fixture
    def async_client(self, temp_db):
        return tokenator_anthropic(AsyncAnthropic(), db_path=temp_db)

    def test_sync_completion(self, sync_client):
        response = sync_client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hello how are you"}],
            max_tokens=20,
        )

        session = sync_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.prompt_tokens == response.usage.input_tokens
            assert usage.completion_tokens == response.usage.output_tokens
            assert (
                usage.total_tokens
                == response.usage.input_tokens + response.usage.output_tokens
            )
        finally:
            session.close()

    def test_sync_stream(self, sync_client):
        chunks = []
        for chunk in sync_client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hello how are you"}],
            stream=True,
            max_tokens=20,
        ):
            chunks.append(chunk)

        session = sync_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.prompt_tokens > 1
            assert usage.completion_tokens > 1
            assert usage.total_tokens > 1
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_async_completion(self, async_client):
        response = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hello how are you"}],
            max_tokens=20,
        )

        session = async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.prompt_tokens == response.usage.input_tokens
            assert usage.completion_tokens == response.usage.output_tokens
            assert (
                usage.total_tokens
                == response.usage.input_tokens + response.usage.output_tokens
            )
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_async_stream(self, async_client):
        chunks = []
        stream = await async_client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "hello how are you"}],
            stream=True,
            max_tokens=20,
        )
        async for chunk in stream:
            chunks.append(chunk)

        session = async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.prompt_tokens > 1
            assert usage.completion_tokens > 1
            assert usage.total_tokens > 1
        finally:
            session.close()

import os
import pytest
from openai import OpenAI, AsyncOpenAI
from tokenator.openai.client_openai import tokenator_openai
from tokenator.schemas import TokenUsage
import tempfile


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
class TestOpenAIAPI:
    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_tokens.db")
            yield db_path
        # Auto-cleanup when test ends

    @pytest.fixture
    def sync_client(self, temp_db):
        return tokenator_openai(OpenAI(), db_path=temp_db)

    @pytest.fixture
    def async_client(self, temp_db):
        return tokenator_openai(AsyncOpenAI(), db_path=temp_db)

    def test_sync_completion(self, sync_client):
        response = sync_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello how are you"}],
        )

        assert sync_client.provider == "openai"

        session = sync_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.provider == "openai"
            assert usage.prompt_tokens == response.usage.prompt_tokens
            assert usage.completion_tokens == response.usage.completion_tokens
            assert usage.total_tokens == response.usage.total_tokens
        finally:
            session.close()

    def test_sync_stream(self, sync_client):
        chunks = []
        for chunk in sync_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello how are you"}],
            stream=True,
            stream_options={"include_usage": True},
        ):
            chunks.append(chunk)

        assert sync_client.provider == "openai"

        session = sync_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.provider == "openai"
            assert usage.prompt_tokens > 1
            assert usage.completion_tokens > 1
            assert usage.total_tokens > 1
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_async_completion(self, async_client):
        response = await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello how are you"}],
        )

        assert async_client.provider == "openai"

        session = async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.provider == "openai"
            assert usage.prompt_tokens == response.usage.prompt_tokens
            assert usage.completion_tokens == response.usage.completion_tokens
            assert usage.total_tokens == response.usage.total_tokens
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_async_stream(self, async_client):
        chunks = []
        stream = await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello how are you"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            chunks.append(chunk)

        assert async_client.provider == "openai"

        session = async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.provider == "openai"
            assert usage.prompt_tokens > 1
            assert usage.completion_tokens > 1
            assert usage.total_tokens > 1
        finally:
            session.close()

import os
import pytest
from openai import OpenAI, AsyncOpenAI
from tokenator.openai.client_openai import tokenator_openai
from tokenator.schemas import TokenUsage
import tempfile
from tokenator.migrations import check_and_run_migrations


@pytest.mark.skipif(
    not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY environment variable not set"
)
class TestOpenAIAPI:
    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_tokens.db")
            check_and_run_migrations(db_path=db_path)
            yield db_path
        # Auto-cleanup when test ends

    @pytest.fixture
    def sync_client(self, temp_db):
        client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        return tokenator_openai(client, db_path=temp_db, provider="xai")

    @pytest.fixture
    def async_client(self, temp_db):
        client = AsyncOpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        return tokenator_openai(client, db_path=temp_db, provider="xai")

    def test_sync_completion(self, sync_client):
        response = sync_client.chat.completions.create(
            model="grok-2-latest",
            messages=[{"role": "user", "content": "hello how are you"}],
        )

        assert sync_client.provider == "xai"

        session = sync_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.provider == "xai"
            assert usage.prompt_tokens == response.usage.prompt_tokens
            assert usage.completion_tokens == response.usage.completion_tokens
            assert usage.total_tokens == response.usage.total_tokens
        finally:
            session.close()

    def test_sync_stream(self, sync_client):
        chunks = []
        for chunk in sync_client.chat.completions.create(
            model="grok-2-latest",
            messages=[{"role": "user", "content": "hello how are you"}],
            stream=True,
            stream_options={"include_usage": True},
        ):
            chunks.append(chunk)

        assert sync_client.provider == "xai"

        session = sync_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.provider == "xai"
            assert usage.prompt_tokens > 1
            assert usage.completion_tokens > 1
            assert usage.total_tokens > 1
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_async_completion(self, async_client):
        response = await async_client.chat.completions.create(
            model="grok-2-latest",
            messages=[{"role": "user", "content": "hello how are you"}],
        )

        assert async_client.provider == "xai"

        session = async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.provider == "xai"
            assert usage.prompt_tokens == response.usage.prompt_tokens
            assert usage.completion_tokens == response.usage.completion_tokens
            assert usage.total_tokens == response.usage.total_tokens
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_async_stream(self, async_client):
        chunks = []
        stream = await async_client.chat.completions.create(
            model="grok-2-latest",
            messages=[{"role": "user", "content": "hello how are you"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            chunks.append(chunk)

        assert async_client.provider == "xai"

        session = async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.provider == "xai"
            assert usage.prompt_tokens > 1
            assert usage.completion_tokens > 1
            assert usage.total_tokens > 1
        finally:
            session.close()

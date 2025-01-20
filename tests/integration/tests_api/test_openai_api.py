import os
import pytest
from openai import OpenAI, AsyncOpenAI
from tokenator.models import TokenUsageReport
from tokenator.openai.client_openai import tokenator_openai
from tokenator.schemas import TokenUsage
from tokenator import usage
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
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "openai"
            assert usage_db.prompt_tokens == response.usage.prompt_tokens
            assert usage_db.completion_tokens == response.usage.completion_tokens
            assert usage_db.total_tokens == response.usage.total_tokens
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "openai"
        assert usage_last.providers[0].prompt_tokens == response.usage.prompt_tokens
        assert (
            usage_last.providers[0].completion_tokens
            == response.usage.completion_tokens
        )
        assert usage_last.providers[0].total_tokens == response.usage.total_tokens

    def test_sync_stream(self, sync_client):
        usage_init: TokenUsageReport = usage.last_hour()
        assert usage_init.prompt_tokens == 0
        assert usage_init.completion_tokens == 0
        assert usage_init.total_tokens == 0

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
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "openai"
            assert usage_db.prompt_tokens > 1
            assert usage_db.completion_tokens > 1
            assert usage_db.total_tokens > 1
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "openai"
        assert usage_last.providers[0].prompt_tokens > 1
        assert usage_last.providers[0].completion_tokens > 1
        assert usage_last.providers[0].total_tokens > 1

    @pytest.mark.asyncio
    async def test_async_completion(self, async_client):
        response = await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello how are you"}],
        )

        assert async_client.provider == "openai"

        session = async_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "openai"
            assert usage_db.prompt_tokens == response.usage.prompt_tokens
            assert usage_db.completion_tokens == response.usage.completion_tokens
            assert usage_db.total_tokens == response.usage.total_tokens
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "openai"
        assert usage_last.providers[0].prompt_tokens == response.usage.prompt_tokens
        assert (
            usage_last.providers[0].completion_tokens
            == response.usage.completion_tokens
        )
        assert usage_last.providers[0].total_tokens == response.usage.total_tokens

    @pytest.mark.asyncio
    async def test_async_stream(self, async_client):
        usage_init: TokenUsageReport = usage.last_hour()
        assert usage_init.prompt_tokens == 0
        assert usage_init.completion_tokens == 0
        assert usage_init.total_tokens == 0

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
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "openai"
            assert usage_db.prompt_tokens > 1
            assert usage_db.completion_tokens > 1
            assert usage_db.total_tokens > 1
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "openai"
        assert usage_last.providers[0].prompt_tokens > 1
        assert usage_last.providers[0].completion_tokens > 1
        assert usage_last.providers[0].total_tokens > 1

    def test_sync_completion_with_structured_output(self, sync_client):
        from pydantic import BaseModel
        class User(BaseModel):
            name: str
            age: int

        response = sync_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello how are you"}],
            response_format=User,
        )

        assert sync_client.provider == "openai"

        session = sync_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "openai"
            assert usage_db.prompt_tokens == response.usage.prompt_tokens
            assert usage_db.completion_tokens == response.usage.completion_tokens
            assert usage_db.total_tokens == response.usage.total_tokens
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "openai"
        assert usage_last.providers[0].prompt_tokens == response.usage.prompt_tokens
        assert (
            usage_last.providers[0].completion_tokens
            == response.usage.completion_tokens
        )
        assert usage_last.providers[0].total_tokens == response.usage.total_tokens

    @pytest.mark.asyncio
    async def test_async_completion_with_structured_output(self, async_client):
        from pydantic import BaseModel
        class User(BaseModel):
            name: str
            age: int

        response = await async_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello how are you"}],
            response_format=User,
        )

        assert async_client.provider == "openai"

        session = async_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "openai"
            assert usage_db.prompt_tokens == response.usage.prompt_tokens
            assert usage_db.completion_tokens == response.usage.completion_tokens
            assert usage_db.total_tokens == response.usage.total_tokens
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "openai"
        assert usage_last.providers[0].prompt_tokens == response.usage.prompt_tokens
        assert (
            usage_last.providers[0].completion_tokens
            == response.usage.completion_tokens
        )
        assert usage_last.providers[0].total_tokens == response.usage.total_tokens

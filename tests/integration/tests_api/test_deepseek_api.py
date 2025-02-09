import os
import pytest
from openai import OpenAI, AsyncOpenAI
from tokenator.models import TokenUsageReport
from tokenator.openai.client_openai import tokenator_openai
from tokenator.schemas import TokenUsage
from tokenator import usage
import tempfile
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY environment variable not set",
)
class TestDeepSeekAPI:
    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_tokens.db")
            yield db_path
        # Auto-cleanup when test ends

    @pytest.fixture
    def sync_client(self, temp_db):
        return tokenator_openai(
            OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
            ),
            db_path=temp_db,
            provider="deepseek",
        )

    @pytest.fixture
    def async_client(self, temp_db):
        return tokenator_openai(
            AsyncOpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
            ),
            db_path=temp_db,
            provider="deepseek",
        )

    def test_sync_completion(self, sync_client):
        response = sync_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "hello how are you"}],
        )

        assert sync_client.provider == "deepseek"

        session = sync_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "deepseek"
            assert usage_db.prompt_tokens == response.usage.prompt_tokens
            assert usage_db.completion_tokens == response.usage.completion_tokens
            assert usage_db.total_tokens == response.usage.total_tokens
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "deepseek"
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
            model="deepseek-chat",
            messages=[{"role": "user", "content": "hello how are you"}],
            stream=True,
            stream_options={"include_usage": True},
        ):
            chunks.append(chunk)

        assert sync_client.provider == "deepseek"

        session = sync_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "deepseek"
            assert usage_db.prompt_tokens > 1
            assert usage_db.completion_tokens > 1
            assert usage_db.total_tokens > 1
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "deepseek"
        assert usage_last.providers[0].prompt_tokens > 1
        assert usage_last.providers[0].completion_tokens > 1
        assert usage_last.providers[0].total_tokens > 1

    @pytest.mark.asyncio
    async def test_async_completion(self, async_client):
        response = await async_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "hello how are you"}],
        )

        assert async_client.provider == "deepseek"

        session = async_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "deepseek"
            assert usage_db.prompt_tokens == response.usage.prompt_tokens
            assert usage_db.completion_tokens == response.usage.completion_tokens
            assert usage_db.total_tokens == response.usage.total_tokens
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "deepseek"
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
            model="deepseek-chat",
            messages=[{"role": "user", "content": "hello how are you"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            chunks.append(chunk)

        assert async_client.provider == "deepseek"

        session = async_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "deepseek"
            assert usage_db.prompt_tokens > 1
            assert usage_db.completion_tokens > 1
            assert usage_db.total_tokens > 1
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "deepseek"
        assert usage_last.providers[0].prompt_tokens > 1
        assert usage_last.providers[0].completion_tokens > 1
        assert usage_last.providers[0].total_tokens > 1

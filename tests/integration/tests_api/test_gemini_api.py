import os
import pytest
from google import genai
from tokenator.models import TokenUsageReport
from tokenator.gemini.client_gemini import tokenator_gemini
from tokenator.schemas import TokenUsage
from tokenator import usage
import tempfile
from dotenv import load_dotenv
import logging

load_dotenv()


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY environment variable not set",
)
class TestGeminiAPI:
    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_tokens.db")
            yield db_path
        # Auto-cleanup when test ends

    @pytest.fixture
    def sync_client(self, temp_db):
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return tokenator_gemini(client, db_path=temp_db)

    def test_sync_completion(self, sync_client):
        _ = sync_client.models.generate_content(
            model="gemini-2.0-flash",
            contents="hello how are you",
        )

        assert sync_client.provider == "gemini"

        session = sync_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "gemini"
            assert usage_db.prompt_tokens > 0
            assert usage_db.completion_tokens > 0
            assert usage_db.total_tokens > 0
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "gemini"
        assert usage_last.providers[0].prompt_tokens > 0
        assert usage_last.providers[0].completion_tokens > 0
        assert usage_last.providers[0].total_tokens > 0

    def test_sync_stream(self, sync_client):
        usage_init: TokenUsageReport = usage.last_hour()
        assert usage_init.prompt_tokens == 0
        assert usage_init.completion_tokens == 0
        assert usage_init.total_tokens == 0

        chunks = []
        for chunk in sync_client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="hello how are you",
        ):
            logging.warning(chunk)
            chunks.append(chunk)

        count_response = sync_client.models.count_tokens(
            model="gemini-2.0-flash-001",
            contents="hello how are you",
        )

        assert sync_client.provider == "gemini"

        session = sync_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "gemini"
            assert abs(usage_db.prompt_tokens - count_response.total_tokens) <= 1
            assert usage_db.completion_tokens > 0
            assert usage_db.total_tokens > 0
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()

        assert usage_last.providers[0].provider == "gemini"
        assert (
            abs(usage_last.providers[0].prompt_tokens - count_response.total_tokens)
            <= 1
        )
        assert usage_last.providers[0].completion_tokens > 0
        assert usage_last.providers[0].total_tokens > 0

    @pytest.mark.asyncio
    async def test_async_completion(self, sync_client):
        _ = await sync_client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents="hello how are you",
        )

        assert sync_client.provider == "gemini"

        session = sync_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "gemini"
            assert usage_db.prompt_tokens > 0
            assert usage_db.completion_tokens > 0
            assert usage_db.total_tokens > 0
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "gemini"
        assert usage_last.providers[0].prompt_tokens > 0
        assert usage_last.providers[0].completion_tokens > 0
        assert usage_last.providers[0].total_tokens > 0

    @pytest.mark.asyncio
    async def test_async_stream(self, sync_client):
        usage_init: TokenUsageReport = usage.last_hour()
        assert usage_init.prompt_tokens == 0
        assert usage_init.completion_tokens == 0
        assert usage_init.total_tokens == 0

        chunks = []
        stream = await sync_client.aio.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="hello how are you",
        )
        async for chunk in stream:
            chunks.append(chunk)

        assert sync_client.provider == "gemini"

        session = sync_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "gemini"
            assert usage_db.prompt_tokens > 0
            assert usage_db.completion_tokens > 0
            assert usage_db.total_tokens > 0
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "gemini"
        assert usage_last.providers[0].prompt_tokens > 0
        assert usage_last.providers[0].completion_tokens > 0
        assert usage_last.providers[0].total_tokens > 0

    def test_sync_completion_with_wipe(self, sync_client):
        _ = sync_client.models.generate_content(
            model="gemini-2.0-flash",
            contents="hello how are you",
        )

        assert sync_client.provider == "gemini"

        session = sync_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "gemini"
            assert usage_db.prompt_tokens > 0
            assert usage_db.completion_tokens > 0
            assert usage_db.total_tokens > 0
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "gemini"
        assert usage_last.providers[0].prompt_tokens > 0
        assert usage_last.providers[0].completion_tokens > 0
        assert usage_last.providers[0].total_tokens > 0

        usage.wipe()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.prompt_tokens == 0
        assert usage_last.completion_tokens == 0
        assert usage_last.total_tokens == 0

        _ = sync_client.models.generate_content(
            model="gemini-2.0-flash",
            contents="hello how are you",
        )

        assert sync_client.provider == "gemini"

        session = sync_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "gemini"
            assert usage_db.prompt_tokens > 0
            assert usage_db.completion_tokens > 0
            assert usage_db.total_tokens > 0
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "gemini"
        assert usage_last.providers[0].prompt_tokens > 0
        assert usage_last.providers[0].completion_tokens > 0
        assert usage_last.providers[0].total_tokens > 0

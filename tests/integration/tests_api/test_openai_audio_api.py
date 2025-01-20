import os
import pytest
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from tokenator.models import TokenUsageReport
from tokenator.openai.client_openai import tokenator_openai
from tokenator.schemas import TokenUsage
from tokenator import usage
import tempfile
import base64


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
class TestOpenAIPromptCachingAPI:
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

    def test_sync_audio_generation(self, sync_client):
        response: ChatCompletion = sync_client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {"role": "user", "content": "Is a golden retriever a good family dog?"}
            ],
        )

        _ = base64.b64decode(response.choices[0].message.audio.data)

        assert sync_client.provider == "openai"

        session = sync_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "openai"
            assert usage_db.prompt_tokens == response.usage.prompt_tokens
            assert usage_db.completion_tokens == response.usage.completion_tokens
            assert usage_db.total_tokens == response.usage.total_tokens
            assert (
                usage_db.completion_audio_tokens
                == response.usage.completion_tokens_details.audio_tokens
            )
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
        assert (
            usage_last.providers[0].completion_tokens_details.audio_tokens
            == response.usage.completion_tokens_details.audio_tokens
        )

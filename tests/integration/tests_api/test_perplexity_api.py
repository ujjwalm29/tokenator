import os
import pytest
from openai import OpenAI
from tokenator.models import TokenUsageReport
from tokenator.openai.client_openai import tokenator_openai
from tokenator.schemas import TokenUsage
from tokenator import usage
import tempfile


@pytest.mark.skipif(
    not os.getenv("PERPLEXITY_API_KEY"),
    reason="PERPLEXITY_API_KEY environment variable not set",
)
class TestPerplexityAPI:
    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_tokens.db")
            yield db_path
        # Auto-cleanup when test ends

    @pytest.fixture
    def sync_client(self, temp_db):
        client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai",
        )
        return tokenator_openai(client, db_path=temp_db, provider="perplexity")

    def test_sync_completion_pricing(self, sync_client):
        response = sync_client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=[{"role": "user", "content": "Write a short story about a cat"}],
        )

        assert sync_client.provider == "perplexity"

        session = sync_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "perplexity"
            assert usage_db.prompt_tokens == response.usage.prompt_tokens
            assert usage_db.completion_tokens == response.usage.completion_tokens
            assert usage_db.total_tokens == response.usage.total_tokens
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "perplexity"
        assert usage_last.prompt_tokens == response.usage.prompt_tokens
        assert usage_last.completion_tokens == response.usage.completion_tokens
        assert usage_last.total_tokens == response.usage.total_tokens

        total_cost = (
            usage_last.prompt_tokens * 0.000001
            + usage_last.completion_tokens * 0.000001
        )  # taken from online. Might change
        assert total_cost == usage_last.total_cost

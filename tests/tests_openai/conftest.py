import pytest
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, CompletionUsage

@pytest.fixture
def mock_usage():
    return CompletionUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )

@pytest.fixture
def mock_chat_completion(mock_usage):
    return ChatCompletion(
        id="chatcmpl-123",
        model="gpt-4o",
        object="chat.completion",
        created=1677858242,
        usage=mock_usage,
        choices=[],
    )

@pytest.fixture
def sync_client():
    return OpenAI(api_key="test-key")

@pytest.fixture
def async_client():
    return AsyncOpenAI(api_key="test-key") 
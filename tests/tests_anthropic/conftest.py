import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, Usage


@pytest.fixture
def mock_usage():
    return Usage(input_tokens=10, output_tokens=20)


@pytest.fixture
def mock_message(mock_usage):
    return Message(
        id="msg_123",
        type="message",
        role="assistant",
        content=[],
        model="claude-3",
        usage=mock_usage,
        stop_reason="end_turn",
        stop_sequence=None,
    )


@pytest.fixture
def sync_client():
    return Anthropic(api_key="test-key")


@pytest.fixture
def async_client():
    return AsyncAnthropic(api_key="test-key")

import pytest
from unittest.mock import AsyncMock, Mock, patch
import tempfile
import os

from tokenator.openai.client_openai import tokenator_openai
from tokenator.schemas import TokenUsage
from tokenator.migrations import check_and_run_migrations
from sqlalchemy.exc import SQLAlchemyError
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage
from openai import APIConnectionError, RateLimitError
from openai import OpenAI, AsyncOpenAI


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_tokens.db")
        check_and_run_migrations(db_path=db_path)
        yield db_path
    # Auto-cleanup when test ends


@pytest.fixture
def test_sync_client(sync_client, temp_db):
    return tokenator_openai(sync_client, db_path=temp_db)


@pytest.fixture
def test_async_client(async_client, temp_db):
    return tokenator_openai(async_client, db_path=temp_db)


def test_init_sync_client(test_sync_client, sync_client):
    assert test_sync_client.client == sync_client


def test_init_async_client(test_async_client, async_client):
    assert test_async_client.client == async_client


def test_init_invalid_client():
    with pytest.raises(ValueError, match="Client must be an instance"):
        tokenator_openai(Mock())


def test_db_path_creation_sync():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db", "tokens.db")
        _ = tokenator_openai(OpenAI(api_key="test"), db_path=db_path)
        assert os.path.exists(os.path.dirname(db_path))


def test_db_path_creation_async():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db", "tokens.db")
        _ = tokenator_openai(AsyncOpenAI(api_key="test"), db_path=db_path)
        assert os.path.exists(os.path.dirname(db_path))


@pytest.mark.asyncio
async def test_async_create_with_usage(test_async_client, mock_chat_completion):
    with patch.object(
        test_async_client.client.chat.completions, "create"
    ) as mock_create:
        mock_create.return_value = AsyncMock(return_value=mock_chat_completion)()

        response = await test_async_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
        )

        assert response == mock_chat_completion

        session = test_async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage.provider == "openai"
            assert usage.model == "gpt-4o"
            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 20
            assert usage.total_tokens == 30
        finally:
            session.close()


def test_sync_create_with_usage(test_sync_client, mock_chat_completion):
    with patch.object(
        test_sync_client.client.chat.completions, "create"
    ) as mock_create:
        mock_create.return_value = mock_chat_completion

        response = test_sync_client.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
        )

        assert response == mock_chat_completion

        session = test_sync_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage.provider == "openai"
            assert usage.model == "gpt-4o"
            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 20
            assert usage.total_tokens == 30
        finally:
            session.close()


def test_missing_usage_stats(test_sync_client):
    mock_completion = ChatCompletion(
        id="chatcmpl-123",
        model="gpt-4o",
        object="chat.completion",
        created=1677858242,
        choices=[],
        usage=None,
    )

    with patch.object(
        test_sync_client.client.chat.completions, "create"
    ) as mock_create:
        mock_create.return_value = mock_completion

        _ = test_sync_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
        )

        session = test_sync_client.Session()
        try:
            usage_count = session.query(TokenUsage).count()
            assert usage_count == 0
        finally:
            session.close()


def test_db_error_handling(test_sync_client, mock_chat_completion):
    with (
        patch(
            "tokenator.openai.client_openai.BaseOpenAIWrapper._log_usage_impl"
        ) as mock_log,
        patch.object(test_sync_client.client.chat.completions, "create") as mock_create,
    ):
        mock_create.return_value = mock_chat_completion
        mock_log.side_effect = SQLAlchemyError("DB Error")

        response = test_sync_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
        )
        assert response == mock_chat_completion


def test_api_error_handling(test_sync_client):
    with patch.object(
        test_sync_client.client.chat.completions, "create"
    ) as mock_create:
        mock_create.side_effect = APIConnectionError(
            message="API Error", request=Mock()
        )

        with pytest.raises(APIConnectionError):
            test_sync_client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
            )

        session = test_sync_client.Session()
        try:
            assert session.query(TokenUsage).count() == 0
        finally:
            session.close()


def test_rate_limit_error(test_sync_client):
    with patch.object(test_sync_client.chat.completions, "create") as mock_create:
        mock_create.side_effect = RateLimitError(
            message="Rate error", body={}, response=Mock()
        )

        with pytest.raises(RateLimitError):
            test_sync_client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
            )


def test_malformed_response(test_sync_client):
    malformed_completion = {
        "id": "chatcmpl-123",
        "model": "gpt-4o",
        "object": "chat.completion",
        "created": 1677858242,
        "choices": [],
        # Malformed usage data
        "usage": {
            "prompt_tokens": "invalid",  # Should be int
            "completion_tokens": None,
            "total_tokens": -1,
        },
    }

    with patch.object(test_sync_client.chat.completions, "create") as mock_create:
        mock_create.return_value = malformed_completion

        response = test_sync_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
        )

        # Should still return response but not log usage
        assert response == malformed_completion

        session = test_sync_client.Session()
        try:
            assert session.query(TokenUsage).count() == 0
        finally:
            session.close()


@pytest.mark.asyncio
async def test_log_usage_auto_generates_uuid(test_async_client, mock_chat_completion):
    with patch.object(
        test_async_client.client.chat.completions, "create"
    ) as mock_create:
        mock_create.return_value = AsyncMock(return_value=mock_chat_completion)()

        _ = await test_async_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
        )

        session = test_async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage.execution_id is not None
        finally:
            session.close()


@pytest.mark.asyncio
async def test_custom_execution_id(test_async_client, mock_chat_completion):
    custom_id = "test-execution-123"

    with patch.object(
        test_async_client.client.chat.completions, "create"
    ) as mock_create:
        mock_create.return_value = AsyncMock(return_value=mock_chat_completion)()

        _ = await test_async_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            execution_id=custom_id,
        )

        session = test_async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage.execution_id == custom_id
        finally:
            session.close()


def test_custom_execution_id_sync(test_sync_client, mock_chat_completion):
    custom_id = "test-execution-123"

    with patch.object(
        test_sync_client.client.chat.completions, "create"
    ) as mock_create:
        mock_create.return_value = mock_chat_completion

        _ = test_sync_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            execution_id=custom_id,
        )

        session = test_sync_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage.execution_id == custom_id
        finally:
            session.close()


@pytest.mark.asyncio
async def test_async_streaming(test_async_client, mock_chat_completion):
    chunks = [
        ChatCompletionChunk(
            id="1",
            choices=[],
            model="gpt-4",
            object="chat.completion.chunk",
            created=1,
            usage=None,
        ),
        ChatCompletionChunk(
            id="2",
            choices=[],
            model="gpt-4",
            object="chat.completion.chunk",
            created=1,
            usage=None,
        ),
        ChatCompletionChunk(
            id="3",
            choices=[],
            model="gpt-4",
            object="chat.completion.chunk",
            created=1,
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
        ),
    ]

    with patch.object(
        test_async_client.client.chat.completions, "create"
    ) as mock_create:
        # Create an async iterator class for the chunks
        class ChunkStream:
            def __init__(self, chunks):
                self.chunks = chunks
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        # Set up the mock to return our stream directly
        mock_create.return_value = AsyncMock(return_value=ChunkStream(chunks))()

        collected_chunks = []
        stream = await test_async_client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Hello"}], stream=True
        )

        async for chunk in stream:
            collected_chunks.append(chunk)

        assert len(collected_chunks) == 3

        session = test_async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.total_tokens == 30
        finally:
            session.close()


def test_sync_streaming(test_sync_client, mock_chat_completion):
    chunks = [
        ChatCompletionChunk(
            id="1",
            choices=[],
            model="gpt-4",
            object="chat.completion.chunk",
            created=1,
            usage=None,
        ),
        ChatCompletionChunk(
            id="2",
            choices=[],
            model="gpt-4",
            object="chat.completion.chunk",
            created=1,
            usage=None,
        ),
        ChatCompletionChunk(
            id="3",
            choices=[],
            model="gpt-4",
            object="chat.completion.chunk",
            created=1,
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
        ),
    ]

    with patch.object(
        test_sync_client.client.chat.completions, "create"
    ) as mock_create:
        mock_create.return_value = iter(chunks)

        collected_chunks = []
        for chunk in test_sync_client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Hello"}], stream=True
        ):
            collected_chunks.append(chunk)

        assert len(collected_chunks) == 3

        session = test_sync_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.total_tokens == 30
        finally:
            session.close()


def test_sync_streaming_with_include_usage(test_sync_client, mock_chat_completion):
    chunks = [
        ChatCompletionChunk(
            id="1",
            choices=[],
            model="gpt-4",
            object="chat.completion.chunk",
            created=1,
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
        ),
        ChatCompletionChunk(
            id="2",
            choices=[],
            model="gpt-4",
            object="chat.completion.chunk",
            created=1,
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
        ),
        ChatCompletionChunk(
            id="3",
            choices=[],
            model="gpt-4",
            object="chat.completion.chunk",
            created=1,
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
        ),
    ]

    with patch.object(
        test_sync_client.client.chat.completions, "create"
    ) as mock_create:
        mock_create.return_value = iter(chunks)

        collected_chunks = []
        for chunk in test_sync_client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Hello"}], stream=True
        ):
            collected_chunks.append(chunk)

        assert len(collected_chunks) == 3

        session = test_sync_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.total_tokens == 90
        finally:
            session.close()


def test_streaming_no_final_usage(test_sync_client):
    # Test when no chunk has usage stats
    chunks = [
        ChatCompletionChunk(
            id="1",
            choices=[],
            model="gpt-4",
            object="chat.completion.chunk",
            created=1,
            usage=None,
        ),
        ChatCompletionChunk(
            id="2",
            choices=[],
            model="gpt-4",
            object="chat.completion.chunk",
            created=1,
            usage=None,
        ),
    ]

    with patch.object(
        test_sync_client.client.chat.completions, "create"
    ) as mock_create:
        mock_create.return_value = iter(chunks)

        collected_chunks = []
        for chunk in test_sync_client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Hello"}], stream=True
        ):
            collected_chunks.append(chunk)

        session = test_sync_client.Session()
        try:
            assert session.query(TokenUsage).count() == 0
        finally:
            session.close()


def test_streaming_empty_response(test_sync_client):
    # Test with empty stream
    with patch.object(
        test_sync_client.client.chat.completions, "create"
    ) as mock_create:
        mock_create.return_value = iter([])

        collected_chunks = []
        for chunk in test_sync_client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Hello"}], stream=True
        ):
            collected_chunks.append(chunk)

        assert len(collected_chunks) == 0

        session = test_sync_client.Session()
        try:
            assert session.query(TokenUsage).count() == 0
        finally:
            session.close()


@pytest.mark.asyncio
async def test_streaming_with_error(test_async_client):
    with patch.object(
        test_async_client.client.chat.completions, "create"
    ) as mock_create:
        mock_create.side_effect = RateLimitError(
            message="Rate limit exceeded", body={}, response=Mock()
        )

        with pytest.raises(RateLimitError):
            async for chunk in await test_async_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            ):
                pass

        session = test_async_client.Session()
        try:
            assert session.query(TokenUsage).count() == 0
        finally:
            session.close()

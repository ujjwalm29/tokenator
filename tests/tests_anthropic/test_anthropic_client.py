import pytest
from unittest.mock import AsyncMock, Mock, patch
import tempfile
import os

from tokenator.client_anthropic import tokenator_anthropic
from tokenator.schemas import TokenUsage
from sqlalchemy.exc import SQLAlchemyError
from anthropic.types import (
    Message, Usage, MessageStartEvent, MessageStopEvent,
    ContentBlockStartEvent, ContentBlockDeltaEvent,
    ContentBlock, TextDelta, TextBlock
)
from anthropic import Anthropic, AsyncAnthropic, BadRequestError, RateLimitError

@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "tokens.db")
        yield db_path

@pytest.fixture
def test_sync_client(sync_client, temp_db):
    return tokenator_anthropic(sync_client, db_path=temp_db)

@pytest.fixture
def test_async_client(async_client, temp_db):
    return tokenator_anthropic(async_client, db_path=temp_db)

def test_init_sync_client(test_sync_client, sync_client):
    assert test_sync_client.client == sync_client

def test_init_async_client(test_async_client, async_client):
    assert test_async_client.client == async_client

def test_init_invalid_client():
    with pytest.raises(ValueError, match="Client must be an instance"):
        tokenator_anthropic(Mock())

def test_db_path_creation_sync():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db", "tokens.db")
        wrapper = tokenator_anthropic(Anthropic(api_key="test"), db_path=db_path)
        assert os.path.exists(os.path.dirname(db_path))

def test_db_path_creation_async():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db", "tokens.db")
        wrapper = tokenator_anthropic(AsyncAnthropic(api_key="test"), db_path=db_path)
        assert os.path.exists(os.path.dirname(db_path))

@pytest.mark.asyncio
async def test_async_create_with_usage(test_async_client, mock_message):
    with patch.object(test_async_client.client.messages, 'create') as mock_create:
        mock_create.return_value = AsyncMock(return_value=mock_message)()
        
        response = await test_async_client.messages.create(
            model="claude-3",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert response == mock_message
        
        session = test_async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage.provider == "anthropic"
            assert usage.model == "claude-3"
            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 20
            assert usage.total_tokens == 30
        finally:
            session.close()

def test_sync_create_with_usage(test_sync_client, mock_message):
    with patch.object(test_sync_client.client.messages, 'create') as mock_create:
        mock_create.return_value = mock_message
        
        response = test_sync_client.messages.create(
            model="claude-3",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert response == mock_message
        
        session = test_sync_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage.provider == "anthropic"
            assert usage.model == "claude-3"
            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 20
            assert usage.total_tokens == 30
        finally:
            session.close()

def test_zero_usage_stats(test_sync_client):
    mock_msg = Message(
        id="msg_123",
        type="message",
        role="assistant",
        content=[],
        model="claude-3",
        stop_reason="end_turn",
        usage=Usage(
            input_tokens=0,
            output_tokens=0
        )
    )
    
    with patch.object(test_sync_client.client.messages, 'create') as mock_create:
        mock_create.return_value = mock_msg
        
        response = test_sync_client.messages.create(
            model="claude-3",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        session = test_sync_client.Session()
        try:
            usage_count = session.query(TokenUsage).first()
            assert usage_count.prompt_tokens == 0
            assert usage_count.completion_tokens == 0
            assert usage_count.total_tokens == 0
        finally:
            session.close()

def test_db_error_handling(test_sync_client, mock_message):
    with patch('tokenator.client_anthropic.BaseAnthropicWrapper._log_usage_impl') as mock_log, \
         patch.object(test_sync_client.client.messages, 'create') as mock_create:
        mock_create.return_value = mock_message
        mock_log.side_effect = SQLAlchemyError("DB Error")
        
        response = test_sync_client.messages.create(
            model="claude-3",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert response == mock_message

def test_api_error_handling(test_sync_client):
    with patch.object(test_sync_client.client.messages, 'create') as mock_create:
        mock_create.side_effect = BadRequestError(
            response=Mock(status_code=400),
            body=None,
            message="Bad Request"
        )
        
        with pytest.raises(BadRequestError):
            test_sync_client.messages.create(
                model="claude-3",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
        session = test_sync_client.Session()
        try:
            assert session.query(TokenUsage).count() == 0
        finally:
            session.close()

def test_rate_limit_error(test_sync_client):
    with patch.object(test_sync_client.messages, 'create') as mock_create:
        mock_create.side_effect = RateLimitError(
            message="Rate error",
            response=Mock(status_code=429),
            body=None
        )
        
        with pytest.raises(RateLimitError):
            test_sync_client.messages.create(
                model="claude-3",
                messages=[{"role": "user", "content": "Hello"}]
            )

@pytest.mark.asyncio
async def test_log_usage_auto_generates_uuid(test_async_client, mock_message):
    with patch.object(test_async_client.client.messages, 'create') as mock_create:
        mock_create.return_value = AsyncMock(return_value=mock_message)()
        
        response = await test_async_client.messages.create(
            model="claude-3",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        session = test_async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage.execution_id is not None
        finally:
            session.close()

@pytest.mark.asyncio
async def test_custom_execution_id(test_async_client, mock_message):
    custom_id = "test-execution-123"
    
    with patch.object(test_async_client.client.messages, 'create') as mock_create:
        mock_create.return_value = AsyncMock(return_value=mock_message)()
        
        response = await test_async_client.messages.create(
            model="claude-3",
            messages=[{"role": "user", "content": "Hello"}],
            execution_id=custom_id
        )
        
        session = test_async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage.execution_id == custom_id
        finally:
            session.close()

def test_custom_execution_id_sync(test_sync_client, mock_message):
    custom_id = "test-execution-123"
    
    with patch.object(test_sync_client.client.messages, 'create') as mock_create:
        mock_create.return_value = mock_message
        
        response = test_sync_client.messages.create(
            model="claude-3",
            messages=[{"role": "user", "content": "Hello"}],
            execution_id=custom_id
        )
        
        session = test_sync_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage.execution_id == custom_id
        finally:
            session.close()

@pytest.mark.asyncio
async def test_async_streaming(test_async_client):
    chunks = [
        MessageStartEvent(
            type="message_start",
            message=Message(
                id="msg_1",
                type="message",
                role="assistant",
                content=[],
                model="claude-3",
                usage=Usage(input_tokens=0, output_tokens=0),
                stop_reason=None,
                stop_sequence=None
            )
        ),
        ContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=TextBlock(type="text", text="Hello")
        ),
        ContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=TextDelta(type="text_delta", text=" world")
        ),
        MessageStopEvent(
            type="message_stop",
            message=Message(
                id="msg_1",
                type="message",
                role="assistant",
                content=[TextBlock(type="text", text="Hello world")],
                model="claude-3",
                usage=Usage(input_tokens=10, output_tokens=20),
                stop_reason="end_turn",
                stop_sequence=None
            )
        )
    ]
    
    async def async_iter():
        for chunk in chunks:
            yield chunk

    with patch.object(test_async_client.client.messages, 'create') as mock_create:
        mock_create.return_value = async_iter()
        
        collected_chunks = []
        stream = await test_async_client.messages.create(
            model="claude-3",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        async for chunk in stream:
            collected_chunks.append(chunk)
        
        assert len(collected_chunks) == 4
        
        session = test_async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.total_tokens == 30
        finally:
            session.close()

def test_sync_streaming(test_sync_client):
    chunks = [
        MessageStartEvent(
            type="message_start",
            message=Message(
                id="msg_1",
                type="message",
                role="assistant",
                content=[],
                model="claude-3",
                usage=Usage(input_tokens=0, output_tokens=0),
                stop_reason=None,
                stop_sequence=None
            )
        ),
        ContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=TextBlock(type="text", text="Hello")
        ),
        ContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=TextDelta(type="text_delta", text=" world")
        ),
        MessageStopEvent(
            type="message_stop",
            message=Message(
                id="msg_1",
                type="message",
                role="assistant",
                content=[TextBlock(type="text", text="Hello world")],
                model="claude-3",
                usage=Usage(input_tokens=10, output_tokens=20),
                stop_reason="end_turn",
                stop_sequence=None
            )
        )
    ]
    
    with patch.object(test_sync_client.client.messages, 'create') as mock_create:
        mock_create.return_value = iter(chunks)
        
        collected_chunks = []
        for chunk in test_sync_client.messages.create(
            model="claude-3",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        ):
            collected_chunks.append(chunk)
        
        assert len(collected_chunks) == 4
        
        session = test_sync_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage is not None
            assert usage.total_tokens == 30
        finally:
            session.close()

def test_streaming_no_final_usage(test_sync_client):
    chunks = [
        MessageStartEvent(
            type="message_start",
            message=Message(
                id="msg_1",
                type="message",
                role="assistant",
                content=[],
                model="claude-3",
                usage=Usage(input_tokens=0, output_tokens=0),
                stop_reason=None,
                stop_sequence=None
            )
        ),
        MessageStopEvent(
            type="message_stop",
            message=Message(
                id="msg_1",
                type="message",
                role="assistant",
                content=[TextBlock(type="text", text="Hello world")],
                model="claude-3",
                usage=Usage(input_tokens=0, output_tokens=0),
                stop_reason="end_turn",
                stop_sequence=None
            )
        )
    ]
    
    with patch.object(test_sync_client.client.messages, 'create') as mock_create:
        mock_create.return_value = iter(chunks)
        
        collected_chunks = []
        for chunk in test_sync_client.messages.create(
            model="claude-3",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        ):
            collected_chunks.append(chunk)
        
        session = test_sync_client.Session()
        try:
            assert session.query(TokenUsage).count() == 0
        finally:
            session.close()

@pytest.mark.asyncio
async def test_streaming_with_error(test_async_client):
    with patch.object(test_async_client.client.messages, 'create') as mock_create:
        mock_create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=Mock(status_code=429),
            body=None
        )
        
        with pytest.raises(RateLimitError):
            async for chunk in await test_async_client.messages.create(
                model="claude-3",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True
            ):
                pass
        
        session = test_async_client.Session()
        try:
            assert session.query(TokenUsage).count() == 0
        finally:
            session.close()

import pytest
from unittest.mock import AsyncMock, Mock, patch
import tempfile
import os

from tokenator.client_openai import tokenator_openai
from tokenator.models import TokenUsage
from sqlalchemy.exc import SQLAlchemyError
from openai.types.chat import ChatCompletion
from openai import APIConnectionError, RateLimitError
from openai import OpenAI, AsyncOpenAI

@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "tokens.db")
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
        wrapper = tokenator_openai(OpenAI(api_key="test"), db_path=db_path)
        assert os.path.exists(os.path.dirname(db_path))

def test_db_path_creation_async():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db", "tokens.db")
        wrapper = tokenator_openai(AsyncOpenAI(api_key="test"), db_path=db_path)
        assert os.path.exists(os.path.dirname(db_path))

@pytest.mark.asyncio
async def test_async_create_with_usage(test_async_client, mock_chat_completion):
    with patch.object(test_async_client.client.chat.completions, 'create') as mock_create:
        mock_create.return_value = AsyncMock(return_value=mock_chat_completion)()
        
        response = await test_async_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
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
    with patch.object(test_sync_client.client.chat.completions, 'create') as mock_create:
        mock_create.return_value = mock_chat_completion
        
        response = test_sync_client.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
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
        usage=None
    )
    
    with patch.object(test_sync_client.client.chat.completions, 'create') as mock_create:
        mock_create.return_value = mock_completion
        
        response = test_sync_client.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        session = test_sync_client.Session()
        try:
            usage_count = session.query(TokenUsage).count()
            assert usage_count == 0
        finally:
            session.close()

def test_db_error_handling(test_sync_client, mock_chat_completion):
    with patch('tokenator.client_openai.BaseOpenAIWrapper._log_usage_impl') as mock_log, \
         patch.object(test_sync_client.client.chat.completions, 'create') as mock_create:
        mock_create.return_value = mock_chat_completion
        mock_log.side_effect = SQLAlchemyError("DB Error")
        
        response = test_sync_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert response == mock_chat_completion 

def test_api_error_handling(test_sync_client):
    with patch.object(test_sync_client.client.chat.completions, 'create') as mock_create:
        mock_create.side_effect = APIConnectionError(
            message="API Error",
            request=Mock()
        )
        
        with pytest.raises(APIConnectionError):
            test_sync_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
        session = test_sync_client.Session()
        try:
            assert session.query(TokenUsage).count() == 0
        finally:
            session.close()

def test_rate_limit_error(test_sync_client):
    with patch.object(test_sync_client.chat.completions, 'create') as mock_create:
        mock_create.side_effect = RateLimitError(
            message="Rate error",
            body={},
            response=Mock()
        )
        
        with pytest.raises(RateLimitError):
            test_sync_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}]
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
            "total_tokens": -1
        }
    }
    
    with patch.object(test_sync_client.chat.completions, 'create') as mock_create:
        mock_create.return_value = malformed_completion
        
        response = test_sync_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
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
    with patch.object(test_async_client.client.chat.completions, 'create') as mock_create:
        mock_create.return_value = AsyncMock(return_value=mock_chat_completion)()
        
        response = await test_async_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
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
    
    with patch.object(test_async_client.client.chat.completions, 'create') as mock_create:
        mock_create.return_value = AsyncMock(return_value=mock_chat_completion)()
        
        response = await test_async_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            execution_id=custom_id
        )
        
        session = test_async_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage.execution_id == custom_id
        finally:
            session.close()

def test_custom_execution_id_sync(test_sync_client, mock_chat_completion):
    custom_id = "test-execution-123"
    
    with patch.object(test_sync_client.client.chat.completions, 'create') as mock_create:
        mock_create.return_value = mock_chat_completion
        
        response = test_sync_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            execution_id=custom_id
        )
        
        session = test_sync_client.Session()
        try:
            usage = session.query(TokenUsage).first()
            assert usage.execution_id == custom_id
        finally:
            session.close()


import pytest
from unittest.mock import Mock, patch
import tempfile
import os

from src.tokenator.client_openai import tokenator_openai
from src.tokenator.models import TokenUsage
from sqlalchemy.exc import SQLAlchemyError
from openai.types.chat import ChatCompletion, Choice, ChatCompletionMessage
from openai.types import APIError, RateLimitError
import time

def test_init_sync_client(sync_client):
    wrapper = tokenator_openai(sync_client)
    assert wrapper.client == sync_client

def test_init_async_client(async_client):
    wrapper = tokenator_openai(async_client)
    assert wrapper.client == async_client

def test_init_invalid_client():
    with pytest.raises(ValueError, match="Client must be an instance"):
        tokenator_openai(Mock())

def test_db_path_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db", "tokens.db")
        wrapper = tokenator_openai(OpenAI(api_key="test"), db_path=db_path)
        assert os.path.exists(os.path.dirname(db_path))

@pytest.mark.asyncio
async def test_async_create_with_usage(async_client, mock_chat_completion):
    with patch.object(async_client.chat.completions, 'create') as mock_create:
        mock_create.return_value = mock_chat_completion
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "tokens.db")
            wrapper = tokenator_openai(async_client, db_path=db_path)
            
            response = await wrapper.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            # Verify response
            assert response == mock_chat_completion
            
            # Verify database entry
            session = wrapper.Session()
            try:
                usage = session.query(TokenUsage).first()
                assert usage.provider == "openai"
                assert usage.model == "gpt-4o"
                assert usage.prompt_tokens == 10
                assert usage.completion_tokens == 20
                assert usage.total_tokens == 30
            finally:
                session.close()

def test_sync_create_with_usage(sync_client, mock_chat_completion):
    with patch.object(sync_client.chat.completions, 'create') as mock_create:
        mock_create.return_value = mock_chat_completion
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "tokens.db")
            wrapper = tokenator_openai(sync_client, db_path=db_path)
            
            response = wrapper.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            # Verify response
            assert response == mock_chat_completion
            
            # Verify database entry
            session = wrapper.Session()
            try:
                usage = session.query(TokenUsage).first()
                assert usage.provider == "openai"
                assert usage.model == "gpt-4o"
                assert usage.prompt_tokens == 10
                assert usage.completion_tokens == 20
                assert usage.total_tokens == 30
            finally:
                session.close()

def test_missing_usage_stats(sync_client):
    mock_completion = ChatCompletion(
        id="chatcmpl-123",
        model="gpt-4o",
        object="chat.completion",
        created=1677858242,
        choices=[],
        usage=None  # No usage stats
    )
    
    with patch.object(sync_client.chat.completions, 'create') as mock_create:
        mock_create.return_value = mock_completion
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "tokens.db")
            wrapper = tokenator_openai(sync_client, db_path=db_path)
            
            response = wrapper.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            # Verify no DB entries were made
            session = wrapper.Session()
            try:
                usage_count = session.query(TokenUsage).count()
                assert usage_count == 0
            finally:
                session.close()

def test_db_error_handling(sync_client, mock_chat_completion):
    with patch('tokenator.client_openai.BaseOpenAIWrapper._log_usage_impl') as mock_log:
        mock_log.side_effect = SQLAlchemyError("DB Error")
        
        wrapper = tokenator_openai(sync_client)
        
        # Should not raise error but log it
        response = wrapper.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert response == mock_chat_completion 

def test_api_error_handling(sync_client):
    with patch.object(sync_client.chat.completions, 'create') as mock_create:
        mock_create.side_effect = APIError(
            message="API Error", 
            code="api_error", 
            type="invalid_request_error",
            body={},
            response=Mock()
        )
        
        wrapper = tokenator_openai(sync_client)
        
        with pytest.raises(APIError):
            wrapper.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
        # Verify no usage was logged
        session = wrapper.Session()
        try:
            assert session.query(TokenUsage).count() == 0
        finally:
            session.close()

def test_rate_limit_error(sync_client):
    with patch.object(sync_client.chat.completions, 'create') as mock_create:
        mock_create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            code="rate_limit_exceeded",
            type="rate_limit_error",
            body={},
            response=Mock()
        )
        
        wrapper = tokenator_openai(sync_client)
        
        with pytest.raises(RateLimitError):
            wrapper.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}]
            )

def test_malformed_response(sync_client):
    malformed_completion = ChatCompletion(
        id="chatcmpl-123",
        model="gpt-4o",
        object="chat.completion",
        created=1677858242,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Hello"
                ),
                finish_reason="stop"
            )
        ],
        # Malformed usage data
        usage={
            "prompt_tokens": "invalid",  # Should be int
            "completion_tokens": None,
            "total_tokens": -1
        }
    )
    
    with patch.object(sync_client.chat.completions, 'create') as mock_create:
        mock_create.return_value = malformed_completion
        
        wrapper = tokenator_openai(sync_client)
        response = wrapper.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should still return response but not log usage
        assert response == malformed_completion
        
        session = wrapper.Session()
        try:
            assert session.query(TokenUsage).count() == 0
        finally:
            session.close()

@pytest.mark.asyncio
async def test_async_api_error(async_client):
    with patch.object(async_client.chat.completions, 'create') as mock_create:
        mock_create.side_effect = APIError(
            message="Async API Error",
            code="api_error",
            type="invalid_request_error", 
            body={},
            response=Mock()
        )
        
        wrapper = tokenator_openai(async_client)
        
        with pytest.raises(APIError):
            await wrapper.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}]
            )
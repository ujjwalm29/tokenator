"""OpenAI client wrapper with token usage tracking."""

from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union, overload, Iterator, AsyncIterator
import logging

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from .models import get_session, TokenUsage
import uuid

logger = logging.getLogger(__name__)

ResponseType = TypeVar('ResponseType', ChatCompletion, Dict[str, Any])

class BaseOpenAIWrapper:
    def __init__(self, client: Union[OpenAI, AsyncOpenAI], db_path: Optional[str] = None):
        """Initialize the OpenAI wrapper."""
        self.client = client

        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info("Created database directory at: %s", Path(db_path).parent)

        self.Session = get_session(db_path)
        
        logger.debug("Initializing %s with db_path: %s", 
                    self.__class__.__name__, db_path)

    def _log_usage_impl(self, model: str, usage: Dict[str, int], session, execution_id: str) -> None:
        """Implementation of token usage logging."""
        logger.debug("Logging usage for model %s: %s", model, usage)
        try:
            token_usage = TokenUsage(
                execution_id=execution_id,
                provider="openai",
                model=model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0)
            )
            session.add(token_usage)
            logger.info("Logged token usage: model=%s, total_tokens=%d", 
                       model, usage.get("total_tokens", 0))
        except Exception as e:
            logger.error("Failed to log token usage: %s", str(e))


    def _process_response_usage(self, response: ResponseType) -> Optional[Dict[str, Any]]:
        """Process and log usage statistics from a response."""
        try:
            if isinstance(response, ChatCompletion):
                if response.usage is None:
                    return None
                return {
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                }
            elif isinstance(response, dict):
                usage = response.get('usage')
                if not usage:
                    return None
                return {
                    "model": response.get('model', 'unknown'),
                    "usage": {
                        "prompt_tokens": usage.get('prompt_tokens', 0),
                        "completion_tokens": usage.get('completion_tokens', 0),
                        "total_tokens": usage.get('total_tokens', 0),
                    }
                }
        except Exception as e:
            logger.warning("Failed to process usage stats: %s", str(e))
            return None
        
        return None

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def _log_usage(self, model: str, usage: Dict[str, int], execution_id: Optional[str] = None):
        """Log token usage to database."""
        if not execution_id:
            execution_id = str(uuid.uuid4())

        session = self.Session()
        try:
            try:
                self._log_usage_impl(model, usage, session, execution_id)
                session.commit()
            except Exception as e:
                logger.error("Failed to log token usage: %s", str(e))
                session.rollback()
        finally:
            session.close()

class OpenAIWrapper(BaseOpenAIWrapper):
    def create(self, *args: Any, execution_id: Optional[str] = None, **kwargs: Any) -> Union[ChatCompletion, Iterator[ChatCompletion]]:
        """Create a chat completion and log token usage."""
        logger.debug("Creating chat completion with args: %s, kwargs: %s", args, kwargs)
        
        response = self.client.chat.completions.create(*args, **kwargs)
        
        if not kwargs.get('stream', False):
            usage_data = self._process_response_usage(response)
            if usage_data:
                self._log_usage(**usage_data, execution_id=execution_id)
            return response
        
        return self._wrap_streaming_response(response, execution_id)
    
    def _wrap_streaming_response(self, response_iter: Iterator[ChatCompletion], execution_id: Optional[str]) -> Iterator[ChatCompletion]:
        """Wrap streaming response to capture final usage stats"""
        last_chunk = None
        for chunk in response_iter:
            if isinstance(chunk, ChatCompletion) and chunk.usage is not None:
                last_chunk = chunk
            yield chunk
            
        if last_chunk:
            usage_data = self._process_response_usage(last_chunk)
            if usage_data:
                self._log_usage(**usage_data, execution_id=execution_id)

class AsyncOpenAIWrapper(BaseOpenAIWrapper):
    async def create(self, *args: Any, execution_id: Optional[str] = None, **kwargs: Any) -> Union[ChatCompletion, AsyncIterator[ChatCompletion]]:
        """Create a chat completion and log token usage."""
        logger.debug("Creating chat completion with args: %s, kwargs: %s", args, kwargs)
        
        if kwargs.get('stream', False):
            response = self.client.chat.completions.create(*args, **kwargs)
            return self._wrap_streaming_response(response, execution_id)
        
        response = await self.client.chat.completions.create(*args, **kwargs)
        usage_data = self._process_response_usage(response)
        if usage_data:
            self._log_usage(**usage_data, execution_id=execution_id)
        return response

    async def _wrap_streaming_response(self, response_iter: AsyncIterator[ChatCompletion], execution_id: Optional[str]) -> AsyncIterator[ChatCompletion]:
        """Wrap streaming response to capture final usage stats"""
        last_chunk = None
        async for chunk in response_iter:
            if isinstance(chunk, ChatCompletion) and chunk.usage is not None:
                last_chunk = chunk
            yield chunk
            
        if last_chunk:
            usage_data = self._process_response_usage(last_chunk)
            if usage_data:
                self._log_usage(**usage_data, execution_id=execution_id)

@overload
def tokenator_openai(
    client: OpenAI,
    db_path: Optional[str] = None,
) -> OpenAIWrapper: ...

@overload
def tokenator_openai(
    client: AsyncOpenAI,
    db_path: Optional[str] = None,
) -> AsyncOpenAIWrapper: ...

def tokenator_openai(
    client: Union[OpenAI, AsyncOpenAI],
    db_path: Optional[str] = None,
) -> Union[OpenAIWrapper, AsyncOpenAIWrapper]:
    """Create a token-tracking wrapper for an OpenAI client.
    
    Args:
        client: OpenAI or AsyncOpenAI client instance
        db_path: Optional path to SQLite database for token tracking
    """
    if isinstance(client, OpenAI):
        return OpenAIWrapper(client=client, db_path=db_path)
    
    if isinstance(client, AsyncOpenAI):
        return AsyncOpenAIWrapper(client=client, db_path=db_path)
        
    raise ValueError("Client must be an instance of OpenAI or AsyncOpenAI")

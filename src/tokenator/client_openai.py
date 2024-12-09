"""OpenAI client wrapper with token usage tracking."""

from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union, overload, Iterator, AsyncIterator
import logging

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .schemas import get_session, TokenUsage
from .models import Usage, TokenUsageStats
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

    def _log_usage_impl(self, token_usage_stats: TokenUsageStats, session, execution_id: str) -> None:
        """Implementation of token usage logging."""
        logger.debug("Logging usage for model %s: %s", token_usage_stats.model, token_usage_stats.usage.model_dump())
        try:
            token_usage = TokenUsage(
                execution_id=execution_id,
                provider="openai",
                model=token_usage_stats.model,
                prompt_tokens=token_usage_stats.usage.prompt_tokens,
                completion_tokens=token_usage_stats.usage.completion_tokens,
                total_tokens=token_usage_stats.usage.total_tokens
            )
            session.add(token_usage)
            logger.info("Logged token usage: model=%s, total_tokens=%d", 
                       token_usage_stats.model, token_usage_stats.usage.total_tokens)
        except Exception as e:
            logger.error("Failed to log token usage: %s", str(e))


    def _process_response_usage(self, response: ResponseType) -> Optional[TokenUsageStats]:
        """Process and log usage statistics from a response."""
        try:
            if isinstance(response, ChatCompletion):
                if response.usage is None:
                    return None
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
                return TokenUsageStats(model=response.model, usage=usage)
                
            elif isinstance(response, dict):
                usage_dict = response.get('usage')
                if not usage_dict:
                    return None
                usage = Usage(
                    prompt_tokens=usage_dict.get('prompt_tokens', 0),
                    completion_tokens=usage_dict.get('completion_tokens', 0),
                    total_tokens=usage_dict.get('total_tokens', 0)
                )
                return TokenUsageStats(
                    model=response.get('model', 'unknown'),
                    usage=usage
                )
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

    def _log_usage(self, token_usage_stats: TokenUsageStats, execution_id: Optional[str] = None):
        """Log token usage to database."""
        if not execution_id:
            execution_id = str(uuid.uuid4())

        session = self.Session()
        try:
            try:
                self._log_usage_impl(token_usage_stats, session, execution_id)
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
                self._log_usage(usage_data, execution_id=execution_id)
            return response
        
        return self._wrap_streaming_response(response, execution_id)
    
    def _wrap_streaming_response(self, response_iter: Stream[ChatCompletionChunk], execution_id: Optional[str]) -> Iterator[ChatCompletionChunk]:
        """Wrap streaming response to capture final usage stats"""
        chunks_with_usage = []
        for chunk in response_iter:
            if isinstance(chunk, ChatCompletionChunk) and chunk.usage is not None:
                chunks_with_usage.append(chunk)
            yield chunk
            
        if len(chunks_with_usage) > 0:
            usage_data: TokenUsageStats = TokenUsageStats(model=chunks_with_usage[0].model, usage=Usage())
            for chunk in chunks_with_usage:
                usage_data.usage.prompt_tokens += chunk.usage.prompt_tokens
                usage_data.usage.completion_tokens += chunk.usage.completion_tokens
                usage_data.usage.total_tokens += chunk.usage.total_tokens
            
            self._log_usage(usage_data, execution_id=execution_id)
            

class AsyncOpenAIWrapper(BaseOpenAIWrapper):
    async def create(self, *args: Any, execution_id: Optional[str] = None, **kwargs: Any) -> Union[ChatCompletion, AsyncIterator[ChatCompletion]]:
        """Create a chat completion and log token usage."""
        logger.debug("Creating chat completion with args: %s, kwargs: %s", args, kwargs)
        
        if kwargs.get('stream', False):
            response = await self.client.chat.completions.create(*args, **kwargs)
            return self._wrap_streaming_response(response, execution_id)
        
        response = await self.client.chat.completions.create(*args, **kwargs)
        usage_data = self._process_response_usage(response)
        if usage_data:
            self._log_usage(usage_data, execution_id=execution_id)
        return response

    async def _wrap_streaming_response(self, response_iter: AsyncStream[ChatCompletionChunk], execution_id: Optional[str]) -> AsyncIterator[ChatCompletionChunk]:
        """Wrap streaming response to capture final usage stats"""
        chunks_with_usage = []
        async for chunk in response_iter:
            if isinstance(chunk, ChatCompletionChunk) and chunk.usage is not None:
                chunks_with_usage.append(chunk)
            yield chunk

        if len(chunks_with_usage) > 0:
            usage_data: TokenUsageStats = TokenUsageStats(model=chunks_with_usage[0].model, usage=Usage())
            for chunk in chunks_with_usage:
                usage_data.usage.prompt_tokens += chunk.usage.prompt_tokens
                usage_data.usage.completion_tokens += chunk.usage.completion_tokens
                usage_data.usage.total_tokens += chunk.usage.total_tokens
            
            self._log_usage(usage_data, execution_id=execution_id)

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

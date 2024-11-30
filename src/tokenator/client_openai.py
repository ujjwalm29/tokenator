"""OpenAI client wrapper with token usage tracking."""

from pathlib import Path
from typing import Any, Dict, Optional, Union, overload
import logging

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from .models import get_session, TokenUsage

logger = logging.getLogger(__name__)

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

    def _log_usage_impl(self, model: str, usage: Dict[str, int], session) -> None:
        """Implementation of token usage logging."""
        logger.debug("Logging usage for model %s: %s", model, usage)
        try:
            token_usage = TokenUsage(
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
            raise

    def _process_response_usage(self, response: ChatCompletion) -> Optional[Dict[str, Any]]:
        """Process and log usage statistics from a response."""
        if hasattr(response, "usage"):
            logger.debug("Response usage stats: %s", response.usage)
            return {
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            }
        logger.warning("No usage stats available in response")
        return None

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

class OpenAIWrapper(BaseOpenAIWrapper):
    def _log_usage(self, model: str, usage: Dict[str, int]):
        session = self.Session()
        try:
            self._log_usage_impl(model, usage, session)
            session.commit()
        finally:
            session.close()

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        """Create a chat completion and log token usage."""
        logger.debug("Creating chat completion with args: %s, kwargs: %s", args, kwargs)
        response = self.client.chat.completions.create(*args, **kwargs)
        
        usage_data = self._process_response_usage(response)
        if usage_data:
            self._log_usage(**usage_data)
        
        return response

class AsyncOpenAIWrapper(BaseOpenAIWrapper):
    async def _log_usage(self, model: str, usage: Dict[str, int]):
        session = self.Session()
        try:
            self._log_usage_impl(model, usage, session)
            await session.commit()
        finally:
            await session.close()

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        """Create a chat completion and log token usage."""
        logger.debug("Creating chat completion with args: %s, kwargs: %s", args, kwargs)
        response = await self.client.chat.completions.create(*args, **kwargs)
        
        usage_data = self._process_response_usage(response)
        if usage_data:
            await self._log_usage(**usage_data)
        
        return response

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

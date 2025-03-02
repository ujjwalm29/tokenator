"""Gemini client wrapper with token usage tracking."""

from typing import Any, Optional, Iterator, AsyncIterator
import logging

from google import genai
from google.genai.types import GenerateContentResponse

from ..models import (
    TokenMetrics,
    TokenUsageStats,
)
from ..base_wrapper import BaseWrapper, ResponseType
from .stream_interceptors import (
    GeminiAsyncStreamInterceptor,
    GeminiSyncStreamInterceptor,
)
from ..state import is_tokenator_enabled

logger = logging.getLogger(__name__)


def _create_usage_callback(execution_id, log_usage_fn):
    """Creates a callback function for processing usage statistics from stream chunks."""

    def usage_callback(chunks):
        if not chunks:
            return

        # Skip if tokenator is disabled
        if not is_tokenator_enabled:
            logger.debug("Tokenator is disabled - skipping stream usage logging")
            return

        logger.debug("Processing stream usage for execution_id: %s", execution_id)

        # Build usage_data from the first chunk's model
        usage_data = TokenUsageStats(
            model=chunks[0].model_version,
            usage=TokenMetrics(),
        )

        # Only take usage from the last chunk as it contains complete usage info
        last_chunk = chunks[-1]
        if last_chunk.usage_metadata:
            usage_data.usage.prompt_tokens = (
                last_chunk.usage_metadata.prompt_token_count
            )
            usage_data.usage.completion_tokens = (
                last_chunk.usage_metadata.candidates_token_count or 0
            )
            usage_data.usage.total_tokens = last_chunk.usage_metadata.total_token_count
            log_usage_fn(usage_data, execution_id=execution_id)

    return usage_callback


class BaseGeminiWrapper(BaseWrapper):
    def __init__(self, client, db_path=None, provider: str = "gemini"):
        super().__init__(client, db_path)
        self.provider = provider
        self._async_wrapper = None

    def _process_response_usage(
        self, response: ResponseType
    ) -> Optional[TokenUsageStats]:
        """Process and log usage statistics from a response."""
        try:
            if isinstance(response, GenerateContentResponse):
                if response.usage_metadata is None:
                    return None
                usage = TokenMetrics(
                    prompt_tokens=response.usage_metadata.prompt_token_count,
                    completion_tokens=response.usage_metadata.candidates_token_count,
                    total_tokens=response.usage_metadata.total_token_count,
                )
                return TokenUsageStats(model=response.model_version, usage=usage)

            elif isinstance(response, dict):
                usage_dict = response.get("usage_metadata")
                if not usage_dict:
                    return None
                usage = TokenMetrics(
                    prompt_tokens=usage_dict.get("prompt_token_count", 0),
                    completion_tokens=usage_dict.get("candidates_token_count", 0),
                    total_tokens=usage_dict.get("total_token_count", 0),
                )
                return TokenUsageStats(
                    model=response.get("model", "unknown"), usage=usage
                )
        except Exception as e:
            logger.warning("Failed to process usage stats: %s", str(e))
            return None
        return None

    @property
    def chat(self):
        return self

    @property
    def chats(self):
        return self

    @property
    def models(self):
        return self

    @property
    def aio(self):
        if self._async_wrapper is None:
            self._async_wrapper = AsyncGeminiWrapper(self)
        return self._async_wrapper

    def count_tokens(self, *args: Any, **kwargs: Any):
        return self.client.models.count_tokens(*args, **kwargs)


class AsyncGeminiWrapper:
    """Async wrapper for Gemini client to match the official SDK structure."""

    def __init__(self, wrapper: BaseGeminiWrapper):
        self.wrapper = wrapper
        self._models = None

    @property
    def models(self):
        if self._models is None:
            self._models = AsyncModelsWrapper(self.wrapper)
        return self._models


class AsyncModelsWrapper:
    """Async wrapper for models to match the official SDK structure."""

    def __init__(self, wrapper: BaseGeminiWrapper):
        self.wrapper = wrapper

    async def generate_content(
        self, *args: Any, **kwargs: Any
    ) -> GenerateContentResponse:
        """Async method for generate_content."""
        execution_id = kwargs.pop("execution_id", None)
        return await self.wrapper.generate_content_async(
            *args, execution_id=execution_id, **kwargs
        )

    async def generate_content_stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[GenerateContentResponse]:
        """Async method for generate_content_stream."""
        execution_id = kwargs.pop("execution_id", None)
        return await self.wrapper.generate_content_stream_async(
            *args, execution_id=execution_id, **kwargs
        )


class GeminiWrapper(BaseGeminiWrapper):
    def generate_content(
        self, *args: Any, execution_id: Optional[str] = None, **kwargs: Any
    ) -> GenerateContentResponse:
        """Generate content and log token usage."""
        logger.debug("Generating content with args: %s, kwargs: %s", args, kwargs)

        response = self.client.models.generate_content(*args, **kwargs)
        usage_data = self._process_response_usage(response)
        if usage_data:
            self._log_usage(usage_data, execution_id=execution_id)

        return response

    def generate_content_stream(
        self, *args: Any, execution_id: Optional[str] = None, **kwargs: Any
    ) -> Iterator[GenerateContentResponse]:
        """Generate content with streaming and log token usage."""
        logger.debug(
            "Generating content stream with args: %s, kwargs: %s", args, kwargs
        )

        base_stream = self.client.models.generate_content_stream(*args, **kwargs)
        return GeminiSyncStreamInterceptor(
            base_stream=base_stream,
            usage_callback=_create_usage_callback(execution_id, self._log_usage),
        )

    async def generate_content_async(
        self, *args: Any, execution_id: Optional[str] = None, **kwargs: Any
    ) -> GenerateContentResponse:
        """Generate content asynchronously and log token usage."""
        logger.debug("Generating content async with args: %s, kwargs: %s", args, kwargs)

        response = await self.client.aio.models.generate_content(*args, **kwargs)
        usage_data = self._process_response_usage(response)
        if usage_data:
            self._log_usage(usage_data, execution_id=execution_id)

        return response

    async def generate_content_stream_async(
        self, *args: Any, execution_id: Optional[str] = None, **kwargs: Any
    ) -> AsyncIterator[GenerateContentResponse]:
        """Generate content with async streaming and log token usage."""
        logger.debug(
            "Generating content stream async with args: %s, kwargs: %s", args, kwargs
        )

        base_stream = await self.client.aio.models.generate_content_stream(
            *args, **kwargs
        )
        return GeminiAsyncStreamInterceptor(
            base_stream=base_stream,
            usage_callback=_create_usage_callback(execution_id, self._log_usage),
        )


def tokenator_gemini(
    client: genai.Client,
    db_path: Optional[str] = None,
    provider: str = "gemini",
) -> GeminiWrapper:
    """Create a token-tracking wrapper for a Gemini client.

    Args:
        client: Gemini client instance
        db_path: Optional path to SQLite database for token tracking
        provider: Provider name, defaults to "gemini"
    """
    if not isinstance(client, genai.Client):
        raise ValueError("Client must be an instance of genai.Client")

    return GeminiWrapper(client=client, db_path=db_path, provider=provider)

"""Anthropic client wrapper with token usage tracking."""

from typing import Any, Optional, Union, overload, Iterator, AsyncIterator
import logging

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, RawMessageStartEvent, RawMessageDeltaEvent

from ..models import Usage, TokenUsageStats
from ..base_wrapper import BaseWrapper, ResponseType
from .stream_interceptors import AnthropicAsyncStreamInterceptor, AnthropicSyncStreamInterceptor

logger = logging.getLogger(__name__)


class BaseAnthropicWrapper(BaseWrapper):
    provider = "anthropic"

    def _process_response_usage(
        self, response: ResponseType
    ) -> Optional[TokenUsageStats]:
        """Process and log usage statistics from a response."""
        try:
            if isinstance(response, Message):
                if not hasattr(response, "usage"):
                    return None
                usage = Usage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens
                    + response.usage.output_tokens,
                )
                return TokenUsageStats(model=response.model, usage=usage)
            elif isinstance(response, dict):
                usage_dict = response.get("usage")
                if not usage_dict:
                    return None
                usage = Usage(
                    prompt_tokens=usage_dict.get("input_tokens", 0),
                    completion_tokens=usage_dict.get("output_tokens", 0),
                    total_tokens=usage_dict.get("input_tokens", 0)
                    + usage_dict.get("output_tokens", 0),
                )
                return TokenUsageStats(
                    model=response.get("model", "unknown"), usage=usage
                )
        except Exception as e:
            logger.warning("Failed to process usage stats: %s", str(e))
            return None
        return None

    @property
    def messages(self):
        return self


def _create_usage_callback(execution_id, log_usage_fn):
    """Creates a callback function for processing usage statistics from stream chunks."""
    def usage_callback(chunks):
        if not chunks:
            return
        
        usage_data = TokenUsageStats(
            model=chunks[0].message.model if isinstance(chunks[0], RawMessageStartEvent) else "",
            usage=Usage(),
        )
        
        for chunk in chunks:
            if isinstance(chunk, RawMessageStartEvent):
                usage_data.model = chunk.message.model
                usage_data.usage.prompt_tokens += chunk.message.usage.input_tokens
                usage_data.usage.completion_tokens += chunk.message.usage.output_tokens
            elif isinstance(chunk, RawMessageDeltaEvent):
                usage_data.usage.prompt_tokens += chunk.usage.input_tokens
                usage_data.usage.completion_tokens += chunk.usage.output_tokens
        
        usage_data.usage.total_tokens = usage_data.usage.prompt_tokens + usage_data.usage.completion_tokens
        log_usage_fn(usage_data, execution_id=execution_id)

    return usage_callback


class AnthropicWrapper(BaseAnthropicWrapper):
    def create(
        self, *args: Any, execution_id: Optional[str] = None, **kwargs: Any
    ) -> Union[Message, Iterator[Message]]:
        """Create a message completion and log token usage."""
        logger.debug("Creating message completion with args: %s, kwargs: %s", args, kwargs)

        if kwargs.get("stream", False):
            base_stream = self.client.messages.create(*args, **kwargs)
            return AnthropicSyncStreamInterceptor(
                base_stream=base_stream,
                usage_callback=_create_usage_callback(execution_id, self._log_usage),
            )

        response = self.client.messages.create(*args, **kwargs)
        usage_data = self._process_response_usage(response)
        if usage_data:
            self._log_usage(usage_data, execution_id=execution_id)
        return response


class AsyncAnthropicWrapper(BaseAnthropicWrapper):
    async def create(
        self, *args: Any, execution_id: Optional[str] = None, **kwargs: Any
    ) -> Union[Message, AsyncIterator[Message]]:
        """Create a message completion and log token usage."""
        logger.debug("Creating message completion with args: %s, kwargs: %s", args, kwargs)

        if kwargs.get("stream", False):
            base_stream = await self.client.messages.create(*args, **kwargs)
            return AnthropicAsyncStreamInterceptor(
                base_stream=base_stream,
                usage_callback=_create_usage_callback(execution_id, self._log_usage),
            )

        response = await self.client.messages.create(*args, **kwargs)
        usage_data = self._process_response_usage(response)
        if usage_data:
            self._log_usage(usage_data, execution_id=execution_id)
        return response


@overload
def tokenator_anthropic(
    client: Anthropic,
    db_path: Optional[str] = None,
) -> AnthropicWrapper: ...


@overload
def tokenator_anthropic(
    client: AsyncAnthropic,
    db_path: Optional[str] = None,
) -> AsyncAnthropicWrapper: ...


def tokenator_anthropic(
    client: Union[Anthropic, AsyncAnthropic],
    db_path: Optional[str] = None,
) -> Union[AnthropicWrapper, AsyncAnthropicWrapper]:
    """Create a token-tracking wrapper for an Anthropic client.

    Args:
        client: Anthropic or AsyncAnthropic client instance
        db_path: Optional path to SQLite database for token tracking
    """
    if isinstance(client, Anthropic):
        return AnthropicWrapper(client=client, db_path=db_path)

    if isinstance(client, AsyncAnthropic):
        return AsyncAnthropicWrapper(client=client, db_path=db_path)

    raise ValueError("Client must be an instance of Anthropic or AsyncAnthropic")

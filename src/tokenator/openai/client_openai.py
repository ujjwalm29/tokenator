"""OpenAI client wrapper with token usage tracking."""

from typing import Any, Optional, Union, overload, Iterator, AsyncIterator
import logging

from openai import AsyncOpenAI, OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from ..models import Usage, TokenUsageStats
from ..base_wrapper import BaseWrapper, ResponseType
from .AsyncStreamInterceptor import AsyncStreamInterceptor

logger = logging.getLogger(__name__)


class BaseOpenAIWrapper(BaseWrapper):
    provider = "openai"

    def _process_response_usage(
        self, response: ResponseType
    ) -> Optional[TokenUsageStats]:
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
                usage_dict = response.get("usage")
                if not usage_dict:
                    return None
                usage = Usage(
                    prompt_tokens=usage_dict.get("prompt_tokens", 0),
                    completion_tokens=usage_dict.get("completion_tokens", 0),
                    total_tokens=usage_dict.get("total_tokens", 0),
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
    def completions(self):
        return self


class OpenAIWrapper(BaseOpenAIWrapper):
    def create(
        self, *args: Any, execution_id: Optional[str] = None, **kwargs: Any
    ) -> Union[ChatCompletion, Iterator[ChatCompletion]]:
        """Create a chat completion and log token usage."""
        logger.debug("Creating chat completion with args: %s, kwargs: %s", args, kwargs)

        response = self.client.chat.completions.create(*args, **kwargs)

        if not kwargs.get("stream", False):
            usage_data = self._process_response_usage(response)
            if usage_data:
                self._log_usage(usage_data, execution_id=execution_id)
            return response

        return self._wrap_streaming_response(response, execution_id)

    def _wrap_streaming_response(
        self, response_iter: Stream[ChatCompletionChunk], execution_id: Optional[str]
    ) -> Iterator[ChatCompletionChunk]:
        """Wrap streaming response to capture final usage stats"""
        chunks_with_usage = []
        for chunk in response_iter:
            if isinstance(chunk, ChatCompletionChunk) and chunk.usage is not None:
                chunks_with_usage.append(chunk)
            yield chunk

        if len(chunks_with_usage) > 0:
            usage_data: TokenUsageStats = TokenUsageStats(
                model=chunks_with_usage[0].model, usage=Usage()
            )
            for chunk in chunks_with_usage:
                usage_data.usage.prompt_tokens += chunk.usage.prompt_tokens
                usage_data.usage.completion_tokens += chunk.usage.completion_tokens
                usage_data.usage.total_tokens += chunk.usage.total_tokens

            self._log_usage(usage_data, execution_id=execution_id)


class AsyncOpenAIWrapper(BaseOpenAIWrapper):
    async def create(
        self, *args: Any, execution_id: Optional[str] = None, **kwargs: Any
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """
        Create a chat completion and log token usage.
        """
        logger.debug("Creating chat completion with args: %s, kwargs: %s", args, kwargs)

        # If user wants a stream, return an interceptor
        if kwargs.get("stream", False):
            base_stream = await self.client.chat.completions.create(*args, **kwargs)

            # Define a callback that will get called once the stream ends
            def usage_callback(chunks):
                # Mimic your old logic to gather usage from chunk.usage
                # e.g. ChatCompletionChunk.usage
                # Then call self._log_usage(...)
                if not chunks:
                    return
                # Build usage_data from the first chunk's model
                usage_data = TokenUsageStats(
                    model=chunks[0].model,
                    usage=Usage(),
                )
                # Sum up usage from all chunks
                for ch in chunks:
                    if ch.usage:
                        usage_data.usage.prompt_tokens += ch.usage.prompt_tokens
                        usage_data.usage.completion_tokens += ch.usage.completion_tokens
                        usage_data.usage.total_tokens += ch.usage.total_tokens

                self._log_usage(usage_data, execution_id=execution_id)

            # Return the interceptor that wraps the real AsyncStream
            return AsyncStreamInterceptor(
                base_stream=base_stream,
                usage_callback=usage_callback,
            )

        # Non-streaming path remains unchanged
        response = await self.client.chat.completions.create(*args, **kwargs)
        usage_data = self._process_response_usage(response)
        if usage_data:
            self._log_usage(usage_data, execution_id=execution_id)
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

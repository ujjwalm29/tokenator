import logging
from typing import AsyncIterator, Callable, Generic, List, Optional, TypeVar

from openai import AsyncStream, AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

logger = logging.getLogger(__name__)

_T = TypeVar("_T")  # or you might specifically do _T = ChatCompletionChunk


class AsyncStreamInterceptor(AsyncStream[_T]):
    """
    A wrapper around openai.AsyncStream that delegates all functionality
    to the 'base_stream' but intercepts each chunk to handle usage or
    logging logic. This preserves .response and other methods.

    You can store aggregated usage in a local list and process it when
    the stream ends (StopAsyncIteration).
    """

    def __init__(
        self,
        base_stream: AsyncStream[_T],
        usage_callback: Optional[Callable[[List[_T]], None]] = None,
    ):
        # We do NOT call super().__init__() because openai.AsyncStream
        # expects constructor parameters we don't want to re-initialize.
        # Instead, we just store the base_stream and delegate everything to it.
        self._base_stream = base_stream
        self._usage_callback = usage_callback
        self._chunks: List[_T] = []

    @property
    def response(self):
        """Expose the original stream's 'response' so user code can do stream.response, etc."""
        return self._base_stream.response

    def __aiter__(self) -> AsyncIterator[_T]:
        """
        Called when we do 'async for chunk in wrapped_stream:'
        We simply return 'self'. Then __anext__ does the rest.
        """
        return self

    async def __anext__(self) -> _T:
        """
        Intercept iteration. We pull the next chunk from the base_stream.
        If it's the end, do any final usage logging, then raise StopAsyncIteration.
        Otherwise, we can accumulate usage info or do whatever we need with the chunk.
        """
        try:
            chunk = await self._base_stream.__anext__()
        except StopAsyncIteration:
            # Once the base stream is fully consumed, we can do final usage/logging.
            if self._usage_callback and self._chunks:
                self._usage_callback(self._chunks)
            raise

        # Intercept each chunk
        self._chunks.append(chunk)
        return chunk

    async def __aenter__(self) -> "AsyncStreamInterceptor[_T]":
        """Support async with ... : usage."""
        await self._base_stream.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Ensure we propagate __aexit__ to the base stream,
        so connections are properly closed.
        """
        return await self._base_stream.__aexit__(exc_type, exc_val, exc_tb)

    async def close(self) -> None:
        """Delegate close to the base_stream."""
        await self._base_stream.close()
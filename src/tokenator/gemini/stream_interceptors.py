"""Stream interceptors for Gemini responses."""

import logging
from typing import AsyncIterator, Callable, List, Optional, TypeVar, Iterator

from google.genai.types import GenerateContentResponse

logger = logging.getLogger(__name__)

_T = TypeVar("_T")  # GenerateContentResponse


class GeminiAsyncStreamInterceptor(AsyncIterator[_T]):
    """
    A wrapper around Gemini async stream that intercepts each chunk to handle usage or
    logging logic.
    """

    def __init__(
        self,
        base_stream: AsyncIterator[_T],
        usage_callback: Optional[Callable[[List[_T]], None]] = None,
    ):
        self._base_stream = base_stream
        self._usage_callback = usage_callback
        self._chunks: List[_T] = []

    def __aiter__(self) -> AsyncIterator[_T]:
        """Return self as async iterator."""
        return self

    async def __anext__(self) -> _T:
        """Get next chunk and track it."""
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


class GeminiSyncStreamInterceptor(Iterator[_T]):
    """
    A wrapper around Gemini sync stream that intercepts each chunk to handle usage or
    logging logic.
    """

    def __init__(
        self,
        base_stream: Iterator[_T],
        usage_callback: Optional[Callable[[List[_T]], None]] = None,
    ):
        self._base_stream = base_stream
        self._usage_callback = usage_callback
        self._chunks: List[_T] = []

    def __iter__(self) -> Iterator[_T]:
        """Return self as iterator."""
        return self

    def __next__(self) -> _T:
        """Get next chunk and track it."""
        try:
            chunk = next(self._base_stream)
        except StopIteration:
            # Once the base stream is fully consumed, we can do final usage/logging.
            if self._usage_callback and self._chunks:
                self._usage_callback(self._chunks)
            raise

        # Intercept each chunk
        self._chunks.append(chunk)
        return chunk 
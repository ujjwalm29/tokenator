"""Base wrapper class for token usage tracking."""

from pathlib import Path
from typing import Any, Optional, TypeVar
import logging
import uuid

from .models import TokenUsageStats
from .schemas import get_session, TokenUsage

logger = logging.getLogger(__name__)

ResponseType = TypeVar("ResponseType")


class BaseWrapper:
    def __init__(self, client: Any, db_path: Optional[str] = None):
        """Initialize the base wrapper."""
        self.client = client

        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info("Created database directory at: %s", Path(db_path).parent)

        self.Session = get_session(db_path)

        logger.debug(
            "Initializing %s with db_path: %s", self.__class__.__name__, db_path
        )

    def _log_usage_impl(
        self, token_usage_stats: TokenUsageStats, session, execution_id: str
    ) -> None:
        """Implementation of token usage logging."""
        logger.debug(
            "Logging usage for model %s: %s",
            token_usage_stats.model,
            token_usage_stats.usage.model_dump(),
        )
        try:
            token_usage = TokenUsage(
                execution_id=execution_id,
                provider=self.provider,
                model=token_usage_stats.model,
                prompt_tokens=token_usage_stats.usage.prompt_tokens,
                completion_tokens=token_usage_stats.usage.completion_tokens,
                total_tokens=token_usage_stats.usage.total_tokens,
            )
            session.add(token_usage)
            logger.info(
                "Logged token usage: model=%s, total_tokens=%d",
                token_usage_stats.model,
                token_usage_stats.usage.total_tokens,
            )
        except Exception as e:
            logger.error("Failed to log token usage: %s", str(e))

    def _log_usage(
        self, token_usage_stats: TokenUsageStats, execution_id: Optional[str] = None
    ):
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

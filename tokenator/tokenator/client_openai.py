"""OpenAI client wrapper with token usage tracking."""

from pathlib import Path
from typing import Any, Dict, Optional
import logging

from openai import OpenAI
from openai.types.chat import ChatCompletion

from .models import get_session, TokenUsage

logger = logging.getLogger(__name__)

class OpenAIWrapper:
    def __init__(self, client: OpenAI, db_path: Optional[str] = None):
        """Initialize the OpenAI wrapper."""
        self.client = client
        self.Session = get_session(db_path)
        
        logger.debug("Initializing OpenAIWrapper with db_path: %s", db_path)
        
        # Ensure database directory exists
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info("Created database directory at: %s", Path(db_path).parent)

    def _log_usage(self, model: str, usage: Dict[str, int]):
        """Log token usage to database using SQLAlchemy."""
        logger.debug("Logging usage for model %s: %s", model, usage)
        session = self.Session()
        try:
            token_usage = TokenUsage(
                provider="openai",
                model=model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0)
            )
            session.add(token_usage)
            session.commit()
            logger.info("Logged token usage: model=%s, total_tokens=%d", 
                       model, usage.get("total_tokens", 0))
        except Exception as e:
            logger.error("Failed to log token usage: %s", str(e))
            raise
        finally:
            session.close()

    @property
    def chat(self):
        return ChatCompletionWrapper(self)

class ChatCompletionWrapper:
    def __init__(self, wrapper: OpenAIWrapper):
        self.wrapper = wrapper
        self.completions = self
        logger.debug("Initialized ChatCompletionWrapper")

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        """Create a chat completion and log token usage."""
        logger.debug("Creating chat completion with args: %s, kwargs: %s", args, kwargs)
        response = self.wrapper.client.chat.completions.create(*args, **kwargs)
        
        if hasattr(response, "usage"):
            logger.debug("Response usage stats: %s", response.usage)
            self.wrapper._log_usage(
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )
        else:
            logger.warning("No usage stats available in response")
        
        return response

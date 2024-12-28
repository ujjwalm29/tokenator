"""Tokenator - Track and analyze your OpenAI API token usage and costs."""

import logging
from .openai.client_openai import tokenator_openai
from .anthropic.client_anthropic import tokenator_anthropic
from . import usage
from .utils import get_default_db_path
from .migrations import check_and_run_migrations

__version__ = "0.1.0"
__all__ = ["tokenator_openai", "tokenator_anthropic", "usage", "get_default_db_path"]

logger = logging.getLogger(__name__)

try:
    check_and_run_migrations()
except Exception as e:
    logger.warning(f"Failed to run migrations, but continuing anyway: {e}")

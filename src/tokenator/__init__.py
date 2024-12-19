"""Tokenator - Track and analyze your OpenAI API token usage and costs."""

from .client_openai import OpenAIWrapper
from . import usage
from .utils import get_default_db_path
from .migrations import check_and_run_migrations

__version__ = "0.1.0"
__all__ = ["OpenAIWrapper", "usage", "get_default_db_path"]

# Run migrations on import
check_and_run_migrations() 
"""Shared utility functions for tokenator."""

import os
import platform
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        from importlib.util import find_spec

        return find_spec("google.colab") is not None
    except ImportError:
        return False


def get_default_db_path() -> str:
    """Get the platform-specific default database path."""
    try:
        if is_colab():
            # Use in-memory database for Colab
            return "usage.db"

        system = platform.system().lower()

        if system == "linux" or system == "darwin":
            # Follow XDG Base Directory Specification
            xdg_data_home = os.environ.get("XDG_DATA_HOME", "")
            if not xdg_data_home:
                xdg_data_home = os.path.join(str(Path.home()), ".local", "share")
            db_path = os.path.join(xdg_data_home, "tokenator", "usage.db")
        elif system == "windows":
            # Use %LOCALAPPDATA% on Windows
            local_app_data = os.environ.get("LOCALAPPDATA", "")
            if not local_app_data:
                local_app_data = os.path.join(str(Path.home()), "AppData", "Local")
            db_path = os.path.join(local_app_data, "tokenator", "usage.db")
        else:
            db_path = os.path.join(str(Path.home()), ".tokenator", "usage.db")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return db_path
    except (OSError, IOError) as e:
        # Fallback to current directory if we can't create the default path
        fallback_path = os.path.join(os.getcwd(), "tokenator_usage.db")
        logger.warning(
            f"Could not create default db path, falling back to {fallback_path}. Error: {e}"
        )
        return fallback_path


__all__ = [
    "get_default_db_path",
    "is_colab",
    # ... other exports ...
]

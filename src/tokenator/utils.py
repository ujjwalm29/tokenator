"""Shared utility functions for tokenator."""

import os
import platform
from pathlib import Path
from typing import Optional

def get_default_db_path() -> str:
    """Get the platform-specific default database path.
    
    Returns:
        str: Path to the SQLite database file
        
    The path follows platform conventions:
        - Linux/macOS: ~/.local/share/tokenator/usage.db (XDG spec)
        - Windows: %LOCALAPPDATA%\\tokenator\\usage.db
        - Others: ~/.tokenator/usage.db
    """
    system = platform.system().lower()
    
    if system == "linux" or system == "darwin":
        # Follow XDG Base Directory Specification
        xdg_data_home = os.environ.get("XDG_DATA_HOME", "")
        if not xdg_data_home:
            xdg_data_home = os.path.join(str(Path.home()), ".local", "share")
        return os.path.join(xdg_data_home, "tokenator", "usage.db")
    elif system == "windows":
        # Use %LOCALAPPDATA% on Windows
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if not local_app_data:
            local_app_data = os.path.join(str(Path.home()), "AppData", "Local")
        return os.path.join(local_app_data, "tokenator", "usage.db")
    else:
        # Fallback for other systems
        return os.path.join(str(Path.home()), ".tokenator", "usage.db") 
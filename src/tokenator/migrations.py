"""Database migration utilities for tokenator."""

import os
from pathlib import Path
from alembic.config import Config
from alembic import command
from .utils import get_default_db_path


def get_alembic_config(db_path: str = None) -> Config:
    """Get Alembic config for migrations."""
    if db_path is None:
        db_path = get_default_db_path()

    # Get the directory containing this file
    migrations_dir = Path(__file__).parent / "migrations"

    # Create Config object
    config = Config()
    config.set_main_option("script_location", str(migrations_dir))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")

    return config


def check_and_run_migrations(db_path: str = None):
    """Check and run any pending database migrations."""
    if db_path is None:
        db_path = get_default_db_path()

    dirname = os.path.dirname(db_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Initialize database
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.close()

    config = get_alembic_config(db_path)
    command.upgrade(config, "head")

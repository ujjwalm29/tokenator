"""Automatic database migrations manager."""

import os
from pathlib import Path
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

from .utils import get_default_db_path

def get_alembic_config():
    """Get Alembic config pointing to the package's migrations."""
    package_dir = Path(__file__).parent
    migrations_dir = package_dir / "migrations"
    
    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", str(migrations_dir))
    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{get_default_db_path()}")
    
    return alembic_cfg

def check_and_run_migrations():
    """Check if migrations are needed and run them automatically."""
    engine = create_engine(f"sqlite:///{get_default_db_path()}")
    
    # Create migrations table if it doesn't exist
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current_rev = context.get_current_revision()
    
    # Get latest available revision
    config = get_alembic_config()
    script = ScriptDirectory.from_config(config)
    head_rev = script.get_current_head()
    
    # Run migrations if needed
    if current_rev != head_rev:
        command.upgrade(config, "head") 
"""Development utilities for tokenator."""

import os
from pathlib import Path
from alembic import command
from tokenator.migrations import get_alembic_config

def create_migration():
    """Create a new migration based on model changes."""
    config = get_alembic_config()
    
    # Get the migrations directory
    migrations_dir = Path(__file__).parent / "migrations" / "versions"
    migrations_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate migration
    command.revision(config, autogenerate=True, message="auto generated migration") 


if __name__ == "__main__":
    create_migration()
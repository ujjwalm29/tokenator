"""Development utilities for tokenator."""

import sys
from pathlib import Path
from alembic import command
from tokenator.migrations import get_alembic_config


def create_migration(message: str):
    """Create a new migration based on model changes."""
    config = get_alembic_config()

    # Get the migrations directory
    migrations_dir = Path(__file__).parent / "migrations" / "versions"
    migrations_dir.mkdir(parents=True, exist_ok=True)

    # Generate migration with custom message
    command.revision(config, autogenerate=True, message=message)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
    else:
        msg = "auto generated migration"
    create_migration(msg)

import pytest
import os
import sys
import sqlite3
from tokenator.migrations import check_and_run_migrations, get_alembic_config
from alembic.script import ScriptDirectory


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database file."""
    db_file = tmp_path / "test.db"
    return str(db_file)


@pytest.fixture
def mock_colab(monkeypatch):
    """Mock Colab environment."""

    class MockColab:
        pass

    monkeypatch.setattr("sys.modules", {"google.colab": MockColab(), **sys.modules})
    return MockColab()


@pytest.fixture
def colab_db_path():
    """Fixture for Colab database path."""
    return "usage.db"


def verify_migrations(db_path):
    """Helper to verify migrations ran successfully."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if alembic_version table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'"
    )
    assert cursor.fetchone() is not None

    # Verify we're at the latest migration
    config = get_alembic_config(db_path)
    script = ScriptDirectory.from_config(config)
    head_revision = script.get_current_head()

    cursor.execute("SELECT version_num FROM alembic_version")
    current_revision = cursor.fetchone()[0]
    assert current_revision == head_revision

    conn.close()


def test_check_and_run_migrations(temp_db, monkeypatch):
    """Test that migrations run correctly on a new database."""
    monkeypatch.setattr("tokenator.utils.get_default_db_path", lambda: temp_db)

    check_and_run_migrations(temp_db)

    # For file-based DB, verify file exists
    assert os.path.exists(temp_db)
    verify_migrations(temp_db)


def test_check_and_run_migrations_colab(mock_colab, colab_db_path, monkeypatch):
    """Test migrations in Colab environment with in-memory database."""
    monkeypatch.setattr("tokenator.utils.get_default_db_path", lambda: colab_db_path)

    check_and_run_migrations(colab_db_path)

    # For in-memory DB, just verify migrations
    verify_migrations(colab_db_path)


def test_migrations_idempotent(temp_db, monkeypatch):
    """Test that running migrations multiple times is safe."""
    monkeypatch.setattr("tokenator.utils.get_default_db_path", lambda: temp_db)

    # Run migrations twice
    check_and_run_migrations(temp_db)
    check_and_run_migrations(temp_db)

    verify_migrations(temp_db)


def test_migrations_idempotent_colab(mock_colab, colab_db_path, monkeypatch):
    """Test migrations idempotency in Colab environment."""
    monkeypatch.setattr("tokenator.utils.get_default_db_path", lambda: colab_db_path)

    check_and_run_migrations(colab_db_path)
    check_and_run_migrations(colab_db_path)

    verify_migrations(colab_db_path)


def test_migration_with_existing_data(temp_db, monkeypatch):
    """Test migrations work with existing data in the database."""
    monkeypatch.setattr("tokenator.utils.get_default_db_path", lambda: temp_db)

    # Create a database with some data but no migrations
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_records (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL
        )
    """)
    cursor.execute("INSERT INTO usage_records (timestamp) VALUES ('2024-01-01')")
    conn.commit()
    conn.close()

    check_and_run_migrations(temp_db)

    # Verify original data still exists
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp FROM usage_records")
    assert cursor.fetchone()[0] == "2024-01-01"
    conn.close()


def test_migration_with_existing_data_colab(mock_colab, colab_db_path, monkeypatch):
    """Test migrations with existing data in Colab environment."""
    monkeypatch.setattr("tokenator.utils.get_default_db_path", lambda: colab_db_path)

    # Create test data in memory
    conn = sqlite3.connect(colab_db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_records (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL
        )
    """)
    cursor.execute("INSERT INTO usage_records (timestamp) VALUES ('2024-01-01')")
    conn.commit()

    check_and_run_migrations(colab_db_path)

    # Verify data persists after migration
    cursor.execute("SELECT timestamp FROM usage_records")
    assert cursor.fetchone()[0] == "2024-01-01"
    conn.close()


"""
TODO: Implement real database integration tests

Test cases to cover:
1. PostgreSQL integration
2. MySQL integration 
3. SQLite with different paths
4. Connection error handling
5. Transaction rollbacks
6. Concurrent access

Example structure:
@pytest.mark.integration
@pytest.mark.parametrize("db_url", [
    "postgresql://user:pass@localhost/test",
    "mysql://user:pass@localhost/test",
    "sqlite:///test.db"
])
def test_db_integration(db_url):
    pass
"""

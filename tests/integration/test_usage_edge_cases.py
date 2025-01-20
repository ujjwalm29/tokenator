from unittest.mock import patch
import pytest
from datetime import datetime, timedelta
import tempfile
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import threading
from concurrent.futures import ThreadPoolExecutor

from tokenator.schemas import Base, TokenUsage
from tokenator import usage, state


@pytest.fixture
def temp_db():
    """Create a temporary test database"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_tokens.db")
        db_url = f"sqlite:///{db_path}"

        engine = create_engine(db_url)
        Base.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        state.db_path = db_path
        with patch("tokenator.schemas.get_session", return_value=Session):
            yield Session

        Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(temp_db):
    """Get a database session"""
    session = temp_db()
    try:
        yield session
    finally:
        session.close()


def test_query_performance(db_session):
    """Test query performance with large dataset"""
    # Insert 1000 records
    base_time = datetime.now()
    bulk_data = []
    for i in range(1000):
        bulk_data.append(
            TokenUsage(
                execution_id=f"perf-test-{i}",
                provider="openai",
                model="gpt-4",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                total_cost=0,
                created_at=base_time - timedelta(minutes=i),
            )
        )

    db_session.bulk_save_objects(bulk_data)
    db_session.commit()

    # Measure query time
    start_time = datetime.now()
    result = usage.last_day()
    query_time = (datetime.now() - start_time).total_seconds()

    assert query_time < 1.0  # Query should complete in under 1 second
    assert result.total_tokens > 0


def test_concurrent_writes(temp_db):
    """Test concurrent write operations"""

    def write_records():
        session = temp_db()
        try:
            for i in range(10):
                usage = TokenUsage(
                    execution_id=f"concurrent-{threading.get_ident()}-{i}",
                    provider="openai",
                    model="gpt-4",
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    total_cost=0,
                    created_at=datetime.now(),
                )
                session.add(usage)
            session.commit()
        finally:
            session.close()

    # Run concurrent write operations
    with ThreadPoolExecutor(max_workers=4) as executor:
        [executor.submit(write_records) for _ in range(4)]

    # Verify all records were written
    session = temp_db()
    try:
        record_count = session.query(TokenUsage).count()
        assert record_count == 40  # 4 threads * 10 records each
    finally:
        session.close()


def test_time_boundary_precision(db_session):
    """Test precise time boundary handling in queries"""
    now = datetime.now().replace(microsecond=0)

    # Create records at exact time boundaries
    records = [
        # Exactly 30 minutes ago
        TokenUsage(
            execution_id="boundary-1",
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            total_cost=0,
            created_at=now - timedelta(minutes=30),
        ),
        # 1 second before hour boundary
        TokenUsage(
            execution_id="boundary-2",
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            total_cost=0,
            created_at=now - timedelta(hours=1, seconds=-1),
        ),
        # 3 second after hour boundary
        TokenUsage(
            execution_id="boundary-3",
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            total_cost=0,
            created_at=now - timedelta(hours=1, seconds=3),
        ),
    ]

    for record in records:
        db_session.add(record)
    db_session.commit()

    result = usage.last_hour()
    assert result.total_tokens == 300  # Should include exactly 2 records


def test_transaction_rollback(db_session):
    """Test transaction rollback behavior"""
    try:
        # Start a transaction
        usage = TokenUsage(
            execution_id="rollback-test",
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            total_cost=0,
            created_at=datetime.now(),
        )
        db_session.add(usage)

        # Force an error
        raise Exception("Simulated error")

    except Exception:
        db_session.rollback()

    # Verify record wasn't saved
    count = db_session.query(TokenUsage).filter_by(execution_id="rollback-test").count()
    assert count == 0


def test_index_usage(db_session):
    """Test that queries use indexes effectively"""
    # Create test data
    base_time = datetime.now()
    for i in range(100):
        usage = TokenUsage(
            execution_id=f"index-test-{i}",
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            total_cost=0,
            created_at=base_time - timedelta(days=i),
        )
        db_session.add(usage)
    db_session.commit()

    # Query with EXPLAIN
    from sqlalchemy import text

    explain_query = text("""
        EXPLAIN QUERY PLAN
        SELECT * FROM token_usage 
        WHERE created_at BETWEEN :start AND :end
    """)

    result = db_session.execute(
        explain_query, {"start": base_time - timedelta(days=7), "end": base_time}
    )

    plan = ""
    for row in result:
        print(f"row: {row}")
        plan += str(row)

    # Verify index usage
    assert "USING INDEX" in plan or "SEARCH" in plan


def test_data_persistence(temp_db):
    """Test that data persists across session closures"""
    # Insert data
    session1 = temp_db()
    try:
        usage = TokenUsage(
            execution_id="persistence-test",
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            total_cost=0,
            created_at=datetime.now(),
        )
        session1.add(usage)
        session1.commit()
    finally:
        session1.close()

    # Verify in new session
    session2 = temp_db()
    try:
        record = (
            session2.query(TokenUsage)
            .filter_by(execution_id="persistence-test")
            .first()
        )
        assert record is not None
        assert record.total_tokens == 150
    finally:
        session2.close()

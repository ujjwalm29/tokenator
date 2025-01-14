from datetime import datetime, timedelta
import pytest
from unittest.mock import patch
import tempfile
import os
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from tokenator import usage
from tokenator.models import TokenUsageReport, TokenRate
from tokenator.schemas import TokenUsage, Base
from tokenator import state

MOCK_MODEL_COSTS = {
    "gpt-4": TokenRate(prompt=0.03, completion=0.06),
    "gpt-4o": TokenRate(prompt=0.001, completion=0.002),
    "claude-3-5-haiku": TokenRate(prompt=0.004, completion=0.008),
    "claude-3-5-sonnet": TokenRate(prompt=0.005, completion=0.01),
}


@pytest.fixture
def base_time():
    return datetime.now().replace(microsecond=0)


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
def usage_data(temp_db, base_time):
    """Create actual token usage records in the test database"""
    session = temp_db()
    records = [
        TokenUsage(
            execution_id="exec-recent-1",
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            created_at=base_time - timedelta(minutes=20),
        ),
        TokenUsage(
            execution_id="exec-recent-2",
            provider="anthropic",
            model="claude-3-5-haiku",
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            created_at=base_time - timedelta(minutes=45),
        ),
        # Data from a few hours ago
        TokenUsage(
            execution_id="exec-today-1",
            provider="openai",
            model="gpt-4",
            prompt_tokens=150,
            completion_tokens=75,
            total_tokens=225,
            created_at=base_time - timedelta(hours=4),
        ),
        # Yesterday's data
        TokenUsage(
            execution_id="exec-yesterday-1",
            provider="anthropic",
            model="claude-3-5-haiku",
            prompt_tokens=300,
            completion_tokens=150,
            total_tokens=450,
            created_at=base_time - timedelta(days=2, hours=2),
        ),
        # This week's data
        TokenUsage(
            execution_id="exec-lastweek-1",
            provider="openai",
            model="gpt-4",
            prompt_tokens=400,
            completion_tokens=200,
            total_tokens=600,
            created_at=base_time - timedelta(days=6),
        ),
        # Last week's data
        TokenUsage(
            execution_id="exec-lastweek-2",
            provider="openai",
            model="gpt-4",
            prompt_tokens=400,
            completion_tokens=200,
            total_tokens=600,
            created_at=base_time - timedelta(days=13),
        ),
        # Last month's data
        TokenUsage(
            execution_id="exec-lastmonth-1",
            provider="anthropic",
            model="claude-3-5-sonnet",
            prompt_tokens=500,
            completion_tokens=250,
            total_tokens=750,
            created_at=base_time - timedelta(days=25),
        ),
        # Old data - outside month window
        TokenUsage(
            execution_id="exec-old-1",
            provider="openai",
            model="gpt-4",
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            created_at=base_time - timedelta(days=45),
        ),
    ]

    session.add_all(records)
    session.commit()
    return records


@pytest.fixture
def token_usage_service():
    # Patch just the _get_model_costs method so it returns our test costs
    with patch.object(
        usage.TokenUsageService, "_get_model_costs", return_value=MOCK_MODEL_COSTS
    ):
        yield usage.TokenUsageService()


def test_last_hour(usage_data, base_time):
    # Only return records from last hour

    result = usage.last_hour()

    assert result.total_tokens == 450  # 150 + 300
    assert len(result.providers) == 2

    openai_provider = next(p for p in result.providers if p.provider == "openai")
    assert openai_provider.total_tokens == 150

    anthropic_provider = next(p for p in result.providers if p.provider == "anthropic")
    assert anthropic_provider.total_tokens == 300


def test_last_day(usage_data, base_time):
    # Only return records from last 24 hours

    result = usage.last_day()

    assert result.total_tokens == 675  # 150 + 300 + 225
    assert len(result.providers) == 2


def test_last_week(usage_data, base_time):
    # Only return records from last 7 days
    result = usage.last_week()

    assert result.total_tokens == 1725  # 150 + 300 + 225 + 450 + 600
    assert len(result.providers) == 2


def test_last_month(usage_data, base_time):
    # Only return records from last 30 days
    result = usage.last_month()

    assert result.total_tokens == 3075  # All except the 45-day old record
    assert len(result.providers) == 2


def test_between_date_filtering(usage_data, base_time):
    # Test cases with different date/time formats
    test_cases = [
        # String dates with times
        {
            "start": (base_time - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S"),
            "end": (base_time - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "expected_tokens": 1050,
        },
        # datetime objects
        {
            "start": base_time - timedelta(days=7),
            "end": base_time - timedelta(days=1),
            "expected_tokens": 1050,
        },
        # Mixed formats
        {
            "start": (base_time - timedelta(days=7)).strftime("%Y-%m-%d"),
            "end": base_time - timedelta(days=1),
            "expected_tokens": 1050,
        },
    ]

    for case in test_cases:
        result = usage.between(case["start"], case["end"])
        assert (
            result.total_tokens == case["expected_tokens"]
        ), f"Failed for format: start={case['start']} end={case['end']} | expected {case['expected_tokens']}, got {result.total_tokens}"


def test_between_edge_cases(usage_data, base_time):
    # Test same day
    same_day = base_time.strftime("%Y-%m-%d")
    result = usage.between(same_day, same_day)
    assert isinstance(result, TokenUsageReport)

    # Test empty range
    result = usage.between(
        (base_time + timedelta(days=1)).strftime("%Y-%m-%d"),
        base_time.strftime("%Y-%m-%d"),
    )
    assert result.total_tokens == 0
    assert len(result.providers) == 0


def test_between_provider_filtering(usage_data, base_time):
    start_date = (base_time - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (base_time - timedelta(days=1)).strftime("%Y-%m-%d")

    result = usage.between(start_date, end_date, provider="openai")
    assert len(result.providers) == 1
    assert result.providers[0].provider == "openai"


def test_between_model_filtering(usage_data, base_time):
    start_date = (base_time - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (base_time - timedelta(days=1)).strftime("%Y-%m-%d")

    result = usage.between(start_date, end_date, model="gpt-4")
    assert len(result.providers) == 1
    # Ensure all returned models are gpt-4
    assert all(
        model.model == "gpt-4"
        for provider in result.providers
        for model in provider.models
    )


def test_provider_filtering(usage_data, base_time):
    result = usage.last_month(provider="openai")

    assert len(result.providers) == 1
    assert result.providers[0].provider == "openai"


def test_model_filtering(usage_data, base_time):
    result = usage.last_month(model="gpt-4o")

    assert result.total_tokens == 0  # No gpt-4o records in mock data


def test_execution_id_filtering(usage_data):
    result = usage.for_execution("exec-recent-1")

    assert result.total_tokens == 150
    assert len(result.providers) == 1
    assert result.providers[0].provider == "openai"


def test_last_execution(usage_data):
    result = usage.last_execution()

    assert result.total_tokens == 150  # exec-recent-1 is the most recent
    assert result.providers[0].provider == "openai"

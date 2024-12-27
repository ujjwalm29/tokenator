from datetime import datetime, timedelta
import pytest
from unittest.mock import patch, MagicMock

from tokenator.usage import (
    last_hour,
    last_day,
    last_week,
    last_month,
    between,
    for_execution,
    last_execution,
    TokenUsageReport,
    TokenRate,
)
from tokenator.schemas import TokenUsage

MOCK_MODEL_COSTS = {
    "gpt-4": TokenRate(prompt=0.03, completion=0.06),
    "gpt-4o": TokenRate(prompt=0.001, completion=0.002),
    "claude-3-5-haiku": TokenRate(prompt=0.004, completion=0.008),
    "claude-3-5-sonnet": TokenRate(prompt=0.005, completion=0.01),
}


@pytest.fixture
def mock_session():
    with patch("tokenator.usage.get_session") as mock:
        session = MagicMock()
        mock.return_value = lambda: session
        yield session


@pytest.fixture
def base_time():
    return datetime.now().replace(microsecond=0)


@pytest.fixture
def mock_usage_data(base_time):
    """Create a dataset of token usage records across different time periods"""
    return [
        # Recent data - within last hour
        TokenUsage(
            id=1,
            execution_id="exec-recent-1",
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            created_at=base_time - timedelta(minutes=20),
        ),
        TokenUsage(
            id=2,
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
            id=3,
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
            id=4,
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
            id=5,
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
            id=6,
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
            id=7,
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
            id=7,
            execution_id="exec-old-1",
            provider="openai",
            model="gpt-4",
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            created_at=base_time - timedelta(days=45),
        ),
    ]


@pytest.fixture(autouse=True)
def mock_model_costs():
    with patch("tokenator.usage.MODEL_COSTS", MOCK_MODEL_COSTS):
        yield


def test_last_hour(mock_session, mock_usage_data, base_time):
    # Only return records from last hour
    expected_data = [
        record
        for record in mock_usage_data
        if base_time - timedelta(hours=1) <= record.created_at <= base_time
    ]
    mock_session.query().filter().all.return_value = expected_data

    result = last_hour()

    assert result.total_tokens == 450  # 150 + 300
    assert len(result.providers) == 2

    openai_provider = next(p for p in result.providers if p.provider == "openai")
    assert openai_provider.total_tokens == 150

    anthropic_provider = next(p for p in result.providers if p.provider == "anthropic")
    assert anthropic_provider.total_tokens == 300


def test_last_day(mock_session, mock_usage_data, base_time):
    # Only return records from last 24 hours
    expected_data = [
        record
        for record in mock_usage_data
        if base_time - timedelta(days=1) <= record.created_at <= base_time
    ]
    mock_session.query().filter().all.return_value = expected_data

    result = last_day()

    assert result.total_tokens == 675  # 150 + 300 + 225
    assert len(result.providers) == 2


def test_last_week(mock_session, mock_usage_data, base_time):
    # Only return records from last 7 days
    expected_data = [
        record
        for record in mock_usage_data
        if base_time - timedelta(weeks=1) <= record.created_at <= base_time
    ]
    mock_session.query().filter().all.return_value = expected_data

    result = last_week()

    assert result.total_tokens == 1725  # 150 + 300 + 225 + 450 + 600
    assert len(result.providers) == 2


def test_last_month(mock_session, mock_usage_data, base_time):
    # Only return records from last 30 days
    expected_data = [
        record
        for record in mock_usage_data
        if base_time - timedelta(days=30) <= record.created_at <= base_time
    ]
    mock_session.query().filter().all.return_value = expected_data

    result = last_month()

    assert result.total_tokens == 3075  # All except the 45-day old record
    assert len(result.providers) == 2


def test_between_date_filtering(mock_session, mock_usage_data, base_time):
    # Test cases with different date/time formats
    test_cases = [
        # String dates only
        {
            "start": (base_time - timedelta(days=7)).strftime("%Y-%m-%d"),
            "end": (base_time - timedelta(days=1)).strftime("%Y-%m-%d"),
            "expected_tokens": 1050,  # 450 + 600
        },
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
        # Calculate expected data for this case
        if isinstance(case["start"], str):
            try:
                start = datetime.strptime(case["start"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                start = datetime.strptime(case["start"], "%Y-%m-%d")  # Sets to 00:00:00
        else:
            start = case["start"]

        if isinstance(case["end"], str):
            try:
                end = datetime.strptime(case["end"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                # Set to 23:59:59 of the day
                end = datetime.strptime(case["end"], "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
        else:
            end = case["end"]

        expected_data = [
            record for record in mock_usage_data if start <= record.created_at <= end
        ]
        mock_session.query().filter().all.return_value = expected_data

        result = between(case["start"], case["end"])
        assert (
            result.total_tokens == case["expected_tokens"]
        ), f"Failed for format: start={type(case['start'])}, end={type(case['end'])}"


def test_between_edge_cases(mock_session, mock_usage_data, base_time):
    # Test same day
    same_day = base_time.strftime("%Y-%m-%d")
    expected_data = [
        record
        for record in mock_usage_data
        if record.created_at.date() == base_time.date()
    ]
    mock_session.query().filter().all.return_value = expected_data
    result = between(same_day, same_day)
    assert isinstance(result, TokenUsageReport)

    # Test empty range
    mock_session.query().filter().all.return_value = []
    result = between(
        (base_time + timedelta(days=1)).strftime("%Y-%m-%d"),
        base_time.strftime("%Y-%m-%d"),
    )
    assert result.total_tokens == 0
    assert len(result.providers) == 0


def test_between_provider_filtering(mock_session, mock_usage_data, base_time):
    start_date = (base_time - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (base_time - timedelta(days=1)).strftime("%Y-%m-%d")

    expected_data = [
        record
        for record in mock_usage_data
        if (
            datetime.strptime(start_date, "%Y-%m-%d")
            <= record.created_at
            <= datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            and record.provider == "openai"
        )
    ]

    query_mock = MagicMock()
    query_mock.filter.return_value = query_mock
    query_mock.all.return_value = expected_data
    mock_session.query.return_value = query_mock

    result = between(start_date, end_date, provider="openai")
    assert len(result.providers) == 1
    assert result.providers[0].provider == "openai"


def test_between_model_filtering(mock_session, mock_usage_data, base_time):
    start_date = (base_time - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (base_time - timedelta(days=1)).strftime("%Y-%m-%d")

    expected_data = [
        record
        for record in mock_usage_data
        if (
            datetime.strptime(start_date, "%Y-%m-%d")
            <= record.created_at
            <= datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            and record.model == "gpt-4"
        )
    ]

    query_mock = MagicMock()
    query_mock.filter.return_value = query_mock
    query_mock.all.return_value = expected_data
    mock_session.query.return_value = query_mock

    result = between(start_date, end_date, model="gpt-4")
    assert len(result.providers) == 1
    assert all(
        model.model == "gpt-4"
        for provider in result.providers
        for model in provider.models
    )


def test_provider_filtering(mock_session, mock_usage_data, base_time):
    # Filter for last month + specific provider
    expected_data = [
        record
        for record in mock_usage_data
        if (
            base_time - timedelta(days=30) <= record.created_at <= base_time
            and record.provider == "openai"
        )
    ]

    query_mock = MagicMock()
    query_mock.filter.return_value = query_mock
    query_mock.all.return_value = expected_data
    mock_session.query.return_value = query_mock

    result = last_month(provider="openai")

    assert len(result.providers) == 1
    assert result.providers[0].provider == "openai"


def test_model_filtering(mock_session, mock_usage_data, base_time):
    # Filter for last month + specific model
    expected_data = [
        record
        for record in mock_usage_data
        if (
            base_time - timedelta(days=30) <= record.created_at <= base_time
            and record.model == "gpt-4o"
        )
    ]

    query_mock = MagicMock()
    query_mock.filter.return_value = query_mock
    query_mock.all.return_value = expected_data
    mock_session.query.return_value = query_mock

    result = last_month(model="gpt-4o")

    assert result.total_tokens == 0  # No gpt-4o records in mock data


def test_execution_id_filtering(mock_session, mock_usage_data):
    expected_data = [u for u in mock_usage_data if u.execution_id == "exec-recent-1"]
    mock_session.query().filter().all.return_value = expected_data

    result = for_execution("exec-recent-1")

    assert result.total_tokens == 150
    assert len(result.providers) == 1
    assert result.providers[0].provider == "openai"


def test_last_execution(mock_session, mock_usage_data):
    # Mock the most recent execution query
    most_recent = max(mock_usage_data, key=lambda x: x.created_at)
    mock_session.query().order_by().first.return_value = most_recent

    # Mock the usage for that execution
    expected_data = [
        u for u in mock_usage_data if u.execution_id == most_recent.execution_id
    ]
    mock_session.query().filter().all.return_value = expected_data

    result = last_execution()

    assert result.total_tokens == 150  # exec-recent-1 is the most recent
    assert result.providers[0].provider == "openai"

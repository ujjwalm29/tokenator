"""Cost analysis functions for token usage."""

from datetime import datetime, timedelta, timezone
from typing import Dict

from sqlalchemy import and_

from .models import get_session, TokenUsage

MODEL_COSTS = {
    "gpt-4o-2024-08-06": {
        "prompt": 3,
        "completion": 6
    }
}

def _calculate_cost(usages: list[TokenUsage], provider: str) -> Dict:
    """Calculate cost from token usage records."""
    total_cost = 0.0
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    
    for usage in usages:
        if usage.model in MODEL_COSTS:
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
            total_tokens += usage.total_tokens
            
            total_cost += (usage.prompt_tokens * MODEL_COSTS[usage.model]["prompt"])
            total_cost += (usage.completion_tokens * MODEL_COSTS[usage.model]["completion"])
    
    return {
        "provider": provider,
        "total_cost": round(total_cost, 6),
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens
    }

def _query_usage(start_date: datetime, end_date: datetime, provider: str) -> Dict:
    """Query token usage for a specific time period."""
    session = get_session()()
    try:
        usages = session.query(TokenUsage).filter(
            and_(
                TokenUsage.provider == provider,
                TokenUsage.created_at.between(start_date, end_date)
            )
        ).all()
        return _calculate_cost(usages, provider)
    finally:
        session.close()

def last_hour(provider: str = "openai") -> Dict:
    """Get cost analysis for the last hour."""
    end = datetime.now()
    start = end - timedelta(hours=1)
    return _query_usage(start, end, provider)

def last_day(provider: str = "openai") -> Dict:
    """Get cost analysis for the last 24 hours."""
    end = datetime.now()
    start = end - timedelta(days=1)
    return _query_usage(start, end, provider)

def last_week(provider: str = "openai") -> Dict:
    """Get cost analysis for the last 7 days."""
    end = datetime.now()
    start = end - timedelta(weeks=1)
    return _query_usage(start, end, provider)

def last_month(provider: str = "openai") -> Dict:
    """Get cost analysis for the last 30 days."""
    end = datetime.now()
    start = end - timedelta(days=30)
    return _query_usage(start, end, provider)

def between(start_date: str, end_date: str, provider: str = "openai") -> Dict:
    """Get cost analysis between two dates (format: YYYY-MM-DD)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)  # Include the end date
    return _query_usage(start, end, provider)

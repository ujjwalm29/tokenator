"""Cost analysis functions for token usage."""

from datetime import datetime, timedelta, timezone
from typing import Dict

from sqlalchemy import and_

from .models import get_session, TokenUsage
from .schemas import TokenRate, TokenUsageReport, ModelUsage, ProviderUsage

import requests

def get_model_costs() -> Dict[str, TokenRate]:
    url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
    response = requests.get(url)
    data = response.json()
    
    return {
        model: TokenRate(
            prompt=info["input_cost_per_token"],
            completion=info["output_cost_per_token"]
        )
        for model, info in data.items()
        if "input_cost_per_token" in info and "output_cost_per_token" in info
    }

MODEL_COSTS = get_model_costs()

def _calculate_cost(usages: list[TokenUsage], provider: str | None = None) -> TokenUsageReport:
    """Calculate cost from token usage records."""
    # Group usages by provider and model
    provider_model_usages: Dict[str, Dict[str, list[TokenUsage]]] = {}
    
    for usage in usages:
        if usage.model not in MODEL_COSTS:
            continue
            
        provider = usage.provider
        if provider not in provider_model_usages:
            provider_model_usages[provider] = {}
        
        if usage.model not in provider_model_usages[provider]:
            provider_model_usages[provider][usage.model] = []
            
        provider_model_usages[provider][usage.model].append(usage)

    # Calculate totals for each level
    providers_list = []
    total_metrics = {"total_cost": 0.0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    for provider, model_usages in provider_model_usages.items():
        provider_metrics = {"total_cost": 0.0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        models_list = []

        for model, usages in model_usages.items():
            model_cost = 0.0
            model_total = 0
            model_prompt = 0
            model_completion = 0

            for usage in usages:
                model_prompt += usage.prompt_tokens
                model_completion += usage.completion_tokens
                model_total += usage.total_tokens
                
                model_cost += (usage.prompt_tokens * MODEL_COSTS[usage.model].prompt)
                model_cost += (usage.completion_tokens * MODEL_COSTS[usage.model].completion)

            models_list.append(ModelUsage(
                model=model,
                total_cost=round(model_cost, 6),
                total_tokens=model_total,
                prompt_tokens=model_prompt,
                completion_tokens=model_completion
            ))

            # Add to provider totals
            provider_metrics["total_cost"] += model_cost
            provider_metrics["total_tokens"] += model_total
            provider_metrics["prompt_tokens"] += model_prompt
            provider_metrics["completion_tokens"] += model_completion

        providers_list.append(ProviderUsage(
            provider=provider,
            models=models_list,
            **{k: (round(v, 6) if k == "total_cost" else v) for k, v in provider_metrics.items()}
        ))

        # Add to grand totals
        for key in total_metrics:
            total_metrics[key] += provider_metrics[key]

    return TokenUsageReport(
        providers=providers_list,
        **{k: (round(v, 6) if k == "total_cost" else v) for k, v in total_metrics.items()}
    )

def _query_usage(start_date: datetime, end_date: datetime, provider: str | None = None, model: str | None = None) -> TokenUsageReport:
    """Query token usage for a specific time period."""
    session = get_session()()
    try:
        query = session.query(TokenUsage).filter(
            TokenUsage.created_at.between(start_date, end_date)
        )
        
        if provider:
            query = query.filter(TokenUsage.provider == provider)
        if model:
            query = query.filter(TokenUsage.model == model)
            
        usages = query.all()
        return _calculate_cost(usages, provider or "all")
    finally:
        session.close()

def last_hour(provider: str | None = None, model: str | None = None) -> TokenUsageReport:
    """Get cost analysis for the last hour."""
    end = datetime.now()
    start = end - timedelta(hours=1)
    return _query_usage(start, end, provider, model)

def last_day(provider: str | None = None, model: str | None = None) -> TokenUsageReport:
    """Get cost analysis for the last 24 hours."""
    end = datetime.now()
    start = end - timedelta(days=1)
    return _query_usage(start, end, provider, model)

def last_week(provider: str | None = None, model: str | None = None) -> TokenUsageReport:
    """Get cost analysis for the last 7 days."""
    end = datetime.now()
    start = end - timedelta(weeks=1)
    return _query_usage(start, end, provider, model)

def last_month(provider: str | None = None, model: str | None = None) -> TokenUsageReport:
    """Get cost analysis for the last 30 days."""
    end = datetime.now()
    start = end - timedelta(days=30)
    return _query_usage(start, end, provider, model)

def between(start_date: str, end_date: str, provider: str | None = None, model: str | None = None) -> TokenUsageReport:
    """Get cost analysis between two dates (format: YYYY-MM-DD)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)  # Include the end date
    return _query_usage(start, end, provider, model)

def for_execution(execution_id: str) -> TokenUsageReport:
    """Get cost analysis for a specific execution."""
    session = get_session()()
    query = session.query(TokenUsage).filter(TokenUsage.execution_id == execution_id)
    return _calculate_cost(query.all())

def last_execution() -> TokenUsageReport:
    """Get cost analysis for the last execution_id."""
    session = get_session()()
    query = session.query(TokenUsage).order_by(TokenUsage.created_at.desc()).first()
    return for_execution(query.execution_id)

"""Cost analysis functions for token usage."""

from datetime import datetime, timedelta
from typing import Dict, Optional, Union

from .schemas import get_session, TokenUsage
from .models import TokenRate, TokenUsageReport, ModelUsage, ProviderUsage
from . import state

import requests
import logging

logger = logging.getLogger(__name__)


class TokenUsageService:
    def __init__(self):
        if not state.is_tokenator_enabled:
            logger.info("Tokenator is disabled. Database access is unavailable.")

        self.MODEL_COSTS = self._get_model_costs()

    def _get_model_costs(self) -> Dict[str, TokenRate]:
        if not state.is_tokenator_enabled:
            return {}
        url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
        response = requests.get(url)
        data = response.json()

        return {
            model: TokenRate(
                prompt=info["input_cost_per_token"],
                completion=info["output_cost_per_token"],
            )
            for model, info in data.items()
            if "input_cost_per_token" in info and "output_cost_per_token" in info
        }

    def _calculate_cost(
        self, usages: list[TokenUsage], provider: Optional[str] = None
    ) -> TokenUsageReport:
        if not state.is_tokenator_enabled:
            logger.warning("Tokenator is disabled. Skipping cost calculation.")
            return TokenUsageReport()

        if not self.MODEL_COSTS:
            logger.warning("No model costs available.")
            return TokenUsageReport()

        GPT4O_PRICING = self.MODEL_COSTS.get(
            "gpt-4o", TokenRate(prompt=0.0000025, completion=0.000010)
        )

        # Existing calculation logic...
        provider_model_usages: Dict[str, Dict[str, list[TokenUsage]]] = {}
        logger.debug(f"usages: {len(usages)}")

        for usage in usages:
            # 1st priority - direct match
            model_key = usage.model
            if model_key in self.MODEL_COSTS:
                pass
            # 2nd priority - provider/model format
            elif f"{usage.provider}/{usage.model}" in self.MODEL_COSTS:
                model_key = f"{usage.provider}/{usage.model}"
            # 3rd priority - contains search
            else:
                matched_keys = [k for k in self.MODEL_COSTS.keys() if usage.model in k]
                if matched_keys:
                    model_key = matched_keys[0]
                    logger.warning(
                        f"Model {usage.model} matched with {model_key} in pricing data via contains search"
                    )
                else:
                    # Fallback to GPT4O pricing
                    logger.warning(
                        f"Model {model_key} not found in pricing data. Using gpt-4o pricing as fallback "
                        f"(prompt: ${GPT4O_PRICING.prompt}/token, completion: ${GPT4O_PRICING.completion}/token)"
                    )
                    self.MODEL_COSTS[model_key] = GPT4O_PRICING

            provider_key = usage.provider or "default"
            provider_model_usages.setdefault(provider_key, {}).setdefault(
                model_key, []
            ).append(usage)

        # Calculate totals for each level
        providers_list = []
        total_metrics = {
            "total_cost": 0.0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

        for provider, model_usages in provider_model_usages.items():
            provider_metrics = {
                "total_cost": 0.0,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
            models_list = []

            for model_key, usages in model_usages.items():
                model_cost = sum(
                    usage.prompt_tokens * self.MODEL_COSTS[model_key].prompt
                    + usage.completion_tokens * self.MODEL_COSTS[model_key].completion
                    for usage in usages
                )
                model_total = sum(usage.total_tokens for usage in usages)
                model_prompt = sum(usage.prompt_tokens for usage in usages)
                model_completion = sum(usage.completion_tokens for usage in usages)

                models_list.append(
                    ModelUsage(
                        model=model_key,
                        total_cost=round(model_cost, 6),
                        total_tokens=model_total,
                        prompt_tokens=model_prompt,
                        completion_tokens=model_completion,
                    )
                )

                provider_metrics["total_cost"] += model_cost
                provider_metrics["total_tokens"] += model_total
                provider_metrics["prompt_tokens"] += model_prompt
                provider_metrics["completion_tokens"] += model_completion

            providers_list.append(
                ProviderUsage(
                    provider=provider,
                    models=models_list,
                    **{
                        k: (round(v, 6) if k == "total_cost" else v)
                        for k, v in provider_metrics.items()
                    },
                )
            )

            for key in total_metrics:
                total_metrics[key] += provider_metrics[key]

        return TokenUsageReport(
            providers=providers_list,
            **{
                k: (round(v, 6) if k == "total_cost" else v)
                for k, v in total_metrics.items()
            },
        )

    def _query_usage(
        self,
        start_date: datetime,
        end_date: datetime,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> TokenUsageReport:
        if not state.is_tokenator_enabled:
            logger.warning("Tokenator is disabled. Skipping usage query.")
            return TokenUsageReport()

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

            return self._calculate_cost(usages, provider or "all")
        finally:
            session.close()

    def last_hour(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> TokenUsageReport:
        if not state.is_tokenator_enabled:
            return TokenUsageReport()
        logger.debug(
            f"Getting cost analysis for last hour (provider={provider}, model={model})"
        )
        end = datetime.now()
        start = end - timedelta(hours=1)
        return self._query_usage(start, end, provider, model)

    def last_day(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> TokenUsageReport:
        if not state.is_tokenator_enabled:
            return TokenUsageReport()
        logger.debug(
            f"Getting cost analysis for last 24 hours (provider={provider}, model={model})"
        )
        end = datetime.now()
        start = end - timedelta(days=1)
        return self._query_usage(start, end, provider, model)

    def last_week(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> TokenUsageReport:
        if not state.is_tokenator_enabled:
            return TokenUsageReport()
        logger.debug(
            f"Getting cost analysis for last 7 days (provider={provider}, model={model})"
        )
        end = datetime.now()
        start = end - timedelta(weeks=1)
        return self._query_usage(start, end, provider, model)

    def last_month(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> TokenUsageReport:
        if not state.is_tokenator_enabled:
            return TokenUsageReport()
        logger.debug(
            f"Getting cost analysis for last 30 days (provider={provider}, model={model})"
        )
        end = datetime.now()
        start = end - timedelta(days=30)
        return self._query_usage(start, end, provider, model)

    def between(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> TokenUsageReport:
        if not state.is_tokenator_enabled:
            return TokenUsageReport()
        logger.debug(
            f"Getting cost analysis between {start_date} and {end_date} (provider={provider}, model={model})"
        )

        if isinstance(start_date, str):
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                logger.warning(
                    f"Date-only string provided for start_date: {start_date}. Setting time to 00:00:00"
                )
                start = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start = start_date

        if isinstance(end_date, str):
            try:
                end = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                logger.warning(
                    f"Date-only string provided for end_date: {end_date}. Setting time to 23:59:59"
                )
                end = (
                    datetime.strptime(end_date, "%Y-%m-%d")
                    + timedelta(days=1)
                    - timedelta(seconds=1)
                )
        else:
            end = end_date

        return self._query_usage(start, end, provider, model)

    def for_execution(self, execution_id: str) -> TokenUsageReport:
        if not state.is_tokenator_enabled:
            return TokenUsageReport()
        logger.debug(f"Getting cost analysis for execution_id={execution_id}")
        session = get_session()()
        try:
            query = session.query(TokenUsage).filter(
                TokenUsage.execution_id == execution_id
            )
            return self._calculate_cost(query.all())
        finally:
            session.close()

    def last_execution(self) -> TokenUsageReport:
        if not state.is_tokenator_enabled:
            return TokenUsageReport()
        logger.debug("Getting cost analysis for last execution")
        session = get_session()()
        try:
            query = (
                session.query(TokenUsage).order_by(TokenUsage.created_at.desc()).first()
            )
            if query:
                return self.for_execution(query.execution_id)
            return TokenUsageReport()
        finally:
            session.close()

    def all_time(self) -> TokenUsageReport:
        if not state.is_tokenator_enabled:
            return TokenUsageReport()

        logger.warning("Getting cost analysis for all time. This may take a while...")
        session = get_session()()
        try:
            query = session.query(TokenUsage)
            return self._calculate_cost(query.all())
        finally:
            session.close()

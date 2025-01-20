"""Cost analysis functions for token usage."""

from datetime import datetime, timedelta
from typing import Dict, Optional, Union

from .schemas import get_session, TokenUsage
from .models import (
    CompletionTokenDetails,
    PromptTokenDetails,
    TokenRate,
    TokenUsageReport,
    ModelUsage,
    ProviderUsage,
)
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

        model_costs = {}
        for model, info in data.items():
            if (
                "input_cost_per_token" not in info
                or "output_cost_per_token" not in info
            ):
                continue

            rate = TokenRate(
                prompt=info["input_cost_per_token"],
                completion=info["output_cost_per_token"],
                prompt_audio=info.get("input_cost_per_audio_token"),
                completion_audio=info.get("output_cost_per_audio_token"),
                prompt_cached_input=info.get("cache_read_input_token_cost") or 0,
                prompt_cached_creation=info.get("cache_read_creation_token_cost") or 0,
            )
            model_costs[model] = rate

        return model_costs

    def _calculate_cost(
        self, usages: list[TokenUsage], provider: Optional[str] = None
    ) -> TokenUsageReport:
        if not state.is_tokenator_enabled:
            logger.warning("Tokenator is disabled. Skipping cost calculation.")
            return TokenUsageReport()

        if not self.MODEL_COSTS:
            logger.warning("No model costs available.")
            return TokenUsageReport()

        # Default GPT4O pricing updated with provided values
        GPT4O_PRICING = TokenRate(
            prompt=0.0000025,
            completion=0.000010,
            prompt_audio=0.0001,
            completion_audio=0.0002,
            prompt_cached_input=0.00000125,
            prompt_cached_creation=0.00000125,
        )

        provider_model_usages: Dict[str, Dict[str, list[TokenUsage]]] = {}
        logger.debug(f"usages: {len(usages)}")

        for usage in usages:
            # Model key resolution logic (unchanged)
            model_key = usage.model
            if model_key in self.MODEL_COSTS:
                pass
            elif f"{usage.provider}/{usage.model}" in self.MODEL_COSTS:
                model_key = f"{usage.provider}/{usage.model}"
            else:
                matched_keys = [k for k in self.MODEL_COSTS.keys() if usage.model in k]
                if matched_keys:
                    model_key = matched_keys[0]
                    logger.warning(
                        f"Model {usage.model} matched with {model_key} in pricing data via contains search"
                    )
                else:
                    logger.warning(
                        f"Model {model_key} not found in pricing data. Using gpt-4o pricing as fallback"
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
                "prompt_cached_input_tokens": 0,
                "prompt_cached_creation_tokens": 0,
                "prompt_audio_tokens": 0,
                "completion_audio_tokens": 0,
                "completion_reasoning_tokens": 0,
                "completion_accepted_prediction_tokens": 0,
                "completion_rejected_prediction_tokens": 0,
            }
            models_list = []

            for model_key, usages in model_usages.items():
                model_rates = self.MODEL_COSTS[model_key]
                model_cost = 0.0
                model_total = 0
                model_prompt = 0
                model_completion = 0

                for usage in usages:
                    # Base token costs
                    prompt_text_tokens = usage.prompt_tokens
                    if usage.prompt_cached_input_tokens:
                        prompt_text_tokens = (
                            usage.prompt_tokens - usage.prompt_cached_input_tokens
                        )
                    if usage.prompt_audio_tokens:
                        prompt_text_tokens = (
                            usage.prompt_tokens - usage.prompt_audio_tokens
                        )

                    completion_text_tokens = usage.completion_tokens
                    if usage.completion_audio_tokens:
                        completion_text_tokens = (
                            usage.completion_tokens - usage.completion_audio_tokens
                        )

                    prompt_cost = prompt_text_tokens * model_rates.prompt
                    completion_cost = completion_text_tokens * model_rates.completion
                    model_cost += prompt_cost + completion_cost

                    # Audio token costs
                    if usage.prompt_audio_tokens:
                        if model_rates.prompt_audio:
                            model_cost += (
                                usage.prompt_audio_tokens * model_rates.prompt_audio
                            )
                        else:
                            logger.warning(
                                f"Audio prompt tokens present for {model_key} but no audio rate defined"
                            )

                    if usage.completion_audio_tokens:
                        if model_rates.completion_audio:
                            model_cost += (
                                usage.completion_audio_tokens
                                * model_rates.completion_audio
                            )
                        else:
                            logger.warning(
                                f"Audio completion tokens present for {model_key} but no audio rate defined"
                            )

                    # Cached token costs
                    if usage.prompt_cached_input_tokens:
                        if model_rates.prompt_cached_input:
                            model_cost += (
                                usage.prompt_cached_input_tokens
                                * model_rates.prompt_cached_input
                            )
                        else:
                            logger.warning(
                                f"Cached input tokens present for {model_key} but no cache input rate defined"
                            )

                    if usage.prompt_cached_creation_tokens:
                        if model_rates.prompt_cached_creation:
                            model_cost += (
                                usage.prompt_cached_creation_tokens
                                * model_rates.prompt_cached_creation
                            )
                        else:
                            logger.warning(
                                f"Cached creation tokens present for {model_key} but no cache creation rate defined"
                            )

                    model_total += usage.total_tokens
                    model_prompt += usage.prompt_tokens
                    model_completion += usage.completion_tokens

                models_list.append(
                    ModelUsage(
                        model=model_key,
                        total_cost=round(model_cost, 6),
                        total_tokens=model_total,
                        prompt_tokens=model_prompt,
                        completion_tokens=model_completion,
                        prompt_tokens_details=PromptTokenDetails(
                            cached_input_tokens=sum(
                                u.prompt_cached_input_tokens or 0 for u in usages
                            ),
                            cached_creation_tokens=sum(
                                u.prompt_cached_creation_tokens or 0 for u in usages
                            ),
                            audio_tokens=sum(
                                u.prompt_audio_tokens or 0 for u in usages
                            ),
                        )
                        if any(
                            u.prompt_cached_input_tokens
                            or u.prompt_cached_creation_tokens
                            or u.prompt_audio_tokens
                            for u in usages
                        )
                        else None,
                        completion_tokens_details=CompletionTokenDetails(
                            audio_tokens=sum(
                                u.completion_audio_tokens or 0 for u in usages
                            ),
                            reasoning_tokens=sum(
                                u.completion_reasoning_tokens or 0 for u in usages
                            ),
                            accepted_prediction_tokens=sum(
                                u.completion_accepted_prediction_tokens or 0
                                for u in usages
                            ),
                            rejected_prediction_tokens=sum(
                                u.completion_rejected_prediction_tokens or 0
                                for u in usages
                            ),
                        )
                        if any(
                            getattr(u, attr, None)
                            for u in usages
                            for attr in [
                                "completion_audio_tokens",
                                "completion_reasoning_tokens",
                                "completion_accepted_prediction_tokens",
                                "completion_rejected_prediction_tokens",
                            ]
                        )
                        else None,
                    )
                )

                # Update provider metrics with all token types
                provider_metrics["total_cost"] += model_cost
                provider_metrics["total_tokens"] += model_total
                provider_metrics["prompt_tokens"] += model_prompt
                provider_metrics["completion_tokens"] += model_completion
                provider_metrics["prompt_cached_input_tokens"] += sum(
                    u.prompt_cached_input_tokens or 0 for u in usages
                )
                provider_metrics["prompt_cached_creation_tokens"] += sum(
                    u.prompt_cached_creation_tokens or 0 for u in usages
                )
                provider_metrics["prompt_audio_tokens"] += sum(
                    u.prompt_audio_tokens or 0 for u in usages
                )
                provider_metrics["completion_audio_tokens"] += sum(
                    u.completion_audio_tokens or 0 for u in usages
                )
                provider_metrics["completion_reasoning_tokens"] += sum(
                    u.completion_reasoning_tokens or 0 for u in usages
                )
                provider_metrics["completion_accepted_prediction_tokens"] += sum(
                    u.completion_accepted_prediction_tokens or 0 for u in usages
                )
                provider_metrics["completion_rejected_prediction_tokens"] += sum(
                    u.completion_rejected_prediction_tokens or 0 for u in usages
                )

            providers_list.append(
                ProviderUsage(
                    provider=provider,
                    models=models_list,
                    total_cost=round(provider_metrics["total_cost"], 6),
                    total_tokens=provider_metrics["total_tokens"],
                    prompt_tokens=provider_metrics["prompt_tokens"],
                    completion_tokens=provider_metrics["completion_tokens"],
                    prompt_tokens_details=PromptTokenDetails(
                        cached_input_tokens=provider_metrics[
                            "prompt_cached_input_tokens"
                        ],
                        cached_creation_tokens=provider_metrics[
                            "prompt_cached_creation_tokens"
                        ],
                        audio_tokens=provider_metrics["prompt_audio_tokens"],
                    )
                    if provider_metrics["prompt_cached_input_tokens"]
                    or provider_metrics["prompt_cached_creation_tokens"]
                    or provider_metrics["prompt_audio_tokens"]
                    else None,
                    completion_tokens_details=CompletionTokenDetails(
                        audio_tokens=provider_metrics["completion_audio_tokens"],
                        reasoning_tokens=provider_metrics[
                            "completion_reasoning_tokens"
                        ],
                        accepted_prediction_tokens=provider_metrics[
                            "completion_accepted_prediction_tokens"
                        ],
                        rejected_prediction_tokens=provider_metrics[
                            "completion_rejected_prediction_tokens"
                        ],
                    )
                    if any(
                        provider_metrics[k]
                        for k in [
                            "completion_audio_tokens",
                            "completion_reasoning_tokens",
                            "completion_accepted_prediction_tokens",
                            "completion_rejected_prediction_tokens",
                        ]
                    )
                    else None,
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

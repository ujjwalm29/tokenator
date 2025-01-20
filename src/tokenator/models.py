from pydantic import BaseModel, Field
from typing import List, Optional


class TokenRate(BaseModel):
    prompt: float = Field(..., description="Cost per prompt token")
    completion: float = Field(..., description="Cost per completion token")
    prompt_audio: Optional[float] = Field(
        None, description="Cost per audio prompt token"
    )
    completion_audio: Optional[float] = Field(
        None, description="Cost per audio completion token"
    )
    prompt_cached_input: Optional[float] = Field(
        None, description="Cost per cached prompt input token"
    )
    prompt_cached_creation: Optional[float] = Field(
        None, description="Cost per cached prompt creation token"
    )


class PromptTokenDetails(BaseModel):
    cached_input_tokens: Optional[int] = None
    cached_creation_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None


class CompletionTokenDetails(BaseModel):
    reasoning_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    accepted_prediction_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None


class TokenMetrics(BaseModel):
    total_cost: float = Field(default=0, description="Total cost in USD")
    total_tokens: int = Field(default=0, description="Total tokens used")
    prompt_tokens: int = Field(default=0, description="Number of prompt tokens")
    completion_tokens: int = Field(default=0, description="Number of completion tokens")
    prompt_tokens_details: Optional[PromptTokenDetails] = None
    completion_tokens_details: Optional[CompletionTokenDetails] = None


class ModelUsage(TokenMetrics):
    model: str = Field(..., description="Model name")


class ProviderUsage(TokenMetrics):
    provider: str = Field(..., description="Provider name")
    models: List[ModelUsage] = Field(
        default_factory=list, description="Usage breakdown by model"
    )


class TokenUsageReport(TokenMetrics):
    providers: List[ProviderUsage] = Field(
        default_factory=list, description="Usage breakdown by provider"
    )


class TokenUsageStats(BaseModel):
    model: str
    usage: TokenMetrics

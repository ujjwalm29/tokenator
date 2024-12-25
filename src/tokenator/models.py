from pydantic import BaseModel, Field
from typing import List


class TokenRate(BaseModel):
    prompt: float = Field(..., description="Cost per prompt token")
    completion: float = Field(..., description="Cost per completion token")


class TokenMetrics(BaseModel):
    total_cost: float = Field(..., description="Total cost in USD")
    total_tokens: int = Field(..., description="Total tokens used")
    prompt_tokens: int = Field(..., description="Number of prompt tokens")
    completion_tokens: int = Field(..., description="Number of completion tokens")


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


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TokenUsageStats(BaseModel):
    model: str
    usage: Usage

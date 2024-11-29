from pydantic import BaseModel, Field
from typing import Dict, List

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
    models: List[ModelUsage] = Field(default_factory=list, description="Usage breakdown by model")

class TokenUsageReport(TokenMetrics):
    providers: List[ProviderUsage] = Field(default_factory=list, description="Usage breakdown by provider")
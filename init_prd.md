

# Tokenator PRD

## Product Overview

### Problem Statement
Developers using OpenAI APIs lack visibility into their token usage and associated costs, making it difficult to track and manage expenses.

### Target Audience
- Python developers using OpenAI APIs
- DevOps teams managing AI costs
- Engineering managers tracking AI spending

### Business Objectives
- Provide real-time token usage tracking
- Enable cost analysis across different time periods
- Simplify OpenAI cost management

### Success Metrics
- GitHub stars/forks
- Number of PyPI downloads
- Community contributions

## Requirements

### Functional Requirements

1. **OpenAI Client Wrapper**
```python
from tokenator import OpenAIWrapper

client = OpenAIWrapper(api_key="sk-...")
response = client.chat.completions.create(...)  # Works exactly like openai client
```

2. **Cost Analysis Functions**
```python
from tokenator import cost

# Predefined periods
cost.last_hour("openai")
cost.last_day("openai")
cost.last_week("openai")
cost.last_month("openai")

# Custom period
cost.between("2024-03-01", "2024-03-15", "openai")
```

### Non-Functional Requirements
- Zero impact on API performance
- Thread-safe database operations
- Minimal memory footprint
- No external dependencies except OpenAI client

## Technical Specifications

### Database Schema
```sql
CREATE TABLE token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL
);

CREATE INDEX idx_created_at ON token_usage(created_at);
CREATE INDEX idx_execution_id ON token_usage(execution_id);
CREATE INDEX idx_provider ON token_usage(provider);
CREATE INDEX idx_model ON token_usage(model);
```

### Token Pricing Configuration
```python
MODEL_COSTS = {
    "gpt-4": {
        "prompt": 0.03,
        "completion": 0.06
    },
    "gpt-3.5-turbo": {
        "prompt": 0.0015,
        "completion": 0.002
    }
}
```

### Installation
```bash
pip install tokenator
```

## MVP Scope

### Phase 1
- Basic OpenAI client wrapper
- SQLite integration
- Cost calculation for predefined periods
- Support for GPT-4 models

### Phase 2
- API for real-time pricing updates
- Export functionality (CSV, JSON)
- Cost alerts/thresholds
- Dashboard UI (optional)

### Out of Scope
- Support for other AI providers
- Complex analytics
- Multi-user support
- Cloud storage options

## Timeline
1. Core Implementation (2 weeks)
2. Testing & Documentation (1 week)
3. Initial Release (v0.1.0)
4. Community Feedback (2 weeks)
5. Phase 2 Features (1 month)

## Future Considerations
- Azure OpenAI support
- Anthropic Claude support
- Cost prediction based on usage patterns
- Multi-database support (PostgreSQL, MongoDB)
- Team collaboration features

## Analytics
Track:
- Number of API calls
- Token usage patterns
- Cost per execution_id
- Most used time periods for analysis

This library will be maintained on GitHub under MIT license to encourage community contributions.

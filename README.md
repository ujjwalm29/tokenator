# Tokenator : Easiest way to track and analyze LLM token usage and cost

Have you ever wondered about :
- How many tokens does your AI agent consume? 
- How much does it cost to do run a complex AI workflow with multiple LLM providers?
- How much money did I spent today on development?

Afraid not, tokenator is here! With tokenator's easy to use API, you can start tracking LLM usage in a matter of minutes.

Get started with just 3 lines of code!

## Installation

```bash
pip install tokenator
```

## Usage

### OpenAI

```python
from openai import OpenAI
from tokenator import tokenator_openai

openai_client = OpenAI(api_key="your-api-key")

# Wrap it with Tokenator
client = tokenator_openai(openai_client)

# Use it exactly like the OpenAI client
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Cost Analysis

```python
from tokenator import cost

# Get usage for different time periods
cost.last_hour()
cost.last_day()
cost.last_week()
cost.last_month()

# Custom date range
cost.between("2024-03-01", "2024-03-15")

# Get usage for different LLM providers
cost.last_day("openai")
cost.last_day("anthropic")
cost.last_day("google")
```

### Example `cost` object

```json
# print(cost.last_hour().model_dump_json(indent=4))

usage : {
    "total_cost": 0.0004,
    "total_tokens": 79,
    "prompt_tokens": 52,
    "completion_tokens": 27,
    "providers": [
        {
            "total_cost": 0.0004,
            "total_tokens": 79,
            "prompt_tokens": 52,
            "completion_tokens": 27,
            "provider": "openai",
            "models": [
                {
                    "total_cost": 0.0004,
                    "total_tokens": 79,
                    "prompt_tokens": 52,
                    "completion_tokens": 27,
                    "model": "gpt-4o-2024-08-06"
                }
            ]
        }
    ]
}
```

## Features

- Drop-in replacement for OpenAI, Anthropic client
- Automatic token usage tracking
- Cost analysis for different time periods
- SQLite storage with zero configuration
- Thread-safe operations
- Minimal memory footprint
- Minimal latency footprint

Most importantly, none of your data is ever sent to any server.

## License

MIT 
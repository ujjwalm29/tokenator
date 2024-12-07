# Tokenator

Track and analyze your OpenAI API token usage and costs.

## Installation

```bash
pip install tokenator
```

## Usage

### OpenAI Client Wrapper

```python
from tokenator import tokenator_openai
from openai import OpenAI

# Initialize the OpenAI client
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
cost.last_hour("openai")
cost.last_day("openai")
cost.last_week("openai")
cost.last_month("openai")

# Custom date range
cost.between("2024-03-01", "2024-03-15", "openai")
```

## Features

- Drop-in replacement for OpenAI client
- Automatic token usage tracking
- Cost analysis for different time periods
- SQLite storage with zero configuration
- Thread-safe operations
- Minimal memory footprint

## License

MIT 
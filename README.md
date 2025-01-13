# Tokenator : Track and analyze LLM token usage and cost

Have you ever wondered about :
- How many tokens does your AI agent consume? 
- How much does it cost to do run a complex AI workflow with multiple LLM providers?
- How much money/tokens did you spend today on developing with LLMs?

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

Works with AsyncOpenAI and `streaming=True` as well!
Note : When streaming, don't forget to add `stream_options={"include_usage": True}` to the `create()` call!

### Cost Analysis

```python
from tokenator import usage

# Get usage for different time periods
usage.last_hour()
usage.last_day()
usage.last_week()
usage.last_month()

# Custom date range
usage.between("2024-03-01", "2024-03-15")

# Get usage for different LLM providers
usage.last_day("openai")
usage.last_day("anthropic")
usage.last_day("google")
```

### Example `usage` object

```python
print(cost.last_hour().model_dump_json(indent=4))
```

```json
{
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

### Anthropic

```python
from anthropic import Anthropic, AsyncAnthropic
from tokenator import tokenator_anthropic

anthropic_client = AsyncAnthropic(api_key="your-api-key")

# Wrap it with Tokenator
client = tokenator_anthropic(anthropic_client)

# Use it exactly like the Anthropic client
response = await client.messages.create(
    model="claude-3-5-haiku-20241022",
    messages=[{"role": "user", "content": "hello how are you"}],
    max_tokens=20,
)

print(response)

print(usage.last_execution().model_dump_json(indent=4))
"""
{
    "total_cost": 0.0001,
    "total_tokens": 23,
    "prompt_tokens": 10,
    "completion_tokens": 13,
    "providers": [
        {
            "total_cost": 0.0001,
            "total_tokens": 23,
            "prompt_tokens": 10,
            "completion_tokens": 13,
            "provider": "anthropic",
            "models": [
                {
                    "total_cost": 0.0004,
                    "total_tokens": 79,
                    "prompt_tokens": 52,
                    "completion_tokens": 27,
                    "model": "claude-3-5-haiku-20241022"
                }
            ]
        }
    ]
}
"""
```

### xAI

You can use xAI models through the `openai` SDK and track usage using `provider` parameter in `tokenator`.

```python
from openai import OpenAI
from tokenator import tokenator_openai

xai_client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )

# Wrap it with Tokenator
client = tokenator_openai(client, db_path=temp_db, provider="xai")

# Use it exactly like the OpenAI client but with xAI models
response = client.chat.completions.create(
    model="grok-2-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response)

print(usage.last_execution())
```

### Other AI model providers through openai SDKs

Today, a variety of AI companies have made their APIs compatible to the `openai` SDK.
You can track usage of any such AI models using `tokenator`'s `provider` parameter.

For example, let's see how we can track usage of `perplexity` tokens.

```python
from openai import OpenAI
from tokenator import tokenator_openai

xai_client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai"
        )

# Wrap it with Tokenator
client = tokenator_openai(client, db_path=temp_db, provider="perplexity")

# Use it exactly like the OpenAI client but with xAI models
response = client.chat.completions.create(
    model="grok-2-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response)

print(usage.last_execution())

print(usage.provider("perplexity"))
```

---

Most importantly, none of your data is ever sent to any server.

## License

MIT 
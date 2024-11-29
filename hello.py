from openai import OpenAI
import os
from dotenv import load_dotenv
from tokenator import OpenAIWrapper
from tokenator import cost
import logging

# Add logging config at the top after imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# More granular control
logging.getLogger("openai").setLevel(logging.WARN)
logging.getLogger("tokenator").setLevel(logging.DEBUG)

load_dotenv()
client = OpenAIWrapper(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
print("-"*100)
print(completion.usage)
print("-"*100)
print(cost.last_hour("openai").model_dump_json(indent=2))
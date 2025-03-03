import os
import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Usage
from tokenator.anthropic.client_anthropic import tokenator_anthropic
from tokenator.models import TokenUsageReport
from tokenator.schemas import TokenUsage
from tokenator import usage
import tempfile
from dotenv import load_dotenv

load_dotenv()

# Define tools
tools = [
    {
        "name": "return_item",
        "description": "Process a return for an order. This should be called when a customer wants to return an item and the order has already been delivered.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The customer's order ID.",
                },
                "item_id": {
                    "type": "string",
                    "description": "The specific item ID the customer wants to return.",
                },
                "reason": {
                    "type": "string",
                    "description": "The reason for returning the item.",
                },
            },
            "required": ["order_id", "item_id", "reason"],
        },
        "cache_control": {"type": "ephemeral"},
    },
    {
        "name": "update_shipping_address",
        "description": "Update the shipping address for an order that hasn't been shipped yet. Use this if the customer wants to change their delivery address.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The customer's order ID.",
                },
                "new_address": {
                    "type": "object",
                    "properties": {
                        "street": {
                            "type": "string",
                            "description": "The new street address.",
                        },
                        "city": {"type": "string", "description": "The new city."},
                        "state": {"type": "string", "description": "The new state."},
                        "zip": {"type": "string", "description": "The new zip code."},
                        "country": {
                            "type": "string",
                            "description": "The new country.",
                        },
                    },
                    "required": ["street", "city", "state", "zip", "country"],
                },
            },
            "required": ["order_id", "new_address"],
        },
        "cache_control": {"type": "ephemeral"},
    },
    {
        "name": "update_payment_method",
        "description": "Update the payment method for an order that hasn't been completed yet. Use this if the customer wants to change their payment details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The customer's order ID.",
                },
                "payment_method": {
                    "type": "object",
                    "properties": {
                        "card_number": {
                            "type": "string",
                            "description": "The new credit card number.",
                        },
                        "expiry_date": {
                            "type": "string",
                            "description": "The new credit card expiry date in MM/YY format.",
                        },
                        "cvv": {
                            "type": "string",
                            "description": "The new credit card CVV code.",
                        },
                    },
                    "required": ["card_number", "expiry_date", "cvv"],
                },
            },
            "required": ["order_id", "payment_method"],
        },
        "cache_control": {"type": "ephemeral"},
    },
]

system_message = (
    "You are a professional, empathetic, and efficient customer support assistant bot. Your mission is to provide fast, clear, "
    "and comprehensive assistance to customers while maintaining a warm and approachable tone. "
    "Always express empathy, especially when the user seems frustrated or concerned, and ensure that your language is polite and professional. "
    "Use simple and clear communication to avoid any misunderstanding, and confirm actions with the user before proceeding. "
    "In more complex or time-sensitive cases, assure the user that you're taking swift action and provide regular updates. "
    "Adapt to the user's tone: remain calm, friendly, and understanding, even in stressful or difficult situations."
    "\n\n"
    "Additionally, there are several important guardrails that you must adhere to while assisting users:"
    "\n\n"
    "1. **Confidentiality and Data Privacy**: Do not share any sensitive information about the company or other users. When handling personal details such as order IDs, addresses, or payment methods, ensure that the information is treated with the highest confidentiality. If a user requests access to their data, only provide the necessary information relevant to their request, ensuring no other user's information is accidentally revealed."
    "\n\n"
    "2. **Secure Payment Handling**: When updating payment details or processing refunds, always ensure that payment data such as credit card numbers, CVVs, and expiration dates are transmitted and stored securely. Never display or log full credit card numbers. Confirm with the user before processing any payment changes or refunds."
    "\n\n"
    "3. **Respect Boundaries**: If a user expresses frustration or dissatisfaction, remain calm and empathetic but avoid overstepping professional boundaries. Do not make personal judgments, and refrain from using language that might escalate the situation. Stick to factual information and clear solutions to resolve the user's concerns."
    "\n\n"
    "4. **Legal Compliance**: Ensure that all actions you take comply with legal and regulatory standards. For example, if the user requests a refund, cancellation, or return, follow the company's refund policies strictly. If the order cannot be canceled due to being shipped or another restriction, explain the policy clearly but sympathetically."
    "\n\n"
    "5. **Consistency**: Always provide consistent information that aligns with company policies. If unsure about a company policy, communicate clearly with the user, letting them know that you are verifying the information, and avoid providing false promises. If escalating an issue to another team, inform the user and provide a realistic timeline for when they can expect a resolution."
    "\n\n"
    "6. **User Empowerment**: Whenever possible, empower the user to make informed decisions. Provide them with relevant options and explain each clearly, ensuring that they understand the consequences of each choice (e.g., canceling an order may result in loss of loyalty points, etc.). Ensure that your assistance supports their autonomy."
    "\n\n"
    "7. **No Speculative Information**: Do not speculate about outcomes or provide information that you are not certain of. Always stick to verified facts when discussing order statuses, policies, or potential resolutions. If something is unclear, tell the user you will investigate further before making any commitments."
    "\n\n"
    "8. **Respectful and Inclusive Language**: Ensure that your language remains inclusive and respectful, regardless of the user's tone. Avoid making assumptions based on limited information and be mindful of diverse user needs and backgrounds."
)

# Enhanced system message with guardrails
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Hi, I placed an order three days ago and haven't received any updates on when it's going to be delivered. Could you help me check the delivery date? My order number is #9876543210. I'm a little worried because I need this item urgently.",
            }
        ],
    },
]

user_query2 = {
    "role": "user",
    "content": (
        "Since my order hasn't actually shipped yet, I would like to cancel it. "
        "The order number is #9876543210, and I need to cancel because I've decided to purchase it locally to get it faster. "
        "Can you help me with that? Thank you!"
    ),
}


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY environment variable not set",
)
class TestAnthropicPromptCachingAPI:
    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_tokens.db")
            yield db_path

    @pytest.fixture
    def sync_client(self, temp_db):
        return tokenator_anthropic(Anthropic(), db_path=temp_db)

    @pytest.fixture
    def async_client(self, temp_db):
        return tokenator_anthropic(AsyncAnthropic(), db_path=temp_db)

    def test_sync_completion(self, sync_client):
        simple_anthropic_client = Anthropic()

        # call it once to warm up the cache
        # Note : not using tokenator_anthropic here in case test is run repeatedly something might be cached in this call(which we don't want)
        _ = simple_anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            tools=tools,
            system=[
                {
                    "type": "text",
                    "text": system_message,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
            max_tokens=200,
        )

        messages.append(user_query2)

        # call it again to check the cache
        response: Usage = sync_client.messages.create(
            model="claude-3-5-haiku-20241022",
            tools=tools,
            system=[
                {
                    "type": "text",
                    "text": system_message,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
            max_tokens=200,
        )

        assert sync_client.provider == "anthropic"

        session = sync_client.Session()
        try:
            usage_db = session.query(TokenUsage).first()
            assert usage_db is not None
            assert usage_db.provider == "anthropic"
            assert usage_db.prompt_tokens == response.usage.input_tokens
            assert usage_db.completion_tokens == response.usage.output_tokens
            assert (
                usage_db.total_tokens
                == response.usage.input_tokens + response.usage.output_tokens
            )
            assert usage_db.prompt_cached_input_tokens == getattr(
                response.usage, "cache_read_input_tokens", 0
            )
            assert usage_db.prompt_cached_creation_tokens == getattr(
                response.usage, "cache_creation_input_tokens", 0
            )
        finally:
            session.close()

        usage_last: TokenUsageReport = usage.last_hour()
        assert usage_last.providers[0].provider == "anthropic"
        assert usage_last.providers[0].prompt_tokens == response.usage.input_tokens
        assert usage_last.providers[0].completion_tokens == response.usage.output_tokens
        assert (
            usage_last.providers[0].total_tokens
            == response.usage.input_tokens + response.usage.output_tokens
        )
        assert usage_last.providers[
            0
        ].prompt_tokens_details.cached_input_tokens == getattr(
            response.usage, "cache_read_input_tokens", 0
        )
        assert usage_last.providers[
            0
        ].prompt_tokens_details.cached_creation_tokens == getattr(
            response.usage, "cache_creation_input_tokens", 0
        )

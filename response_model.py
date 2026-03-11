"""
response_model.py
-----------------
Generates customer support responses using the NEW Google GenAI SDK
(google-genai) with Gemini 2.0 Flash — v1 API, no v1beta issues!
"""

import re
from google import genai
from google.genai import types

# ── Model config ───────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-pro"

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a friendly and professional Customer Support Agent for an e-commerce platform.
You help customers with questions about their orders using the order information provided to you.

Guidelines:
- Be concise, polite, and empathetic.
- Answer ONLY based on the order context provided. Do NOT invent or assume data.
- If an order is delayed or has issues, express empathy and offer constructive next steps.
- Format monetary amounts clearly (e.g. ₹1,149 INR).
- If the context does not contain enough information to answer, say so honestly.
- For tracking questions, always mention the tracking number and courier service.
- For refund/cancellation questions, refer to the payment status and order status.
- For warranty questions, mention the warranty period if available.
- Keep responses under 200 words unless the question requires more detail.
"""

# Global client
_client: genai.Client | None = None


def set_client(client: genai.Client) -> None:
    """Set the shared Gemini client (called from RAG-Application.py)."""
    global _client
    _client = client


def get_client() -> genai.Client:
    if _client is None:
        raise RuntimeError("Gemini client not set. Call set_client() first.")
    return _client


# ── Intent detection ───────────────────────────────────────────────────────────
def detect_order_id(query: str) -> str | None:
    """Extract an order ID like ORD1010 from the query."""
    match = re.search(r"\bORD\d+\b", query, re.IGNORECASE)
    return match.group(0).upper() if match else None


def detect_intent(query: str) -> str:
    """Classify the user's intent."""
    q = query.lower()
    if any(k in q for k in ["track", "tracking", "courier", "shipping"]):
        return "tracking"
    if any(k in q for k in ["pay", "payment", "transaction", "invoice", "money"]):
        return "payment"
    if any(k in q for k in ["deliver", "delivery", "arrive", "when will"]):
        return "delivery"
    if any(k in q for k in ["return", "cancel", "refund", "replace", "exchange"]):
        return "return_refund"
    if any(k in q for k in ["warranty", "guarantee", "repair"]):
        return "warranty"
    if any(k in q for k in ["product", "item", "brand", "category", "price", "discount"]):
        return "product_info"
    if any(k in q for k in ["status", "update", "order"]):
        return "order_status"
    return "general"


# ── Context builder ────────────────────────────────────────────────────────────
def build_context(retrieved_orders: list[dict]) -> str:
    """Format retrieved orders into a clean context block for Gemini."""
    if not retrieved_orders:
        return "No matching order records found."
    lines = []
    for idx, order in enumerate(retrieved_orders, 1):
        o = {k: v for k, v in order.items() if not k.startswith("_")}
        lines.append(f"--- Order Record {idx} ---")
        for key, val in o.items():
            lines.append(f"  {key.replace('_', ' ').title()}: {val if val is not None else 'N/A'}")
        lines.append("")
    return "\n".join(lines)


# ── Response generation ────────────────────────────────────────────────────────
def generate_response(
    user_query: str,
    retrieved_orders: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    """Generate a grounded customer support response using Gemini."""
    client   = get_client()
    context  = build_context(retrieved_orders)
    intent   = detect_intent(user_query)
    order_id = detect_order_id(user_query)

    grounded_prompt = f"""ORDER CONTEXT (retrieved from database):
{context}

Detected Intent: {intent}
{"Detected Order ID: " + order_id if order_id else ""}

CUSTOMER QUESTION:
{user_query}

Please answer the customer's question based solely on the order context above."""

    # Build conversation history
    history = []
    if chat_history:
        for turn in chat_history[:-1]:
            role = "user" if turn["role"] == "user" else "model"
            history.append(
                types.Content(role=role, parts=[types.Part(text=turn["content"])])
            )

    # Add system prompt + current question
    full_contents = [
        types.Content(role="user", parts=[types.Part(text=grounded_prompt)])
    ]

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            top_p=0.9,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()


# ── Fallback response ──────────────────────────────────────────────────────────
def generate_fallback_response(user_query: str) -> str:
    """Polite fallback when no relevant orders are found."""
    client = get_client()
    prompt = (
        f"The customer asked: '{user_query}'\n\n"
        "No matching order records were found in the database. "
        "Please respond politely, acknowledge the query, and advise the customer "
        "to double-check their Order ID or contact support with more details."
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=512,
        ),
    )
    return response.text.strip()

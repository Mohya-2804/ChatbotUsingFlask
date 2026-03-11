"""
embeddings.py
-------------
Handles text embedding generation using the NEW Google GenAI SDK
(google-genai) which uses the v1 API — fixing the v1beta 404 errors.

Install: pip install google-genai
"""

import json
from google import genai
from google.genai import types

# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-004"   # v1 supported model
EMBEDDING_DIM   = 768
MAX_BATCH_SIZE  = 5

# Global client (set on init)
_client: genai.Client | None = None


# ── Initialisation ─────────────────────────────────────────────────────────────
def init_gemini(api_key: str) -> None:
    """
    Initialise the Gemini GenAI client with your API key.
    Get your key from: https://aistudio.google.com/app/apikey
    """
    global _client
    _client = genai.Client(api_key=api_key)
    print("[Gemini] ✅ GenAI client initialised successfully.")


def get_client() -> genai.Client:
    if _client is None:
        raise RuntimeError("Gemini client not initialised. Call init_gemini() first.")
    return _client


# ── Order → Text conversion ────────────────────────────────────────────────────
def order_to_text(order: dict) -> str:
    """Convert a customer order dict into a rich human-readable string."""
    return (
        f"Order ID: {order.get('order_id', 'N/A')}. "
        f"Platform: {order.get('platform', 'N/A')}. "
        f"Product: {order.get('product_name', 'N/A')} "
        f"({order.get('product_category', 'N/A')}) by {order.get('brand', 'N/A')}. "
        f"Seller: {order.get('seller_name', 'N/A')}. "
        f"Quantity: {order.get('quantity', 'N/A')}. "
        f"Price: {order.get('final_price', 'N/A')} {order.get('currency', 'INR')} "
        f"(original {order.get('price', 'N/A')}, discount {order.get('discount', 0)}). "
        f"Purchase Date: {order.get('date_of_purchase', 'N/A')}. "
        f"Estimated Delivery: {order.get('estimated_delivery_date', 'N/A')}. "
        f"Actual Delivery: {order.get('delivery_date', 'Not yet delivered')}. "
        f"Order Status: {order.get('order_status', 'N/A')}. "
        f"Courier: {order.get('courier_service', 'N/A')}, "
        f"Tracking: {order.get('tracking_number', 'N/A')}. "
        f"Payment: {order.get('payment_method', 'N/A')} — "
        f"Status: {order.get('payment_status', 'N/A')}, "
        f"Transaction ID: {order.get('transaction_id', 'N/A')}, "
        f"Invoice: {order.get('invoice_number', 'N/A')}. "
        f"Warranty: {order.get('warranty_period', 'N/A')}. "
        f"Return Window: {order.get('return_window_days', 'N/A')} days. "
        f"Return Requested: {order.get('return_requested', False)}. "
        f"Refund Status: {order.get('refund_status', 'N/A')}."
    )


# ── Embedding helpers ──────────────────────────────────────────────────────────
def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document strings for indexing into Qdrant.
    Processes in batches of MAX_BATCH_SIZE.
    """
    client      = get_client()
    all_vectors = []

    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch    = texts[i: i + MAX_BATCH_SIZE]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )
        for emb in response.embeddings:
            all_vectors.append(emb.values)
        print(f"[Embeddings] Batch {i // MAX_BATCH_SIZE + 1} done ({len(batch)} docs)")

    return all_vectors


def embed_query(query: str) -> list[float]:
    """
    Embed a single user query for semantic search.
    Returns a float vector of length EMBEDDING_DIM (768).
    """
    client   = get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return response.embeddings[0].values


# ── Load & embed orders ────────────────────────────────────────────────────────
def load_and_embed_orders(json_path: str) -> tuple[list[dict], list[list[float]]]:
    """
    Load Customerdetails.json, convert every order to text, and embed.

    Returns:
        (orders, vectors) — parallel lists ready for Qdrant indexing.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both plain list and {"orders": [...]} formats
    orders: list[dict] = data if isinstance(data, list) else data.get("orders", [])
    print(f"[Embeddings] Loaded {len(orders)} orders from '{json_path}'")

    texts   = [order_to_text(o) for o in orders]
    vectors = embed_documents(texts)

    print(f"[Embeddings] ✅ All {len(vectors)} order embeddings ready.")
    return orders, vectors

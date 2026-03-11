"""
qdrant.py
---------
All Qdrant vector-database operations:
  • Connect to Qdrant Cloud
  • Create / reset collection
  • Index order embeddings
  • Semantic search
  • Exact order lookup by order_id
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from embeddings import EMBEDDING_DIM

# ── Configuration ──────────────────────────────────────────────────────────────
COLLECTION_NAME = "customer_orders"

# ┌─────────────────────────────────────────────────────────────┐
# │          🔑  PASTE YOUR QDRANT CLOUD CREDENTIALS HERE       │
# └─────────────────────────────────────────────────────────────┘
QDRANT_URL           = "https://95df3407-4971-4ac5-bf3f-50fcbf6cfeb6.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY       = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.BegD8NOMpNf22O_Ss_CZFhwZHSAUSHi3vnhy86M0bS0"                   # 👈 Your Qdrant API Key


# ── Client factory ─────────────────────────────────────────────────────────────
def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant Cloud using URL and API key."""
    print(f"[Qdrant] Connecting to Qdrant Cloud → {QDRANT_URL}")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ── Collection management ──────────────────────────────────────────────────────
def create_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create the orders collection. Pass recreate=True to wipe & rebuild."""
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            client.delete_collection(COLLECTION_NAME)
            print(f"[Qdrant] Deleted existing collection '{COLLECTION_NAME}'")
        else:
            print(f"[Qdrant] Collection '{COLLECTION_NAME}' already exists — skipping.")
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"[Qdrant] ✅ Collection '{COLLECTION_NAME}' created (dim={EMBEDDING_DIM})")


# ── Indexing ───────────────────────────────────────────────────────────────────
def index_orders(
    client: QdrantClient,
    orders: list[dict],
    vectors: list[list[float]],
) -> None:
    """Upsert all orders with their vectors into Qdrant."""
    if len(orders) != len(vectors):
        raise ValueError("orders and vectors must have the same length.")

    points = [
        PointStruct(id=i, vector=vectors[i], payload=orders[i])
        for i in range(len(orders))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[Qdrant] ✅ Indexed {len(points)} orders into '{COLLECTION_NAME}'")


# ── Semantic search ────────────────────────────────────────────────────────────
def semantic_search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 20,
    score_threshold: float = 0.3,
) -> list[dict]:
    """Find the most semantically similar orders to a query vector."""
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )
    matched = []
    for hit in results.points:
        order = hit.payload.copy()
        order["_score"] = round(hit.score, 4)
        matched.append(order)

    print(f"[Qdrant] Semantic search returned {len(matched)} result(s)")
    return matched


# ── Exact order-ID lookup ──────────────────────────────────────────────────────
def lookup_by_order_id(client: QdrantClient, order_id: str) -> dict | None:
    """Exact match lookup by order_id payload field."""
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="order_id", match=MatchValue(value=order_id))]
        ),
        limit=1,
        with_payload=True,
    )
    if results:
        print(f"[Qdrant] Exact match found for order_id='{order_id}'")
        return results[0].payload
    print(f"[Qdrant] No exact match for order_id='{order_id}'")
    return None


# ── Collection stats ───────────────────────────────────────────────────────────
def collection_info(client: QdrantClient) -> dict:
    """Return basic stats about the orders collection."""
    info  = client.get_collection(COLLECTION_NAME)
    total = (
        info.points_count
        if hasattr(info, "points_count")
        else getattr(info, "vectors_count", 0)
    )
    return {
        "name":          COLLECTION_NAME,
        "total_vectors": total,
        "status":        str(info.status),
    }

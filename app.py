"""
app.py
------
Customer Support FAQ Bot — Flask Backend
Data is indexed ONCE and persists in Qdrant Cloud.
Use "Build / Rebuild Index" button to re-index manually.
"""

from flask import Flask, render_template, request, jsonify
from google import genai
import os

from embeddings import init_gemini, embed_query, load_and_embed_orders
from qdrant import (
    get_qdrant_client,
    create_collection,
    index_orders,
    semantic_search,
    lookup_by_order_id,
    collection_info,
)
from response_model import (
    set_client,
    generate_response,
    generate_fallback_response,
    detect_order_id,
    detect_intent,
)

# ══════════════════════════════════════════════════════════════════════════════
# 🔑  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
GEMINI_API_KEY   = "AIzaSyDvS-x1K-krZTGzj46_9td6N9suB5nJac4"         # 👈 paste your Gemini API key
ORDERS_JSON_PATH = "C:\\Users\\gudla\\Downloads\\chatbotwithflask\\Customerdetails.json"
TOP_K_RESULTS    = 20
SCORE_THRESHOLD  = 0.30

# ── Flask app ──────────────────────────────────────────────────────────────────
app            = Flask(__name__)
app.secret_key = "customer-support-secret"

# ── Global state ───────────────────────────────────────────────────────────────
qdrant_client  = None
total_orders   = 0
pipeline_ready = False
index_built    = False     # ← tracks if index was already built this session


def init_clients():
    """Initialise Gemini + Qdrant clients (no indexing yet)."""
    global qdrant_client, pipeline_ready

    print("[App] 🔧 Initialising clients...")
    init_gemini(GEMINI_API_KEY)

    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    set_client(gemini_client)

    qdrant_client = get_qdrant_client()
    pipeline_ready = True
    print("[App] ✅ Clients ready.")


def build_index(force_rebuild: bool = False):
    """
    Build or rebuild the Qdrant index from Customerdetails.json.
    If force_rebuild=False and collection already exists → skip indexing.
    """
    global total_orders, index_built

    try:
        # Check if collection already has data
        info = collection_info(qdrant_client)
        already_indexed = (info.get("total_vectors", 0) or 0) > 0
    except Exception:
        already_indexed = False

    if already_indexed and not force_rebuild:
        try:
            info         = collection_info(qdrant_client)
            total_orders = info.get("total_vectors", 0) or 0
            index_built  = True
            print(f"[App] ✅ Collection already has {total_orders} orders — skipping re-index.")
            return {"status": "skipped", "total_orders": total_orders}
        except Exception:
            pass

    # Create / recreate collection and index
    print("[App] 📦 Building index...")
    create_collection(qdrant_client, recreate=True)
    orders, vectors = load_and_embed_orders(ORDERS_JSON_PATH)
    index_orders(qdrant_client, orders, vectors)
    total_orders = len(orders)
    index_built  = True
    print(f"[App] ✅ Indexed {total_orders} orders.")
    return {"status": "built", "total_orders": total_orders}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main chat page."""
    return render_template("index.html")


@app.route("/build_index", methods=["POST"])
def api_build_index():
    """
    Called by the 'Build / Rebuild Index' button.
    force=true  → always re-index
    force=false → skip if already indexed
    """
    data  = request.get_json() or {}
    force = data.get("force", False)
    try:
        result = build_index(force_rebuild=force)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status():
    """Return current pipeline and index status."""
    indexed = 0
    try:
        info    = collection_info(qdrant_client)
        indexed = info.get("total_vectors", 0) or 0
    except Exception:
        pass
    return jsonify({
        "pipeline_ready": pipeline_ready,
        "index_built":    index_built,
        "total_orders":   indexed,
    })


@app.route("/chat", methods=["POST"])
def chat():
    """Handle a chat message and return AI response."""
    if not pipeline_ready:
        return jsonify({"error": "Pipeline not ready."}), 503
    if not index_built:
        return jsonify({"error": "Index not built yet. Click 'Build / Rebuild Index' first."}), 400

    data            = request.get_json()
    prompt          = data.get("message", "").strip()
    top_k           = int(data.get("top_k",           TOP_K_RESULTS))
    score_thresh    = float(data.get("score_thresh",   SCORE_THRESHOLD))
    category_filter = data.get("category_filter",      "All Categories")
    status_filter   = data.get("status_filter",        "All Statuses")
    chat_history    = data.get("chat_history",         [])

    if not prompt:
        return jsonify({"error": "Empty message."}), 400

    try:
        # Build augmented prompt with filters
        filter_hint = ""
        if category_filter != "All Categories": filter_hint += f" category:{category_filter}"
        if status_filter   != "All Statuses":   filter_hint += f" status:{status_filter}"
        augmented = prompt + filter_hint

        # Step 1: Exact order ID lookup
        retrieved = []
        oid = detect_order_id(augmented)
        if oid:
            exact = lookup_by_order_id(qdrant_client, oid)
            if exact:
                retrieved = [exact]

        # Step 2: Semantic search
        qvec = embed_query(augmented)
        hits = semantic_search(qdrant_client, qvec, top_k=top_k, score_threshold=score_thresh)

        seen = {o.get("order_id") for o in retrieved}
        for h in hits:
            if h.get("order_id") not in seen:
                retrieved.append(h)
                seen.add(h.get("order_id"))

        # Apply sidebar filters
        if category_filter != "All Categories":
            retrieved = [o for o in retrieved if o.get("product_category") == category_filter]
        if status_filter != "All Statuses":
            retrieved = [o for o in retrieved if o.get("order_status") == status_filter]

        retrieved = retrieved[:top_k]

        # Step 3: Generate response
        answer = (
            generate_response(prompt, retrieved, chat_history)
            if retrieved else generate_fallback_response(prompt)
        )

        intent = detect_intent(prompt)

        # Clean metadata
        clean = [{k: v for k, v in o.items() if k != "_score"} for o in retrieved]

        return jsonify({
            "answer":           answer,
            "intent":           intent.replace("_", " ").title(),
            "retrieved_count":  len(retrieved),
            "retrieved_orders": clean,
        })

    except Exception as e:
        print(f"[App] ❌ {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear():
    return jsonify({"status": "cleared"})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_clients()          # init Gemini + Qdrant clients
    build_index(force_rebuild=False)   # index ONLY if not already indexed
    app.run(debug=True, host="0.0.0.0", port=5000)

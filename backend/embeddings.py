import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

# -- Config --------------------------------------------------------
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "railway_knowledge"

# Dense embeddings — semantic search
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Reranker — scores combined results for final ranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -- Railway domain documents ----------------------------------------
DOCUMENTS = [
    # BFI knowledge
    "BFI stands for Ballast Fouling Index. It measures ballast contamination "
    "in railway track segments. BFI above 0.40 indicates fouled ballast. "
    "BFI above 0.8 is WARNING status. BFI above 1.0 is CRITICAL and requires "
    "immediate maintenance to prevent derailment risk.",

    "BFI values are calculated from LIDAR point cloud scans processed by a "
    "PointNet model that identifies centerline points and top-of-rail points, "
    "then calculates ballast volume to derive the fouling index per segment.",

    "Track segment status is derived from BFI score: OK means BFI below 0.8, "
    "WARNING means BFI between 0.8 and 1.0, CRITICAL means BFI above 1.0 "
    "requiring scheduled maintenance action.",

    # Maintenance knowledge
    "TAMPING is a maintenance procedure that lifts and re-levels track while "
    "packing ballast under sleepers. Most common action for high BFI segments. "
    "Typical cost between $5,000 and $50,000 per segment.",

    "BALLAST_CLEANING removes fouled ballast and replaces with clean crushed "
    "stone. Used for severely fouled segments with BFI above 1.0. "
    "Cost ranges from $30,000 to $120,000 per segment.",

    "UNDERCUTTING is deep cleaning that removes and cleans ballast below tie "
    "level. Most expensive maintenance type, used for chronically fouled segments.",

    "SPOT_REPAIR addresses localized ballast deficiencies in specific areas "
    "rather than the full segment. Used for WARNING status segments.",

    "FULL_REPLACEMENT involves complete removal and replacement of ballast, "
    "ties, and sometimes rail. Reserved for the most degraded segments.",

    # Geometry survey knowledge
    "Geometry surveys collect precise track measurements including centerline "
    "coordinates, top-of-rail heights on left and right rails, track gauge in "
    "millimeters, and cross-level measurements. Standard gauge is 1435mm. "
    "Deviation beyond plus or minus 3mm requires immediate attention.",

    "Cross-level measurements indicate height difference between left and right "
    "rails. Values beyond plus or minus 3mm indicate geometry issues that may "
    "cause passenger discomfort or derailment risk.",

    # Asset knowledge
    "Railway bridges require inspection every 180 days. Condition is rated "
    "GOOD, FAIR, or POOR. POOR condition requires engineering assessment "
    "and may require speed restrictions or closure.",

    "Railway tunnels require quarterly inspections. POOR condition tunnels "
    "may require reduced speed limits or temporary closure for repairs.",

    # Subdivision knowledge
    "The Plainview subdivision covers track in the Texas Panhandle region. "
    "High-traffic freight corridor for grain and agricultural shipments.",

    "The Amarillo subdivision serves the Amarillo Texas area connecting to "
    "major classification yards. High fouling rates due to coal dust.",

    "The Clovis subdivision in New Mexico connects major interchange points. "
    "Track geometry issues common due to soil expansion in the region.",

    "The Belen subdivision serves the Belen New Mexico hub, a major "
    "classification yard with mixed freight and intermodal traffic.",

    "The Needles subdivision crosses desert terrain in California. Extreme "
    "temperature variations cause rail expansion and accelerated ballast "
    "degradation.",

    # Schema knowledge
    "The track_segments table contains one row per segment: segment_id, "
    "track_id, mile_post_start, mile_post_end, bfi_value, ballast_volume, "
    "status (OK/WARNING/CRITICAL), survey_date, subdivision.",

    "The track_assets table contains infrastructure assets (bridges and "
    "tunnels): asset_id, asset_type, asset_name, segment_id, mile_post, "
    "subdivision, inspection_date, condition (GOOD/FAIR/POOR).",

    "The geometry_surveys table contains track geometry data: survey_id, "
    "segment_id, survey_date, centerline_x, centerline_y, top_of_rail_l, "
    "top_of_rail_r, gauge_mm, cross_level_mm.",

    "The maintenance_log table tracks all maintenance work: log_id, "
    "segment_id, work_type, crew_size, cost_usd, work_date, "
    "completed (1=done, 0=pending).",
]

# -- ChromaDB setup ----------------------------------------
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

def build_knowledge_base():
    collection = get_collection()
    if collection.count() == 0:
        ids = [f"doc_{i}" for i in range(len(DOCUMENTS))]
        collection.add(documents=DOCUMENTS, ids=ids)
        print(f"✅ Knowledge base built: {len(DOCUMENTS)} documents indexed")
    else:
        print(f"✅ Knowledge base ready: {collection.count()} documents")
    return collection

# -- BM25 keyword search -----------------------------------------
def _build_bm25():
    """Tokenize documents for BM25 keyword search."""
    tokenized = [doc.lower().split() for doc in DOCUMENTS]
    return BM25Okapi(tokenized)

def _keyword_search(query: str, top_k: int = 10) -> list:
    """BM25 keyword search — returns (doc_index, score) pairs."""
    bm25 = _build_bm25()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    # Get top_k indices sorted by score
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]

# -- Dense semantic search -----------------------------------------
def _semantic_search(query: str, top_k: int = 10) -> list:
    """ChromaDB dense vector search — returns (doc_index, score) pairs."""
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count())
    )
    if not results or not results["ids"]:
        return []

    pairs = []
    for i, doc_id in enumerate(results["ids"][0]):
        idx = int(doc_id.replace("doc_", ""))
        # ChromaDB returns distances — convert to similarity score
        distance = results["distances"][0][i]
        score = 1 / (1 + distance)
        pairs.append((idx, score))
    return pairs

# -- Hybrid search with reranking --------------------------------
def retrieve_context(query: str, n_results: int = 3) -> str:
    """
    Hybrid search pipeline:
    1. BM25 keyword search → top 10 candidates
    2. Dense semantic search → top 10 candidates
    3. Merge and deduplicate candidates
    4. Reranker scores all candidates
    5. Return top n_results documents
    """

    # Step 1 — BM25 keyword search
    keyword_results = _keyword_search(query, top_k=10)
    keyword_indices = {idx for idx, _ in keyword_results}

    # Step 2 — Dense semantic search
    semantic_results = _semantic_search(query, top_k=10)
    semantic_indices = {idx for idx, _ in semantic_results}

    # Step 3 — Merge and deduplicate
    # Union of both result sets gives us diverse candidates
    all_indices = list(keyword_indices | semantic_indices)
    candidate_docs = [DOCUMENTS[i] for i in all_indices]

    if not candidate_docs:
        return ""

    # Step 4 — Reranker scores each candidate against the query
    # CrossEncoder takes (query, document) pairs and scores relevance
    pairs = [[query, doc] for doc in candidate_docs]
    rerank_scores = reranker.predict(pairs)

    # Step 5 — Sort by reranker score and take top n_results
    scored = sorted(
        zip(candidate_docs, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )
    top_docs = [doc for doc, score in scored[:n_results]]

    return "\n\n".join([f"- {doc}" for doc in top_docs])

# -- Test --------------------------------------------------------
if __name__ == "__main__":
    build_knowledge_base()

    test_questions = [
        # Tests semantic search — no exact keyword match
        "When is track ballast considered dangerous?",
        # Tests keyword search — exact term match
        "TAMPING cost and procedure",
        # Tests hybrid — needs both meaning and keywords
        "What happens when BFI exceeds 1.0 in Clovis subdivision?",
    ]

    print("\n--Hybrid Search Results -------------------------------\n")
    for q in test_questions:
        print(f"❓ {q}")
        context = retrieve_context(q, n_results=2)
        print(f"📚 Retrieved:\n{context}\n")
        print("─" * 50)
"""
Hybrid retriever: Pinecone (semantic) + local BM25 (keyword) via
reciprocal rank fusion, deduplicated by answer. BM25 stays local since
Pinecone doesn't support keyword search natively.
"""

import os
import pickle
from pathlib import Path

import numpy as np
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_STORE_DIR = Path("rag/bm25_store")

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "anchor-logistics-kb")

TOP_K = 4
CANDIDATE_POOL = 30       # how many candidates each method contributes before fusion
RERANK_POOL = 15          # how many fused candidates get passed to the reranker
COMPANY_PROFILE_BOOST = 2.5

# Cross-encoder scores are raw logits, not 0-1 probabilities.
RELEVANCE_THRESHOLD = -6.0


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_MODEL)

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = pc.Index(PINECONE_INDEX_NAME)

        # Local chunk store, aligned by position with Pinecone ids
        # ("0", "1", ...) -- built by ingest.py. Used both for BM25 and as
        # the source of chunk text/answer during reranking, so reranking
        # doesn't depend on what Pinecone happens to return in metadata.
        with open(BM25_STORE_DIR / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        tokenized_corpus = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = TOP_K, debug: bool = False) -> list[dict]:
        # Semantic search via Pinecone
        query_emb = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        pinecone_response = self.index.query(
            vector=query_emb, top_k=CANDIDATE_POOL, include_metadata=False
        )
        vector_hits = [int(match["id"]) for match in pinecone_response["matches"]]

        # Keyword search (unchanged, local)
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_hits = np.argsort(bm25_scores)[::-1][:CANDIDATE_POOL]

        # Reciprocal rank fusion (unchanged)
        fused_scores: dict[int, float] = {}
        for rank, idx in enumerate(vector_hits):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (rank + 60)
        for rank, idx in enumerate(bm25_hits):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (rank + 60)

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_idxs = [int(idx) for idx, _ in ranked[:RERANK_POOL]]

        # Cross-encoder reranking (unchanged)
        pairs = [(query, self.chunks[idx]["text"]) for idx in candidate_idxs]
        rerank_scores = self.reranker.predict(pairs)

        scored = []
        for idx, raw_score in zip(candidate_idxs, rerank_scores):
            chunk = self.chunks[idx]
            boosted_score = raw_score + COMPANY_PROFILE_BOOST if chunk["source"] == "company_profile" else raw_score
            scored.append((chunk, raw_score, boosted_score))

        scored.sort(key=lambda x: x[2], reverse=True)

        if debug:
            print("\n--- raw rerank scores (top 8) ---")
            for chunk, raw_score, _ in scored[:8]:
                print(f"  {raw_score:+.2f}  ({chunk['source']}/{chunk['intent']})  {chunk['text'][:60]}")

        # De-duplicate by answer text, and drop anything below the relevance
        # threshold -- a low raw score means "best of a bad lot," not "relevant."
        results = []
        seen_answers = set()
        for chunk, raw_score, _boosted_score in scored:
            if raw_score < RELEVANCE_THRESHOLD:
                continue
            if chunk["answer"] in seen_answers:
                continue
            seen_answers.add(chunk["answer"])
            results.append(chunk)
            if len(results) >= k:
                break

        return results


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Same reasoning as ingest.py: this file is imported (not run directly)
    # by the app, where main.py already calls load_dotenv(). Running this
    # module standalone for debugging needs its own call.
    load_dotenv()

    retriever = Retriever()
    while True:
        query = input("\nAsk a question ('q' to quit): ")
        if query.lower() == "q":
            break
        results = retriever.search(query, debug=True)
        if not results:
            print("No relevant results found — this is where the fallback response kicks in.")
            continue
        for i, result in enumerate(results, start=1):
            print(f"\n[{i}] ({result['source']} / {result['intent']})")
            print(result["answer"][:300])
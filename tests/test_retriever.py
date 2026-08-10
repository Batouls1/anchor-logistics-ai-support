"""
Tests for retriever.py's fusion, dedup, and threshold logic against a
fixture corpus. Embedding model, cross-encoder, and Pinecone are mocked.
"""

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.retriever import RELEVANCE_THRESHOLD


def make_chunk(text, answer, source="bitext", intent="test_intent"):
    return {
        "text": text,
        "answer": answer,
        "category": "TEST",
        "intent": intent,
        "source": source,
    }


@pytest.fixture
def sample_chunks():
    # Index positions double as Pinecone ids ("0", "1", ...), matching how
    # ingest.py assigns them.
    return [
        make_chunk("how do i track my order", "Track via Order History."),           # 0
        make_chunk("how do i cancel my order", "Cancel within 1 hour."),              # 1
        make_chunk("what payment methods do you accept", "Visa, Mastercard, COD."),   # 2
        make_chunk("delivery to dubai", "We ship to Dubai.", source="company_profile"),  # 3
        make_chunk("duplicate answer entry", "Track via Order History."),             # 4 (same answer as 0)
    ]


def build_retriever(sample_chunks, tmp_path, pinecone_matches, rerank_scores):
    """
    Builds a Retriever with every networked/heavy dependency mocked:
    - SentenceTransformer.encode -> a fixed dummy embedding
    - Pinecone().Index().query -> the `pinecone_matches` passed in
    - CrossEncoder.predict -> the `rerank_scores` passed in, aligned by
      RERANK_POOL candidate order (tests below keep candidate counts small
      and matches/scores intentionally paired so this stays predictable)
    - chunks.pkl -> `sample_chunks`, so BM25 runs for real against known text
    """
    vector_store_dir = tmp_path / "vector_store"
    vector_store_dir.mkdir()
    with open(vector_store_dir / "chunks.pkl", "wb") as f:
        pickle.dump(sample_chunks, f)

    with patch("rag.retriever.BM25_STORE_DIR", vector_store_dir), \
         patch("rag.retriever.SentenceTransformer") as mock_st, \
         patch("rag.retriever.CrossEncoder") as mock_ce, \
         patch("rag.retriever.Pinecone") as mock_pinecone_cls, \
         patch.dict("os.environ", {"PINECONE_API_KEY": "test-key"}):

        mock_st.return_value.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        mock_index = MagicMock()
        mock_index.query.return_value = {"matches": pinecone_matches}
        mock_pinecone_cls.return_value.Index.return_value = mock_index

        mock_ce.return_value.predict.return_value = rerank_scores

        from rag.retriever import Retriever  # imported here so the patches above apply
        retriever = Retriever()

    return retriever, mock_index


def test_semantic_hits_are_fetched_from_pinecone(sample_chunks, tmp_path):
    pinecone_matches = [{"id": "0", "score": 0.9}, {"id": "2", "score": 0.5}]
    rerank_scores = [5.0, 4.0, 3.0, 2.0, 1.0]

    retriever, mock_index = build_retriever(sample_chunks, tmp_path, pinecone_matches, rerank_scores)

    results = retriever.search("how do i track my order", k=3)

    mock_index.query.assert_called_once()
    called_kwargs = mock_index.query.call_args.kwargs
    assert called_kwargs["top_k"] == 30  # CANDIDATE_POOL
    assert len(results) > 0
    assert results[0]["answer"] == "Track via Order History."


def test_deduplicates_by_answer_text(sample_chunks, tmp_path):
    pinecone_matches = [{"id": "0", "score": 0.9}, {"id": "4", "score": 0.85}]
    # idx 0 and idx 4 share the same answer text -- both score highly, but
    # only one should survive dedup.
    rerank_scores = [5.0, 4.0, 3.0, 2.0, 4.5]

    retriever, _ = build_retriever(sample_chunks, tmp_path, pinecone_matches, rerank_scores)

    results = retriever.search("track my order", k=5)

    answers = [r["answer"] for r in results]
    assert answers.count("Track via Order History.") == 1


def test_relevance_threshold_drops_low_scoring_candidates(sample_chunks, tmp_path):
    pinecone_matches = [{"id": str(i), "score": 0.5} for i in range(5)]
    # Every candidate scores below RELEVANCE_THRESHOLD -- nothing should
    # come back, even though Pinecone/BM25 both returned hits.
    rerank_scores = [RELEVANCE_THRESHOLD - 1] * len(sample_chunks)

    retriever, _ = build_retriever(sample_chunks, tmp_path, pinecone_matches, rerank_scores)

    results = retriever.search("irrelevant gibberish query", k=4)

    assert results == []


def test_company_profile_boost_affects_ordering_not_the_threshold_check(sample_chunks, tmp_path):
    """
    COMPANY_PROFILE_BOOST is applied to the sort score only. A
    company_profile chunk with a middling raw score can outrank a bitext
    chunk with a slightly better raw score -- but the threshold check
    itself always uses the unboosted raw score.
    """
    pinecone_matches = [{"id": "3", "score": 0.6}, {"id": "1", "score": 0.6}]
    # idx 3 (company_profile) scores lower on raw score than idx 1 (bitext),
    # but both clear the threshold -- the boost should still let idx 3 win.
    rerank_scores = [1.0, 1.2, 1.0, RELEVANCE_THRESHOLD + 0.5, 1.0]

    retriever, _ = build_retriever(sample_chunks, tmp_path, pinecone_matches, rerank_scores)

    results = retriever.search("delivery to dubai", k=5)

    result_answers = [r["answer"] for r in results]
    assert result_answers.index("We ship to Dubai.") < result_answers.index("Cancel within 1 hour.")
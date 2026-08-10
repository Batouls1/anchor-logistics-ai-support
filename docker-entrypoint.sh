#!/bin/sh
set -e

# rag/bm25_store is a named volume (see docker-compose.yml), so this only
# actually runs against a genuinely fresh volume -- not on every restart.
# ingest.py takes a few minutes (embedding + Pinecone upsert); repeating
# that on every container start would be pure waste.
if [ ! -f "rag/bm25_store/chunks.pkl" ]; then
    echo "No local BM25 store found -- running ingest.py once..."
    python rag/ingest.py
fi

exec uvicorn main:app --host 0.0.0.0 --port 8000
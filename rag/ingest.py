"""
Builds the Pinecone index from the bitext dataset and company_profile.md,
and saves a matching local pickle for BM25 (Pinecone doesn't do keyword
search). Vector ids are just chunk position, keeping both stores aligned.
Safe to re-run -- upserts overwrite by id.
"""

import json
import os
import re
import pickle
from pathlib import Path

from dotenv import load_dotenv

# ingest.py is run as a standalone script (not through main.py), so
# nothing else loads .env for this process -- without this, os.environ
# is just whatever the shell happens to have, and PINECONE_API_KEY isn't
# in it.
load_dotenv()

import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 output size -- must match the index dim

DATA_DIR = Path("data")
BM25_STORE_DIR = Path("rag/bm25_store")  # local BM25 corpus + chunk metadata

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "anchor-logistics-kb")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")

UPSERT_BATCH_SIZE = 100


def load_bitext_chunks() -> list[dict]:
    df = pd.read_csv(DATA_DIR / "bitext_dataset" / "processed.csv")

    with open(DATA_DIR / "company_docs" / "canonical_answers.json", encoding="utf-8") as f:
        canonical = json.load(f)

    missing = set(df["intent"].unique()) - set(canonical.keys())
    if missing:
        print(f"WARNING: no canonical answer written for intents: {missing} — "
              f"falling back to the dataset's own response for these.")

    chunks = []
    for _, row in df.iterrows():
        answer = canonical.get(row["intent"])
        if answer is None:
            answer = row["response"]  # fallback only for uncovered intents
        chunks.append({
            "text": row["instruction"],   # what gets embedded & searched
            "answer": answer,             # what gets returned to the model
            "category": row["category"],
            "intent": row["intent"],
            "source": "bitext",
        })
    return chunks


def load_company_profile_chunks() -> list[dict]:
    path = DATA_DIR / "company_docs" / "company_profile.md"
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"\n(?=## )", text)

    chunks = []
    for section in sections:
        section = section.strip()
        # re.split leaves whatever came before the first "## " header (the
        # file's "# ..." title line) as its own fragment -- that's not a
        # real content section, so skip it rather than treating the title
        # as a chunk with no example questions.
        if not section or not section.startswith("##"):
            continue

        match = re.search(r"^Example questions:\s*(.+)$", section, re.MULTILINE | re.IGNORECASE)
        if match:
            questions = [q.strip() for q in match.group(1).split(";") if q.strip()]
            answer = section[:match.start()].strip()
        else:
            questions = []
            answer = section

        if not questions:
            title = answer.splitlines()[0] if answer else "(empty section)"
            print(f"NOTE: no 'Example questions:' line found for company_profile section: '{title}'")

        chunks.append({
            "text": answer,
            "answer": answer,
            "category": "COMPANY_INFO",
            "intent": "company_profile",
            "source": "company_profile",
        })

        for question in questions:
            chunks.append({
                "text": question,
                "answer": answer,
                "category": "COMPANY_INFO",
                "intent": "company_profile",
                "source": "company_profile",
            })

    return chunks


def get_or_create_index(pc: Pinecone):
    existing_names = {idx["name"] for idx in pc.list_indexes()}
    if PINECONE_INDEX_NAME not in existing_names:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' ({EMBED_DIM}d, cosine, "
              f"{PINECONE_CLOUD}/{PINECONE_REGION})...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    else:
        print(f"Using existing Pinecone index '{PINECONE_INDEX_NAME}'.")
    return pc.Index(PINECONE_INDEX_NAME)


def main():
    BM25_STORE_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_bitext_chunks() + load_company_profile_chunks()
    print(f"Total chunks to embed: {len(chunks)}")

    # Save the local BM25 corpus BEFORE upserting -- if the Pinecone call
    # fails partway, we still have a consistent local snapshot to debug
    # against, and re-running is idempotent either way.
    with open(BM25_STORE_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=128
    )
    embeddings = np.array(embeddings, dtype="float32")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = get_or_create_index(pc)

    # Vector ids are just chunk positions, and the chunk count changes any
    # time company_profile.md or the dataset changes. Without clearing
    # first, a run that produces fewer chunks than a previous run leaves
    # orphaned high-numbered ids behind in Pinecone, pointing at stale
    # metadata that no longer has a matching position in chunks.pkl --
    # clearing keeps the index exactly in sync with the current corpus.
    print("Clearing existing vectors before re-upserting...")
    try:
        index.delete(delete_all=True)
    except Exception as e:
        # A brand-new, never-written-to index can reject delete_all in some
        # client versions -- safe to ignore in that case.
        print(f"  (nothing to clear, or index was already empty: {e})")

    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": str(i),
            "values": emb.tolist(),
            "metadata": {
                "text": chunk["text"],
                "answer": chunk["answer"],
                "category": chunk["category"],
                "intent": chunk["intent"],
                "source": chunk["source"],
            },
        })

    print(f"Upserting {len(vectors)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'...")
    for start in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[start:start + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"  upserted {start + len(batch)}/{len(vectors)}")

    print(
        f"Done. Pinecone index '{PINECONE_INDEX_NAME}' populated with "
        f"{len(vectors)} vectors; local BM25 chunk store saved to "
        f"{BM25_STORE_DIR}/chunks.pkl"
    )


if __name__ == "__main__":
    main()
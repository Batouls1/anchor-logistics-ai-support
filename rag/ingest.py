"""
Builds the vector store from data/bitext_dataset/processed.csv and
data/company_docs/company_profile.md.

"""

import json
import re
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR = Path("data")
VECTOR_STORE_DIR = Path("rag/vector_store")


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
        if not section:
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

        # Clean answer, searchable on its own prose too (fallback)
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


def main():
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_bitext_chunks() + load_company_profile_chunks()
    print(f"Total chunks to embed: {len(chunks)}")

    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=128
    )
    embeddings = np.array(embeddings, dtype="float32")

    # Normalized embeddings + inner product = cosine similarity search
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(VECTOR_STORE_DIR / "index.faiss"))

    with open(VECTOR_STORE_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(
        f"Saved FAISS index ({embeddings.shape[0]} vectors, dim {embeddings.shape[1]}) "
        f"and chunk metadata to {VECTOR_STORE_DIR}/"
    )


if __name__ == "__main__":
    main()
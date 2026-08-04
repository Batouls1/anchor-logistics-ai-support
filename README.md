# Anchor Logistics AI Support Assistant

A production-shaped AI customer support assistant with text and voice
chat, built on a hybrid RAG pipeline and Gemini's dual text/audio models.

**Live demo:** http://<your-ec2-ip>:8000

## Features

- **Grounded answers, not guesses** — every response is retrieved from a
  knowledge base via hybrid search (FAISS + BM25 + cross-encoder
  reranking), never generated from model memory
- **Text and voice, both first-class** — typed messages get fast text
  replies; voice notes get transcribed (Whisper), answered, and spoken
  back in natural audio (Gemini Live)
- **Persistent conversations** — every turn saved to PostgreSQL
- **Resilient by design** — recovers from dropped connections, rate
  limits, and failed transcriptions without breaking the conversation
- **Fully containerized** — one command spins up the app + database

## Architecture

**Text path:** Browser → FastAPI → Gemini (text) → RAG tool call → reply

**Voice path:** Browser → FastAPI → Whisper (STT) → Gemini Live (audio) → RAG tool call → spoken + text reply

Both paths call the same retrieval step: **FAISS + BM25 → cross-encoder rerank → answer**, and every turn is saved to **PostgreSQL**.

Text and voice run through **separate Gemini sessions** — Live's audio output is session-wide, not per-turn, so routing text through it would mean generating and discarding speech on every typed reply.


## Notable engineering decisions

- Retrieval threshold **calibrated empirically** against real adversarial
  queries — the naive default rejected correct answers as often as bad
  ones
- Support prompt **never implies a service doesn't exist** just because
  it's missing from the KB — a subtle hallucination pattern most support
  bots miss
- **Live reconnect replays recent history** from Postgres into the fresh
  session, so a dropped connection doesn't wipe conversation context
- **RAG inference runs off the event loop** (`asyncio.to_thread`) so one
  conversation's retrieval can't stall every other concurrent session

## Known limitations

- Voice replies take ~5–10s on CPU (`small.en` chosen for accuracy over
  speed) — would shrink substantially on GPU
- No automated test suite — correctness verified via structured manual
  testing; flagged as a deliberate next step, not an oversight
- Text and voice sessions don't share context with each other by design

## Tech stack

FastAPI · PostgreSQL + SQLAlchemy (async) · Gemini Live + Gemini API ·
FAISS · BM25 · Sentence-Transformers (embeddings + reranking) ·
faster-whisper · Docker Compose · vanilla JS/CSS

## Run it locally

```bash
git clone https://github.com/<you>/anchor-logistics-ai-support
cd anchor-logistics-ai-support
cp .env.example .env   # add your GEMINI_API_KEY
docker compose up --build
```
→ `http://localhost:8000`

## Structure

| Folder | Contents |
|---|---|
| `data/` | dataset prep + company policy docs |
| `rag/` | hybrid retriever + vector store |
| `gemini/` | Live (voice) + text Gemini clients, RAG tool wrapper |
| `whisper/` | speech-to-text |
| `conversation/` | turn orchestration + fallback handling |
| `database/` | SQLAlchemy models + async connection layer |
| `frontend/` | vanilla JS/CSS chat UI |
| `main.py` | FastAPI app |
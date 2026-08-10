# Anchor Logistics AI Support Assistant

A production-shaped AI customer support assistant with two distinct
conversation modes — text/voice-note chat and a live voice call — built
on a hybrid RAG pipeline and Gemini's text and native-audio models.

## Features

- **Grounded answers, not guesses** — every response is retrieved from a
  knowledge base via hybrid search (Pinecone + BM25 + cross-encoder
  reranking), never generated from model memory

- **Two conversation modes, deliberately separate**
  - **Chat (default):** typed messages and voice notes both go through
    the same text conversation — a voice note is transcribed (Whisper)
    and answered exactly like a typed message, sharing one history
  - **Live call:** a separate real-time voice call using Gemini Live's
    native audio understanding directly — continuous streamed audio, no
    Whisper, its own session

- **Persistent conversations** — every chat turn saved to PostgreSQL

- **Resilient by design** — recovers from rate limits and failed
  transcriptions without breaking the conversation

- **Automated test suite** — 20 tests covering retrieval logic, turn
  routing, and Live event translation

- **Fully containerized** — one command spins up the app + database

## Architecture

**Chat path (typed or voice note):**
Browser → FastAPI → \ [Whisper, if voice] → Gemini (text) → RAG tool call → text reply

**Live call path:**
Browser (streamed mic audio) → FastAPI WebSocket → Gemini Live (native audio) → RAG tool call → streamed audio + transcript reply

Both paths call the same retrieval step: **Pinecone + BM25 → reciprocal
rank fusion → cross-encoder rerank → answer**. The two paths are
architecturally separate on purpose — Gemini Live's audio output is
session-wide, not per-turn, so a model built for continuous voice can't
also cleanly serve one-off text replies. Chat and Live call also don't
share conversation history with each other.

## Notable engineering decisions

- Retrieval threshold **calibrated empirically** against real adversarial
  queries — the naive default rejected correct answers as often as bad
  ones
- Support prompt **never implies a service doesn't exist** just because
  it's missing from the KB — a subtle hallucination pattern most support
  bots miss
- **Pinecone re-ingestion clears the index first** — vector ids are just
  chunk positions, so a shrinking knowledge base would otherwise leave
  orphaned vectors behind from a previous, larger run
- **RAG's retriever loads lazily**, on first tool call rather than at
  import time — keeps unit tests fast and independent of live credentials
- **RAG inference runs off the event loop** (`asyncio.to_thread`) so one
  conversation's retrieval can't stall every other concurrent session


## Known limitations

- The Live call model is preview-tier (native audio streaming has no
  stable equivalent yet) and doesn't reconnect-with-history if the
  connection drops mid-call — a dropped call just ends, deliberately,
  rather than silently resuming
- Live call transcripts aren't persisted to Postgres in this pass
- First container start runs the RAG ingest step (a few minutes) if the
  local BM25 store doesn't already exist; subsequent restarts reuse it

## Tech stack

FastAPI · PostgreSQL + SQLAlchemy (async) · Gemini API (text) + Gemini
Live (streaming voice) · Pinecone · BM25 · Sentence-Transformers
(embeddings + reranking) · faster-whisper · Docker Compose · pytest ·
vanilla JS/CSS

## Run it locally

```bash
git clone https://github.com/batouls1/anchor-logistics-ai-support
cd anchor-logistics-ai-support
cp .env.example .env   # add your GEMINI_API_KEY and PINECONE_API_KEY
docker compose up --build
```
→ `http://localhost:8000`

First run will take a few minutes longer than usual — the entrypoint
script builds the local BM25 store automatically if it isn't already
present.

## Tests

```bash
pytest
```
20 tests covering RAG fusion/dedup/threshold logic, turn-routing between
typed and voice input, and Gemini Live event translation. All external
services (Pinecone, Gemini, Whisper) are mocked — the suite runs without
live credentials.

## Structure

| Folder | Contents |
|---|---|
| `data/` | dataset prep + company policy docs |
| `rag/` | hybrid retriever (Pinecone + BM25) + ingest script |
| `gemini/` | text client (Chat), Live client (Live call), RAG tool wrapper |
| `whisper/` | speech-to-text for voice notes |
| `conversation/` | turn orchestration + transcription fallback handling |
| `database/` | SQLAlchemy models + async connection layer |
| `frontend/` | vanilla JS/CSS chat widget + live-call UI |
| `tests/` | pytest suite |
| `main.py` | FastAPI app + WebSocket live-call bridge |
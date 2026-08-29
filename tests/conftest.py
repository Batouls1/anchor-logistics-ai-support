"""
Shared test setup.

database/connections.py reads DATABASE_URL at *import* time, and
main.py is the only place that calls load_dotenv(). Under pytest, main.py
is never imported first, so anything importing TurnManager blew up during
collection with KeyError: 'DATABASE_URL'. Loading .env here (before any
test module is imported) fixes that without changing app code.

No test in this suite talks to a real database, Pinecone index, or the
Gemini API -- the values only need to exist so imports succeed.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Fallbacks so the suite still runs on a machine with no .env at all
# (e.g. CI). These are never connected to -- every DB/API call is mocked.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://postgres:testpassword@localhost:5432/test"
)
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
os.environ.setdefault("PINECONE_API_KEY", "test-key-not-used")

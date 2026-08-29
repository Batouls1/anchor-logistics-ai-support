import asyncio
import logging
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope, Receive, Send
from database.connections import init_db, close_conversation

from conversation.rate_limit import RateLimiter
from conversation.session_token import (
    issue_conversation_token,
    verify_conversation_token,
)
from conversation.turn_manager import TurnManager
from gemini.live_client import LiveCallSession
from gemini.tools import warm_up
from gemini.errors import QuotaExceededError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# basicConfig above sets the *root* logger to INFO, which every library
# inherits. A few of them are extremely chatty at that level: loading the
# two RAG models makes ~50 HTTP cache-freshness checks against the HF Hub
# on every single startup, and reload=True replays all of it on each file
# save. Muting their INFO keeps our own startup/error logs readable.
# Warnings and errors from these libraries still come through -- only the
# routine chatter is dropped.
for _noisy_logger in (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "pinecone",
    "urllib3",
    "filelock",
):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

GRACEFUL_ERROR_MESSAGE = (
    "Sorry, something went wrong on my end. Could you try that again?"
)
QUOTA_ERROR_MESSAGE = (
    "I'm getting a lot of requests right now and hit a temporary limit. "
    "Please try again in a minute."
)
BUSY_ERROR_MESSAGE = (
    "You're sending messages faster than I can answer them. "
    "Give me a moment and try again."
)
INVALID_SESSION_MESSAGE = (
    "This conversation has expired. Please refresh the page to start a new one."
)
TOO_LONG_MESSAGE = (
    "That message is too long for me to process. Could you shorten it?"
)

# Input limits. Without these, a single request can push an unbounded
# prompt at the model or spool an arbitrarily large upload to disk.
MAX_MESSAGE_CHARS = 4000
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB -- minutes of webm/opus speech
AUDIO_CHUNK_BYTES = 64 * 1024

# Per-conversation limits, sized around what a person can actually do:
# a typed or spoken turn every few seconds. They exist to contain runaway
# clients and protect the Gemini quota, not to police normal use.
_chat_limiter = RateLimiter(max_events=20, window_seconds=60)
# Keyed by client address instead, since a caller with no valid token yet
# has no conversation to bucket by.
_start_limiter = RateLimiter(max_events=10, window_seconds=60)
_live_call_limiter = RateLimiter(max_events=5, window_seconds=60)

# In-memory session store, keyed by conversation_id. Not the source of
# truth: TurnManager.start() rebuilds history from Postgres, so a session
# missing here (restart, different worker) is re-created rather than lost.
_sessions: dict[str, TurnManager] = {}
_last_active: dict[str, float] = {}


IDLE_TIMEOUT_SECONDS = 15 * 60
SWEEP_INTERVAL_SECONDS = 60


async def _sweep_idle_sessions():
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
        now = time.monotonic()
        stale_ids = [
            cid for cid, last_seen in _last_active.items()
            if now - last_seen > IDLE_TIMEOUT_SECONDS
        ]
        for cid in stale_ids:
            tm = _sessions.pop(cid, None)
            _last_active.pop(cid, None)
            if tm:
                try:
                    await tm.close()
                except Exception:
                    logger.exception("Error closing idle session %s", cid)
            try:
                await close_conversation(cid)
            except Exception:
                logger.exception("Error marking conversation %s closed", cid)
        if stale_ids:
            logger.info("Swept %d idle session(s)", len(stale_ids))

        # Rate limiter buckets outlive the requests that created them, so
        # they're pruned on the same schedule -- otherwise a public
        # endpoint accumulates one dict entry per client indefinitely.
        for limiter in (_chat_limiter, _start_limiter, _live_call_limiter):
            limiter.prune()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("Warming up the RAG retriever...")
    if await asyncio.to_thread(warm_up):
        logger.info("Retriever ready.")
    else:
        # Deliberately not fatal. Warming up is a latency optimisation,
        # not a requirement to serve traffic: the retriever is still lazy
        # underneath, so the next tool call retries building it. Letting
        # this stop startup would turn a degraded feature into a total
        # outage every time the model host or Pinecone had a bad minute.
        logger.warning(
            "Retriever unavailable at startup -- the app is running and "
            "will retry building it on the first knowledge base lookup."
        )
    sweep_task = asyncio.create_task(_sweep_idle_sessions())
    yield
    sweep_task.cancel()
    for tm in _sessions.values():
        await tm.close()


class SafeStaticFiles(StaticFiles):
    """
    Wraps StaticFiles to reject non-HTTP scopes (e.g. malformed WebSocket
    probes from internet bots scanning for vulnerable devices) cleanly,
    instead of raising an unhandled AssertionError that shows up as a
    scary-looking traceback in the logs. The app never actually crashed
    from this -- uvicorn was already catching it safely.

    WebSocket scopes are closed explicitly rather than just returned
    from: returning without sending anything leaves the client hanging
    until it times out, since nothing ever accepts or rejects the
    handshake.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1000})
            return
        if scope["type"] != "http":
            return
        await super().__call__(scope, receive, send)


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    One structured line per request, carrying a generated request id.

    Without this, a report like "the assistant gave a bad answer at 3pm"
    can't be tied to anything in the logs. The id is echoed in the
    response header so a client-side error can be matched to the exact
    server-side request that produced it.
    """
    request_id = uuid.uuid4().hex[:12]
    started = time.monotonic()

    response = await call_next(request)

    duration_ms = (time.monotonic() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    # Static asset noise would drown out the useful lines.
    if request.url.path.startswith(("/chat", "/conversation")):
        logger.info(
            "request_id=%s method=%s path=%s status=%d duration_ms=%.0f client=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            _client_key(request),
        )
    return response


def _client_key(request: Request) -> str:
    """
    Best-effort client identity for rate limiting. Honours
    X-Forwarded-For, since behind a proxy every request otherwise appears
    to come from the proxy itself and shares one bucket.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _resolve_conversation(token: str) -> str | None:
    """Verifies a client-supplied token and returns its conversation id."""
    return verify_conversation_token(token)


@app.post("/conversation/start")
async def start_conversation(request: Request):
    """
    Issues a signed conversation token. Clients can no longer invent
    their own ids, which is what previously let anyone read or append to
    another conversation just by guessing its identifier.
    """
    if not _start_limiter.allow(_client_key(request)):
        return JSONResponse(
            status_code=429,
            content={"type": "error", "agent_text": BUSY_ERROR_MESSAGE},
        )
    return {"conversation_id": issue_conversation_token()}


async def _get_or_create_session(conversation_id: str) -> TurnManager:
    if conversation_id not in _sessions:
        tm = TurnManager(conversation_id)
        await tm.start()
        _sessions[conversation_id] = tm
    _last_active[conversation_id] = time.monotonic()
    return _sessions[conversation_id]


@app.post("/chat/text")
async def chat_text(conversation_id: str = Form(...), message: str = Form(...)):
    session_id = _resolve_conversation(conversation_id)
    if session_id is None:
        return JSONResponse(
            status_code=401,
            content={"type": "error", "user_text": None, "agent_text": INVALID_SESSION_MESSAGE},
        )

    if len(message) > MAX_MESSAGE_CHARS:
        return JSONResponse(
            status_code=413,
            content={"type": "error", "user_text": None, "agent_text": TOO_LONG_MESSAGE},
        )

    if not _chat_limiter.allow(session_id):
        logger.warning("Rate limit hit on /chat/text for conversation %s", session_id)
        return JSONResponse(
            status_code=429,
            content={"type": "error", "user_text": None, "agent_text": BUSY_ERROR_MESSAGE},
        )

    try:
        tm = await _get_or_create_session(session_id)
        result = await tm.handle_text(message)

        return {
            "type": "text",
            "user_text": result.user_text,
            "agent_text": result.agent_text,
        }
    except QuotaExceededError:
        logger.warning("Quota exceeded on /chat/text")
        return {
            "type": "error",
            "user_text": message,
            "agent_text": QUOTA_ERROR_MESSAGE,
        }
    except Exception:
        logger.exception("Unhandled error in /chat/text")
        return {
            "type": "error",
            "user_text": message,
            "agent_text": GRACEFUL_ERROR_MESSAGE,
        }


@app.post("/chat/voice")
async def chat_voice(conversation_id: str = Form(...), audio: UploadFile = File(...)):
    """
    Path A voice note. Text-only end to end: Whisper transcribes the
    upload, the text model answers, and the reply comes back as text --
    nothing is synthesised to speech. Spoken replies are Path B's job
    (/ws/live-call), which streams audio over its own WebSocket.
    """
    session_id = _resolve_conversation(conversation_id)
    if session_id is None:
        return JSONResponse(
            status_code=401,
            content={"type": "error", "user_text": None, "agent_text": INVALID_SESSION_MESSAGE},
        )

    if not _chat_limiter.allow(session_id):
        logger.warning("Rate limit hit on /chat/voice for conversation %s", session_id)
        return JSONResponse(
            status_code=429,
            content={"type": "error", "user_text": None, "agent_text": BUSY_ERROR_MESSAGE},
        )

    tm = await _get_or_create_session(session_id)

    # Streamed in bounded chunks with a running total, rather than
    # copyfileobj'ing whatever arrives: an unbounded upload would
    # otherwise be spooled to disk in full before anyone checked its size.
    suffix = Path(audio.filename or "").suffix or ".webm"
    written = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        while chunk := await audio.read(AUDIO_CHUNK_BYTES):
            written += len(chunk)
            if written > MAX_AUDIO_BYTES:
                break
            tmp.write(chunk)

    if written > MAX_AUDIO_BYTES:
        _unlink_quietly(temp_path)
        logger.warning("Rejected an oversized voice note (%d bytes)", written)
        return JSONResponse(
            status_code=413,
            content={"type": "error", "user_text": None, "agent_text": TOO_LONG_MESSAGE},
        )

    try:
        result = await tm.handle_voice(temp_path)

        return {
            "type": "fallback" if result.is_fallback else "voice",
            "user_text": result.user_text,
            "agent_text": result.agent_text,
        }
    except QuotaExceededError:
        logger.warning("Quota exceeded on /chat/voice")
        return {
            "type": "error",
            "user_text": None,
            "agent_text": QUOTA_ERROR_MESSAGE,
        }
    except Exception:
        logger.exception("Unhandled error in /chat/voice")
        return {
            "type": "error",
            "user_text": None,
            "agent_text": GRACEFUL_ERROR_MESSAGE,
        }
    finally:
        _unlink_quietly(temp_path)


def _unlink_quietly(path: str) -> None:
    """
    Windows keeps a handle open a moment after Whisper releases the file,
    so a PermissionError here means "not yet", not "leaked" -- the OS
    cleans the temp directory regardless.
    """
    try:
        os.unlink(path)
    except (PermissionError, FileNotFoundError):
        pass


@app.post("/conversation/end")
async def end_conversation(conversation_id: str = Form(...)):
    session_id = _resolve_conversation(conversation_id)
    if session_id is None:
        # Nothing to close, and no reason to tell an unauthenticated
        # caller whether the id existed.
        return JSONResponse(status_code=401, content={"status": "invalid_session"})

    tm = _sessions.pop(session_id, None)
    _last_active.pop(session_id, None)
    if tm:
        await tm.close()
    await close_conversation(session_id)
    return {"status": "closed"}


@app.websocket("/ws/live-call")
async def live_call_ws(websocket: WebSocket):
    """
    Path B. Bridges the browser's continuous mic audio to a Gemini Live
    session and streams Gemini's audio back, for the lifetime of one
    WebSocket connection. Entirely separate from the /chat/* endpoints
    and TurnManager -- no shared session store, no shared history, and
    no text: audio goes out as binary frames, and the only JSON frames
    are control signals (interrupted / turn_complete) plus connection
    errors. Nothing said on a live call appears in the text chat.
    """
    await websocket.accept()

    # A live call holds a Gemini session open for its whole duration, so
    # it's by far the most expensive thing an anonymous client can start.
    # Keyed by address rather than conversation: the call deliberately has
    # no conversation of its own.
    forwarded = websocket.headers.get("x-forwarded-for", "")
    client_key = (
        forwarded.split(",")[0].strip()
        if forwarded
        else (websocket.client.host if websocket.client else "unknown")
    )
    if not _live_call_limiter.allow(client_key):
        logger.warning("Rate limit hit on /ws/live-call from %s", client_key)
        await websocket.send_json({"type": "error", "message": BUSY_ERROR_MESSAGE})
        await websocket.close(code=1013)  # 1013 = "try again later"
        return

    session = LiveCallSession()
    try:
        await session.start()
    except QuotaExceededError:
        logger.warning("Quota exceeded starting a live call session")
        await websocket.send_json({"type": "error", "message": QUOTA_ERROR_MESSAGE})
        await websocket.close(code=1011)
        return
    except Exception:
        logger.exception("Failed to start live call session")
        await websocket.send_json({"type": "error", "message": GRACEFUL_ERROR_MESSAGE})
        await websocket.close(code=1011)
        return

    async def browser_to_gemini():
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            audio_bytes = message.get("bytes")
            if audio_bytes:
                await session.send_audio_chunk(audio_bytes)

    async def gemini_to_browser():
        async for event in session.receive_events():
            if event["type"] == "audio":
                await websocket.send_bytes(event["data"])
            else:
                await websocket.send_json(event)

    tasks = [
        asyncio.create_task(browser_to_gemini()),
        asyncio.create_task(gemini_to_browser()),
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if pending:
            # Cancelling isn't enough on its own -- awaiting lets each
            # task actually unwind (raising CancelledError internally)
            # before session.close() runs underneath it. Without this,
            # cancelled tasks can log spurious "exception was never
            # retrieved" warnings and leave things mid-flight.
            await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            exc = task.exception()
            if exc:
                logger.exception("Error in live call bridge", exc_info=exc)
    finally:
        await session.close()


app.mount("/", SafeStaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
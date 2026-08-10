import asyncio
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, UploadFile, WebSocket
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope, Receive, Send
from database.connections import init_db, close_conversation

from conversation.turn_manager import TurnManager
from gemini.live_client import LiveCallSession
from gemini.errors import QuotaExceededError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

GRACEFUL_ERROR_MESSAGE = (
    "Sorry, something went wrong on my end. Could you try that again?"
)
QUOTA_ERROR_MESSAGE = (
    "I'm getting a lot of requests right now and hit a temporary limit. "
    "Please try again in a minute."
)

# Kept mounted for Path B (live call) to use once its audio-delivery
# design is decided -- Path A no longer writes anything here, since it's
# text-only end to end.
STATIC_AUDIO_DIR = Path("static/audio")
STATIC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# In-memory session store, keyed by conversation_id
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
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
    scary-looking traceback in the logs. Purely cosmetic -- the app never
    actually crashed from this, uvicorn was already catching it safely.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        await super().__call__(scope, receive, send)


app = FastAPI(lifespan=lifespan)

app.mount("/static", SafeStaticFiles(directory="static"), name="static")


async def _get_or_create_session(conversation_id: str) -> TurnManager:
    if conversation_id not in _sessions:
        tm = TurnManager(conversation_id)
        await tm.start()
        _sessions[conversation_id] = tm
    _last_active[conversation_id] = time.monotonic()
    return _sessions[conversation_id]


@app.post("/chat/text")
async def chat_text(conversation_id: str = Form(...), message: str = Form(...)):
    try:
        tm = await _get_or_create_session(conversation_id)
        result = await tm.handle_text(message)

        return {
            "type": "text",
            "user_text": result.user_text,
            "agent_text": result.agent_text,
            "audio_url": None,
        }
    except QuotaExceededError:
        logger.warning("Quota exceeded on /chat/text")
        return {
            "type": "error",
            "user_text": message,
            "agent_text": QUOTA_ERROR_MESSAGE,
            "audio_url": None,
        }
    except Exception:
        logger.exception("Unhandled error in /chat/text")
        return {
            "type": "error",
            "user_text": message,
            "agent_text": GRACEFUL_ERROR_MESSAGE,
            "audio_url": None,
        }


@app.post("/chat/voice")
async def chat_voice(conversation_id: str = Form(...), audio: UploadFile = File(...)):
    tm = await _get_or_create_session(conversation_id)

    suffix = Path(audio.filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        temp_path = tmp.name

    try:
        result = await tm.handle_voice(temp_path)

        return {
            "type": "fallback" if result.is_fallback else "voice",
            "user_text": result.user_text,
            "agent_text": result.agent_text,
            # Path A is text-only end to end -- Whisper transcribes, the
            # plain text model replies, nothing is synthesized back to
            # speech. Path B (live call) handles audio on its own path.
            "audio_url": None,
        }
    except QuotaExceededError:
        logger.warning("Quota exceeded on /chat/voice")
        return {
            "type": "error",
            "user_text": None,
            "agent_text": QUOTA_ERROR_MESSAGE,
            "audio_url": None,
        }
    except Exception:
        logger.exception("Unhandled error in /chat/voice")
        return {
            "type": "error",
            "user_text": None,
            "agent_text": GRACEFUL_ERROR_MESSAGE,
            "audio_url": None,
        }
    finally:
        try:
            os.unlink(temp_path)
        except PermissionError:
            pass


@app.post("/conversation/end")
async def end_conversation(conversation_id: str = Form(...)):
    tm = _sessions.pop(conversation_id, None)
    _last_active.pop(conversation_id, None)
    if tm:
        await tm.close()
    await close_conversation(conversation_id)
    return {"status": "closed"}


@app.websocket("/ws/live-call")
async def live_call_ws(websocket: WebSocket):
    """
    Path B. Bridges the browser's continuous mic audio to a Gemini Live
    session and streams Gemini's audio back, for the lifetime of one
    WebSocket connection. Entirely separate from the /chat/* endpoints
    and TurnManager -- no shared session store, no shared history.
    """
    await websocket.accept()

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
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["static/audio/*.wav", "*.wav"],
    )
import asyncio
import logging
import os
import shutil
import tempfile
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope, Receive, Send
from database.connections import init_db

from conversation.turn_manager import TurnManager
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

STATIC_AUDIO_DIR = Path("static/audio")
STATIC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# In-memory session store, keyed by conversation_id
_sessions: dict[str, TurnManager] = {}
_turn_counters: dict[str, int] = {}
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
            _turn_counters.pop(cid, None)
            _last_active.pop(cid, None)
            if tm:
                try:
                    await tm.close()
                except Exception:
                    logger.exception("Error closing idle session %s", cid)
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
        _turn_counters[conversation_id] = 0
    _last_active[conversation_id] = time.monotonic()
    return _sessions[conversation_id]

def _save_audio(conversation_id: str, turn: int, audio_bytes: bytes) -> str:
    filename = f"response_{conversation_id}_{turn}.wav"
    wav_path = STATIC_AUDIO_DIR / filename
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_bytes)
    return f"/static/audio/{filename}"


@app.post("/chat/text")
async def chat_text(conversation_id: str = Form(...), message: str = Form(...)):
    try:
        tm = await _get_or_create_session(conversation_id)
        result = await tm.handle_text(message)

        # Text turns are text-only end to end now
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
    _turn_counters[conversation_id] += 1
    turn = _turn_counters[conversation_id]

    suffix = Path(audio.filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        temp_path = tmp.name

    try:
        result = await tm.handle_voice(temp_path)

        audio_url = None
        if result.agent_audio:
            audio_url = _save_audio(conversation_id, turn, result.agent_audio)

        return {
            "type": "fallback" if result.is_fallback else "voice",
            "user_text": result.user_text,
            "agent_text": result.agent_text,
            "audio_url": audio_url,
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
    _turn_counters.pop(conversation_id, None)
    _last_active.pop(conversation_id, None)
    if tm:
        await tm.close()
    return {"status": "closed"}


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
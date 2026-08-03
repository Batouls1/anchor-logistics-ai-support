from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from whisper.stt import transcribe
from gemini.live_client import LiveSession
from gemini.text_client import TextSession
from conversation.fallback import FallbackHandler
from database.connections import get_or_create_conversation, record_turn, get_recent_turns


@dataclass
class TurnResult:
    user_text: Optional[str] = None
    agent_text: Optional[str] = None
    agent_audio: Optional[bytes] = None
    is_fallback: bool = False


class TurnManager:
    """
    One instance of this = one ongoing conversation. Owns two separate
    Gemini clients: a LiveSession for voice turns (needs AUDIO output) and
    a TextSession for typed turns (needs text only)
    """

    def __init__(self, conversation_id: str):
        self._conversation_id = conversation_id

        # LiveSession gets a callback for fetching recent DB history --
        # only used if it ever needs to reconnect mid-conversation, so
        # the fresh Gemini-side session isn't starting from zero context.
        self._session = LiveSession(
            history_provider=lambda: get_recent_turns(self._conversation_id)
        )
        self._text_session = TextSession()
        self._fallback = FallbackHandler(max_retries=1)

    async def start(self):
        await get_or_create_conversation(self._conversation_id)
        await self._session.start()

    async def close(self):
        await self._session.close()

    async def handle_text(self, text: str) -> TurnResult:
        agent_text = await self._text_session.send_message(text)
        self._fallback.reset()

        await record_turn(
            self._conversation_id, "text", user_text=text, agent_text=agent_text
        )

        return TurnResult(user_text=text, agent_text=agent_text, agent_audio=None)

    async def handle_voice(self, audio_path: str | Path) -> TurnResult:
        stt = transcribe(str(audio_path))

        if not stt["confident"]:
            message = self._fallback.record_failure()
            await record_turn(
                self._conversation_id, "voice",
                user_text=None, agent_text=message, is_fallback=True,
            )
            return TurnResult(agent_text=message, is_fallback=True)

        result = await self._session.send_message(stt["text"])
        self._fallback.reset()

        await record_turn(
            self._conversation_id, "voice",
            user_text=stt["text"], agent_text=result["text"],
        )

        return TurnResult(
            user_text=stt["text"], agent_text=result["text"], agent_audio=result["audio"]
        )
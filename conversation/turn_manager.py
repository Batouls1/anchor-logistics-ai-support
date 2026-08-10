"""
One instance = one Path A conversation. Typed text and voice notes share
a single TextSession/history. Path B (live call) is handled separately
by LiveCallSession, not TurnManager.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from whisper.stt import transcribe
from gemini.text_client import TextSession
from conversation.fallback import FallbackHandler
from database.connections import get_or_create_conversation, record_turn


@dataclass
class TurnResult:
    user_text: Optional[str] = None
    agent_text: Optional[str] = None
    is_fallback: bool = False


class TurnManager:
    """
    Owns exactly one TextSession per conversation. No audio in, no audio
    out -- Path A is text-only regardless of whether the input arrived
    typed or as a transcribed voice note.
    """

    def __init__(self, conversation_id: str):
        self._conversation_id = conversation_id
        self._text_session = TextSession()
        self._fallback = FallbackHandler(max_retries=1)

    async def start(self):
        await get_or_create_conversation(self._conversation_id)

    async def close(self):
        # TextSession makes plain generate_content calls -- no persistent
        # connection to tear down. Kept as a no-op (rather than removed)
        # so main.py's session lifecycle code doesn't need to special-case
        # Path A vs whatever Path B's session type ends up needing.
        pass

    async def handle_text(self, text: str) -> TurnResult:
        agent_text = await self._text_session.send_message(text)
        self._fallback.reset()

        await record_turn(
            self._conversation_id, "text", user_text=text, agent_text=agent_text
        )

        return TurnResult(user_text=text, agent_text=agent_text)

    async def handle_voice(self, audio_path: str | Path) -> TurnResult:
        stt = transcribe(str(audio_path))

        if not stt["confident"]:
            message = self._fallback.record_failure()
            await record_turn(
                self._conversation_id, "voice",
                user_text=None, agent_text=message, is_fallback=True,
            )
            return TurnResult(agent_text=message, is_fallback=True)

        # Same TextSession as typed messages -- the whole point of this
        # refactor. Whisper's transcript is just another user turn in the
        # one shared conversation history.
        agent_text = await self._text_session.send_message(stt["text"])
        self._fallback.reset()

        await record_turn(
            self._conversation_id, "voice",
            user_text=stt["text"], agent_text=agent_text,
        )

        return TurnResult(user_text=stt["text"], agent_text=agent_text)
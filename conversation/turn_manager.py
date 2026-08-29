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
from database.connections import (
    get_or_create_conversation,
    load_recent_turns,
    record_turn,
)

# How much prior conversation to replay when a session is rebuilt. Long
# enough to keep context that matters, short enough that a long-running
# conversation doesn't grow the prompt without bound.
HISTORY_REPLAY_LIMIT = 20


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
        """
        Prepares the session, rebuilding its history from Postgres if the
        conversation already exists.

        This is what stops the in-memory session store from being a hard
        single-process constraint: whichever worker picks up the next
        request reconstructs the same conversation from durable storage
        rather than starting blank. Restarts and multiple workers stop
        silently losing context.
        """
        await get_or_create_conversation(self._conversation_id)

        previous_turns = await load_recent_turns(
            self._conversation_id, limit=HISTORY_REPLAY_LIMIT
        )
        if previous_turns:
            self._text_session.prime_history(previous_turns)

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
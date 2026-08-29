"""
Tests for TurnManager: voice notes route through the same TextSession as
typed text, and low-confidence transcriptions hit the fallback path
without reaching Gemini. TextSession, Whisper, and the DB layer are
mocked.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from conversation.turn_manager import TurnManager


@pytest.fixture
def mocked_turn_manager():
    with patch("conversation.turn_manager.TextSession") as mock_text_session_cls, \
         patch("conversation.turn_manager.transcribe") as mock_transcribe, \
         patch("conversation.turn_manager.record_turn", new_callable=AsyncMock) as mock_record_turn:

        mock_text_session = mock_text_session_cls.return_value
        mock_text_session.send_message = AsyncMock(return_value="Mocked agent reply.")

        tm = TurnManager("test-conversation-id")

        yield tm, mock_text_session, mock_transcribe, mock_record_turn


def test_starting_a_session_replays_stored_history():
    """
    The in-memory session store isn't the source of truth: after a
    restart, or on a different worker, the conversation has to be rebuilt
    from Postgres rather than silently starting blank.
    """
    stored = [("where is my order", "It ships in 2 days."), ("thanks", "Any time!")]

    with patch("conversation.turn_manager.TextSession") as mock_cls, \
         patch("conversation.turn_manager.get_or_create_conversation", new_callable=AsyncMock), \
         patch("conversation.turn_manager.load_recent_turns", new_callable=AsyncMock) as mock_load:
        mock_load.return_value = stored
        tm = TurnManager("existing-conversation")
        asyncio.run(tm.start())

    mock_cls.return_value.prime_history.assert_called_once_with(stored)


def test_starting_a_brand_new_conversation_replays_nothing():
    with patch("conversation.turn_manager.TextSession") as mock_cls, \
         patch("conversation.turn_manager.get_or_create_conversation", new_callable=AsyncMock), \
         patch("conversation.turn_manager.load_recent_turns", new_callable=AsyncMock) as mock_load:
        mock_load.return_value = []
        tm = TurnManager("brand-new")
        asyncio.run(tm.start())

    mock_cls.return_value.prime_history.assert_not_called()


def test_handle_text_uses_text_session_and_records_turn(mocked_turn_manager):
    tm, mock_text_session, _mock_transcribe, mock_record_turn = mocked_turn_manager

    result = asyncio.run(tm.handle_text("how do I track my order?"))

    mock_text_session.send_message.assert_awaited_once_with("how do I track my order?")
    assert result.user_text == "how do I track my order?"
    assert result.agent_text == "Mocked agent reply."
    assert result.is_fallback is False
    mock_record_turn.assert_awaited_once()


def test_confident_voice_note_uses_the_same_text_session_as_typed_text(mocked_turn_manager):
    """
    The actual point of the Path A/B refactor: a voice note isn't routed
    through a separate Gemini session anymore -- it goes through the
    exact same TextSession a typed message would, so both share one
    conversation history.
    """
    tm, mock_text_session, mock_transcribe, _mock_record_turn = mocked_turn_manager
    mock_transcribe.return_value = {"text": "where is my package", "confident": True}

    result = asyncio.run(tm.handle_voice("/tmp/fake-audio.webm"))

    mock_text_session.send_message.assert_awaited_once_with("where is my package")
    assert result.user_text == "where is my package"
    assert result.agent_text == "Mocked agent reply."
    assert result.is_fallback is False


def test_low_confidence_voice_note_never_reaches_the_text_session(mocked_turn_manager):
    tm, mock_text_session, mock_transcribe, mock_record_turn = mocked_turn_manager
    mock_transcribe.return_value = {"text": "", "confident": False}

    result = asyncio.run(tm.handle_voice("/tmp/fake-audio.webm"))

    mock_text_session.send_message.assert_not_awaited()
    assert result.is_fallback is True
    assert result.agent_text == "I didn't catch that. Could you say that again?"
    mock_record_turn.assert_awaited_once()


def test_repeated_low_confidence_escalates_the_fallback_message(mocked_turn_manager):
    tm, _mock_text_session, mock_transcribe, _mock_record_turn = mocked_turn_manager
    mock_transcribe.return_value = {"text": "", "confident": False}

    first = asyncio.run(tm.handle_voice("/tmp/fake-audio.webm"))
    second = asyncio.run(tm.handle_voice("/tmp/fake-audio.webm"))

    assert first.agent_text == "I didn't catch that. Could you say that again?"
    assert second.agent_text == (
        "I'm still having trouble understanding. Please type your question "
        "so I can help you best."
    )


def test_successful_turn_resets_the_fallback_counter(mocked_turn_manager):
    """
    A successful voice note after a failed one should reset the retry
    count -- the next failure should get the soft message again, not
    stay escalated forever.
    """
    tm, _mock_text_session, mock_transcribe, _mock_record_turn = mocked_turn_manager

    mock_transcribe.return_value = {"text": "", "confident": False}
    asyncio.run(tm.handle_voice("/tmp/fake-audio.webm"))  # 1st failure

    mock_transcribe.return_value = {"text": "track my order", "confident": True}
    asyncio.run(tm.handle_voice("/tmp/fake-audio.webm"))  # succeeds, resets the counter

    mock_transcribe.return_value = {"text": "", "confident": False}
    result = asyncio.run(tm.handle_voice("/tmp/fake-audio.webm"))  # fresh 1st failure again

    assert result.agent_text == "I didn't catch that. Could you say that again?"
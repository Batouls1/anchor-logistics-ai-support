"""
Tests for LiveCallSession's event-translation logic (audio/transcript/
interrupted) and tool-calling loop. genai Client and the Live session are
mocked. Quota/connection-error paths aren't covered -- not worth the
fragility of mocking that deep into the SDK's internals.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gemini.live_client import LiveCallSession


def make_response(data=None, output_text=None, input_text=None, interrupted=False, tool_call=None):
    server_content = SimpleNamespace(
        output_transcription=SimpleNamespace(text=output_text) if output_text is not None else None,
        input_transcription=SimpleNamespace(text=input_text) if input_text is not None else None,
        interrupted=interrupted,
    )
    return SimpleNamespace(data=data, server_content=server_content, tool_call=tool_call)


async def _async_iter(items):
    for item in items:
        yield item


def build_session_with_responses(responses):
    """
    Builds a LiveCallSession whose underlying Gemini session.receive()
    yields exactly `responses`, without touching the real genai.Client or
    ever calling start().
    """
    with patch("gemini.live_client.genai"):
        session = LiveCallSession()

    mock_live_session = MagicMock()
    # .receive() itself is synchronous -- it returns an async generator,
    # which is then consumed with `async for`, same as the real SDK.
    mock_live_session.receive = MagicMock(return_value=_async_iter(responses))
    mock_live_session.send_tool_response = AsyncMock()
    session._session = mock_live_session
    return session, mock_live_session


def collect_events(session):
    async def _collect():
        return [event async for event in session.receive_events()]
    return asyncio.run(_collect())


def test_audio_chunk_yields_audio_event():
    responses = [make_response(data=b"\x01\x02")]
    session, _ = build_session_with_responses(responses)

    events = collect_events(session)

    assert events == [{"type": "audio", "data": b"\x01\x02"}]


def test_output_transcription_yields_assistant_transcript_event():
    responses = [make_response(output_text="Your order ships in 2 days.")]
    session, _ = build_session_with_responses(responses)

    events = collect_events(session)

    assert events == [{"type": "transcript", "role": "assistant", "text": "Your order ships in 2 days."}]


def test_input_transcription_yields_user_transcript_event():
    responses = [make_response(input_text="where is my order")]
    session, _ = build_session_with_responses(responses)

    events = collect_events(session)

    assert events == [{"type": "transcript", "role": "user", "text": "where is my order"}]


def test_interrupted_flag_yields_interrupted_event():
    responses = [make_response(interrupted=True)]
    session, _ = build_session_with_responses(responses)

    events = collect_events(session)

    assert events == [{"type": "interrupted"}]


def test_empty_transcription_text_does_not_yield_an_event():
    """
    Gemini sometimes sends a transcription field with empty text -- this
    shouldn't produce a blank transcript bubble in the frontend.
    """
    responses = [make_response(output_text="")]
    session, _ = build_session_with_responses(responses)

    events = collect_events(session)

    assert events == []


def test_tool_call_executes_and_sends_response_without_yielding_a_browser_event():
    function_call = SimpleNamespace(id="call-1", name="search_knowledge_base", args={"query": "refund policy"})
    tool_call = SimpleNamespace(function_calls=[function_call])
    responses = [make_response(tool_call=tool_call)]
    session, mock_live_session = build_session_with_responses(responses)

    with patch(
        "gemini.live_client.execute_tool",
        return_value={"found": True, "answers": ["mocked answer"]},
    ) as mock_execute:
        events = collect_events(session)

    # Tool calls are handled internally -- the browser never sees them as
    # an event, only whatever Gemini says once it has the tool's answer.
    assert events == []
    mock_execute.assert_called_once_with("search_knowledge_base", {"query": "refund policy"})
    mock_live_session.send_tool_response.assert_awaited_once()
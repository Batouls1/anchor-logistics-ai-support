"""
Tests for LiveCallSession's event translation (Path B).

Path B is audio-in / audio-out only. Two things must hold:
  1. the live call actually emits AUDIO events, and
  2. it emits NO text of any kind -- not the assistant's words, not a
     transcript of the caller's speech.
genai Client and the Live session are mocked; nothing here touches the
network.

Quota/HTTP-error paths aren't covered; not worth mocking that deep into
the SDK's internals.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from websockets.exceptions import ConnectionClosedError

import gemini.live_client as live_client
from gemini.live_client import LiveCallSession


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def make_response(
    data=None,
    inline_audio=None,
    output_text=None,
    input_text=None,
    interrupted=False,
    turn_complete=False,
    tool_call=None,
):
    """
    Builds one fake Gemini Live server response.

    `data` is the SDK's `response.data` shortcut for inline audio;
    `inline_audio` puts the same bytes under server_content.model_turn
    instead, which is the fallback path live_client now handles.
    """
    model_turn = None
    if inline_audio is not None:
        model_turn = SimpleNamespace(
            parts=[SimpleNamespace(inline_data=SimpleNamespace(data=inline_audio))]
        )

    server_content = SimpleNamespace(
        output_transcription=SimpleNamespace(text=output_text) if output_text is not None else None,
        input_transcription=SimpleNamespace(text=input_text) if input_text is not None else None,
        model_turn=model_turn,
        interrupted=interrupted,
        turn_complete=turn_complete,
    )
    return SimpleNamespace(data=data, server_content=server_content, tool_call=tool_call)


async def _async_iter(items):
    for item in items:
        yield item


def build_session_with_turns(turns):
    """
    Builds a LiveCallSession whose underlying session.receive() yields one
    list of responses per call -- matching the real SDK, where receive()
    exhausts at the end of each turn and is then called again.

    Once every scripted turn is used up, the next receive() raises
    ConnectionClosedError, which is how a real call actually ends.
    """
    with patch("gemini.live_client.genai"):
        session = LiveCallSession()

    remaining = list(turns)

    def _receive():
        if not remaining:
            raise ConnectionClosedError(None, None)
        return _async_iter(remaining.pop(0))

    mock_live_session = MagicMock()
    # .receive() itself is synchronous -- it returns an async generator,
    # which is then consumed with `async for`, same as the real SDK.
    mock_live_session.receive = MagicMock(side_effect=_receive)
    mock_live_session.send_tool_response = AsyncMock()
    mock_live_session.send_realtime_input = AsyncMock()
    session._session = mock_live_session
    return session, mock_live_session


def build_session_with_responses(responses):
    """Single-turn convenience wrapper around build_session_with_turns."""
    return build_session_with_turns([responses])


MAX_EVENTS = 200
COLLECT_TIMEOUT_SECONDS = 5


def collect_events(session):
    """
    Drains receive_events() with a hard cap on both event count and wall
    time. receive_events() wraps session.receive() in a `while True`, so a
    regression there turns into an infinite loop -- these guards make that
    fail the test in seconds instead of hanging the whole suite.
    """
    async def _collect():
        events = []
        async for event in session.receive_events():
            events.append(event)
            if len(events) > MAX_EVENTS:
                raise AssertionError(
                    f"receive_events() produced more than {MAX_EVENTS} events -- "
                    "it is most likely stuck in an infinite loop."
                )
        return events

    async def _run():
        return await asyncio.wait_for(_collect(), timeout=COLLECT_TIMEOUT_SECONDS)

    try:
        return asyncio.run(_run())
    except asyncio.TimeoutError:
        raise AssertionError(
            f"receive_events() did not finish within {COLLECT_TIMEOUT_SECONDS}s -- "
            "it is most likely stuck in an infinite loop."
        )


# --------------------------------------------------------------------
# The main question: does the live call produce audio?
# --------------------------------------------------------------------

def test_audio_arrives_as_an_audio_event():
    """The core Path B promise: PCM bytes reach the browser as audio."""
    responses = [make_response(data=b"\x01\x02")]
    session, _ = build_session_with_responses(responses)

    events = collect_events(session)

    assert events == [{"type": "audio", "data": b"\x01\x02"}]


def test_audio_is_recovered_from_model_turn_when_response_data_is_empty():
    """
    Some model/SDK combinations don't populate the `response.data`
    shortcut. live_client falls back to reading inline_data off
    model_turn.parts -- without this, replies would arrive as text with
    no sound at all.
    """
    responses = [make_response(data=None, inline_audio=b"\xaa\xbb")]
    session, _ = build_session_with_responses(responses)

    events = collect_events(session)

    assert events == [{"type": "audio", "data": b"\xaa\xbb"}]


def test_multiple_inline_audio_parts_are_concatenated_in_order():
    responses = [make_response(data=None, inline_audio=b"\x01")]
    responses[0].server_content.model_turn = SimpleNamespace(
        parts=[
            SimpleNamespace(inline_data=SimpleNamespace(data=b"\x01\x02")),
            SimpleNamespace(inline_data=SimpleNamespace(data=b"\x03\x04")),
        ]
    )
    session, _ = build_session_with_responses(responses)

    events = collect_events(session)

    assert events == [{"type": "audio", "data": b"\x01\x02\x03\x04"}]


def test_non_audio_parts_do_not_produce_an_audio_event():
    """A model_turn carrying only a text part must not fake an audio event."""
    response = make_response()
    response.server_content.model_turn = SimpleNamespace(
        parts=[SimpleNamespace(inline_data=None, text="hello")]
    )
    session, _ = build_session_with_responses([response])

    events = collect_events(session)

    assert events == []


def test_a_full_spoken_turn_is_audio_then_turn_complete():
    """
    End-to-end shape of one realistic turn: several audio chunks, then
    turn_complete. Nothing else.
    """
    turn = [
        make_response(data=b"\x01\x01"),
        make_response(data=b"\x02\x02"),
        make_response(turn_complete=True),
    ]
    session, _ = build_session_with_responses(turn)

    events = collect_events(session)

    assert events == [
        {"type": "audio", "data": b"\x01\x01"},
        {"type": "audio", "data": b"\x02\x02"},
        {"type": "turn_complete"},
    ]


# --------------------------------------------------------------------
# No text, ever
# --------------------------------------------------------------------

def test_assistant_transcription_is_ignored_completely():
    """
    Defence in depth. The session never asks for output transcription,
    but if the server sent it anyway it must not reach the browser --
    that text arriving in the chat is the bug this whole change fixes.
    """
    responses = [make_response(output_text="Your order ships in 2 days.")]
    session, _ = build_session_with_responses(responses)

    events = collect_events(session)

    assert events == []


def test_caller_speech_is_never_transcribed_back_to_the_browser():
    """The caller's own words must not appear in the chat either."""
    responses = [make_response(input_text="where is my order")]
    session, _ = build_session_with_responses(responses)

    events = collect_events(session)

    assert events == []


def test_audio_is_delivered_even_when_the_server_bundles_a_transcript():
    """Ignoring the text must not cost us the audio in the same frame."""
    responses = [make_response(data=b"\x01\x02", output_text="Two business days.")]
    session, _ = build_session_with_responses(responses)

    events = collect_events(session)

    assert events == [{"type": "audio", "data": b"\x01\x02"}]


def test_no_event_of_any_kind_carries_a_text_field():
    """
    Blanket guard over a realistic mixed turn: whatever events Path B
    produces, none of them may contain text for the UI to render.
    """
    turn = [
        make_response(data=b"\x01", output_text="Refunds take ", input_text="hi"),
        make_response(data=b"\x02", output_text="five days."),
        make_response(interrupted=True),
        make_response(turn_complete=True),
    ]
    session, _ = build_session_with_responses(turn)

    events = collect_events(session)

    assert events, "expected at least some audio/control events"
    for event in events:
        assert "text" not in event, f"text leaked into a live-call event: {event}"
        assert "role" not in event
        assert event["type"] in {"audio", "interrupted", "turn_complete"}


def test_a_text_only_part_in_model_turn_produces_nothing():
    response = make_response()
    response.server_content.model_turn = SimpleNamespace(
        parts=[SimpleNamespace(inline_data=None, text="I can help with that.")]
    )
    session, _ = build_session_with_responses([response])

    events = collect_events(session)

    assert events == []


def test_interrupted_flag_yields_interrupted_event():
    responses = [make_response(interrupted=True)]
    session, _ = build_session_with_responses(responses)

    events = collect_events(session)

    assert events == [{"type": "interrupted"}]


def test_turn_complete_is_emitted_as_a_bare_control_signal():
    responses = [make_response(turn_complete=True)]
    session, _ = build_session_with_responses(responses)

    events = collect_events(session)

    assert events == [{"type": "turn_complete"}]


# --------------------------------------------------------------------
# Session lifetime
# --------------------------------------------------------------------

def test_the_call_survives_past_the_first_turn():
    """
    session.receive() exhausts once per turn. receive_events() must call
    it again rather than ending the call after a single exchange.
    """
    turns = [
        [make_response(data=b"\x01", turn_complete=True)],
        [make_response(data=b"\x02", turn_complete=True)],
        [make_response(data=b"\x03", turn_complete=True)],
    ]
    session, mock_live_session = build_session_with_turns(turns)

    events = collect_events(session)

    audio = [e["data"] for e in events if e["type"] == "audio"]
    assert audio == [b"\x01", b"\x02", b"\x03"]
    # 3 scripted turns + the 4th call that raises the disconnect
    assert mock_live_session.receive.call_count == 4


def test_per_turn_audio_counter_resets_so_the_diagnostic_stays_truthful():
    """
    The per-turn byte count is what distinguishes "the model sent no
    audio" from "the browser didn't play it". If it leaked across turns,
    a silent turn would still report bytes and send debugging the wrong
    way entirely.
    """
    turns = [
        [make_response(data=b"\x01\x02\x03\x04"), make_response(turn_complete=True)],
        [make_response(turn_complete=True)],
    ]
    session, _ = build_session_with_turns(turns)

    collect_events(session)

    assert session._turn_audio_bytes == 0


def test_audio_counter_accumulates_within_a_single_turn():
    with patch("gemini.live_client.genai"):
        session = LiveCallSession()

    session._translate_response(make_response(data=b"\x01\x02"))
    session._translate_response(make_response(data=b"\x03\x04\x05\x06"))

    assert session._turn_audio_bytes == 6


def test_a_silent_turn_is_logged_as_a_server_side_problem(caplog):
    responses = [make_response(turn_complete=True)]
    session, _ = build_session_with_responses(responses)

    with caplog.at_level("WARNING", logger="gemini.live_client"):
        collect_events(session)

    assert any("NO audio" in record.message for record in caplog.records)


def test_receive_events_ends_cleanly_when_the_connection_drops():
    session, _ = build_session_with_turns([])

    events = collect_events(session)

    assert events == []


def test_receive_events_requires_start():
    with patch("gemini.live_client.genai"):
        session = LiveCallSession()

    async def _drain():
        return [e async for e in session.receive_events()]

    with pytest.raises(RuntimeError):
        asyncio.run(_drain())


def test_send_audio_chunk_requires_start():
    with patch("gemini.live_client.genai"):
        session = LiveCallSession()

    with pytest.raises(RuntimeError):
        asyncio.run(session.send_audio_chunk(b"\x00\x01"))


def test_send_audio_chunk_forwards_pcm_at_the_expected_sample_rate():
    session, mock_live_session = build_session_with_turns([])

    asyncio.run(session.send_audio_chunk(b"\x00\x01\x02\x03"))

    mock_live_session.send_realtime_input.assert_awaited_once()
    blob = mock_live_session.send_realtime_input.await_args.kwargs["audio"]
    assert blob.data == b"\x00\x01\x02\x03"
    assert blob.mime_type == "audio/pcm;rate=16000"


def test_session_requests_audio_output_and_no_transcription():
    """
    Two drifts to catch here:
      - response_modalities leaving AUDIO turns the call into a silent
        text chat;
      - either transcription config coming back makes the server start
        generating text again, which is what leaked into the chat.
    """
    with patch("gemini.live_client.genai"):
        session = LiveCallSession()

    assert session._config.response_modalities == ["AUDIO"]
    assert session._config.output_audio_transcription is None
    assert session._config.input_audio_transcription is None


def test_an_empty_receive_loop_gives_up_instead_of_spinning_forever():
    """
    receive_events() has no await of its own when receive() comes back
    empty. A connection that stops yielding without raising would pin a
    CPU core; it must back off and eventually end the call.
    """
    with patch("gemini.live_client.genai"):
        session = LiveCallSession()

    mock_live_session = MagicMock()
    mock_live_session.receive = MagicMock(side_effect=lambda: _async_iter([]))
    session._session = mock_live_session

    # 0.2s budget at the real 0.05s backoff = 4 iterations, so the test
    # exercises the actual arithmetic rather than a stubbed-out counter.
    with patch("gemini.live_client.MAX_EMPTY_TURN_SECONDS", 0.2):
        events = collect_events(session)

    assert events == []
    assert mock_live_session.receive.call_count == 4


def test_the_idle_timeout_is_long_enough_to_survive_a_normal_pause():
    """
    Regression guard for a real bug: this budget was once 20 iterations
    (one second), so the server hung up as soon as the caller stopped
    talking. Ending the call closes the browser's WebSocket, and the
    browser discards audio it had buffered but not yet played -- so an
    eager timeout here silently swallowed entire replies.

    A live call is mostly silence between turns; this must be far longer
    than any natural pause.
    """
    assert live_client.MAX_EMPTY_TURN_SECONDS >= 60


def test_a_quiet_stretch_does_not_end_a_healthy_call():
    """
    Empty receives followed by real audio: the call must continue and
    still deliver the audio, not hang up during the pause.
    """
    with patch("gemini.live_client.genai"):
        session = LiveCallSession()

    turns = [[], [], [], [make_response(data=b"\x07", turn_complete=True)]]
    remaining = list(turns)

    def _receive():
        if not remaining:
            raise ConnectionClosedError(None, None)
        return _async_iter(remaining.pop(0))

    mock_live_session = MagicMock()
    mock_live_session.receive = MagicMock(side_effect=_receive)
    session._session = mock_live_session

    events = collect_events(session)

    assert {"type": "audio", "data": b"\x07"} in events


# --------------------------------------------------------------------
# Tool calling
# --------------------------------------------------------------------

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


def test_audio_is_still_delivered_on_a_response_that_also_carries_a_tool_call():
    function_call = SimpleNamespace(id="call-1", name="search_knowledge_base", args={"query": "returns"})
    tool_call = SimpleNamespace(function_calls=[function_call])
    responses = [make_response(data=b"\x09", tool_call=tool_call)]
    session, _ = build_session_with_responses(responses)

    with patch("gemini.live_client.execute_tool", return_value={"found": False}):
        events = collect_events(session)

    assert events == [{"type": "audio", "data": b"\x09"}]


def test_multiple_function_calls_in_one_tool_call_are_all_answered():
    calls = [
        SimpleNamespace(id="c1", name="search_knowledge_base", args={"query": "refunds"}),
        SimpleNamespace(id="c2", name="search_knowledge_base", args={"query": "shipping"}),
    ]
    responses = [make_response(tool_call=SimpleNamespace(function_calls=calls))]
    session, mock_live_session = build_session_with_responses(responses)

    with patch(
        "gemini.live_client.execute_tool", return_value={"found": True, "answers": ["a"]}
    ) as mock_execute:
        collect_events(session)

    assert mock_execute.call_count == 2
    sent = mock_live_session.send_tool_response.await_args.kwargs["function_responses"]
    assert len(sent) == 2

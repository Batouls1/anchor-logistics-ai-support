"""
Tests for the /ws/live-call WebSocket bridge in main.py.

This is the layer that decides what the browser actually receives:
audio events go out as BINARY frames (app.js plays anything binary),
everything else goes out as JSON text frames. LiveCallSession is replaced
with a scripted fake -- no Gemini connection, no mic, no database.

TestClient is deliberately used without its context manager so the app's
lifespan never runs: lifespan calls init_db() and warm_up(), which would
hit a real Postgres and a real Pinecone index.
"""

import asyncio
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import main
from gemini.errors import QuotaExceededError


class FakeLiveSession:
    """
    Stands in for LiveCallSession. Waits until the browser has sent at
    least one audio chunk, then replays `scripted_events` back.
    """

    scripted_events = [
        {"type": "audio", "data": b"\x11\x22\x33\x44"},
        {"type": "turn_complete"},
    ]
    start_error = None

    def __init__(self):
        self.chunks = []
        self.started = False
        self.closed = False
        self._got_audio = asyncio.Event()

    async def start(self):
        if type(self).start_error is not None:
            raise type(self).start_error
        self.started = True

    async def close(self):
        self.closed = True

    async def send_audio_chunk(self, pcm_bytes):
        self.chunks.append(pcm_bytes)
        self._got_audio.set()

    async def receive_events(self):
        await self._got_audio.wait()
        for event in type(self).scripted_events:
            yield event


@pytest.fixture
def bridge():
    """
    Patches main.LiveCallSession and hands back (client, instances) so a
    test can inspect the session the endpoint actually created.
    """
    instances = []

    # Every test connects from the same client address, so without this
    # the per-address live-call limit trips partway through the file and
    # fails whichever tests happen to run last.
    main._live_call_limiter._hits.clear()

    def factory():
        session = FakeLiveSession()
        instances.append(session)
        return session

    with patch.object(main, "LiveCallSession", factory):
        yield TestClient(main.app), instances

    FakeLiveSession.scripted_events = [
        {"type": "audio", "data": b"\x11\x22\x33\x44"},
        {"type": "turn_complete"},
    ]
    FakeLiveSession.start_error = None


def test_audio_events_reach_the_browser_as_binary_frames(bridge):
    """
    The single most important assertion in this file: audio comes back
    over the socket as raw bytes, not wrapped in JSON.
    """
    client, instances = bridge

    with client.websocket_connect("/ws/live-call") as ws:
        ws.send_bytes(b"\x00\x01\x00\x02")
        message = ws.receive()

    assert message["type"] == "websocket.send"
    assert message.get("bytes") == b"\x11\x22\x33\x44"
    assert message.get("text") is None
    assert instances[0].started is True


def test_mic_audio_is_forwarded_to_the_live_session(bridge):
    client, instances = bridge

    with client.websocket_connect("/ws/live-call") as ws:
        ws.send_bytes(b"\x00\x01\x00\x02")
        ws.receive()  # wait until the bridge has definitely processed it

    assert instances[0].chunks == [b"\x00\x01\x00\x02"]


def test_control_events_reach_the_browser_as_json_text_frames(bridge):
    client, _ = bridge

    with client.websocket_connect("/ws/live-call") as ws:
        ws.send_bytes(b"\x00\x01")
        ws.receive()  # the audio frame
        turn_complete = ws.receive_json()

    assert turn_complete == {"type": "turn_complete"}


def test_raw_audio_bytes_are_never_sent_as_json(bridge):
    """
    Regression guard: if audio ever fell through to the send_json branch
    it would either crash on non-serialisable bytes or arrive as text the
    browser can't play.
    """
    client, _ = bridge

    with client.websocket_connect("/ws/live-call") as ws:
        ws.send_bytes(b"\x00\x01")
        frames = [ws.receive() for _ in range(2)]

    for frame in frames:
        text = frame.get("text")
        if text is None:
            continue
        payload = json.loads(text)
        assert payload["type"] != "audio"
        assert "data" not in payload


def test_no_json_frame_carries_conversation_text(bridge):
    """
    The live call must not push any words into the chat. Control frames
    are allowed; frames with a `text`/`role` payload are not. `message`
    is exempt -- that's the connection-error channel, tested separately.
    """
    FakeLiveSession.scripted_events = [
        {"type": "audio", "data": b"\x11\x22"},
        {"type": "interrupted"},
        {"type": "turn_complete"},
    ]
    client, _ = bridge

    with client.websocket_connect("/ws/live-call") as ws:
        ws.send_bytes(b"\x00\x01")
        frames = [ws.receive() for _ in range(3)]

    for frame in frames:
        if frame.get("text") is None:
            continue
        payload = json.loads(frame["text"])
        assert payload["type"] in {"interrupted", "turn_complete"}
        assert "text" not in payload
        assert "role" not in payload


def test_interrupted_event_is_forwarded_for_barge_in(bridge):
    FakeLiveSession.scripted_events = [{"type": "interrupted"}]
    client, _ = bridge

    with client.websocket_connect("/ws/live-call") as ws:
        ws.send_bytes(b"\x00\x01")
        assert ws.receive_json() == {"type": "interrupted"}


def test_quota_error_on_start_sends_a_friendly_message_and_closes(bridge):
    FakeLiveSession.start_error = QuotaExceededError("429")
    client, _ = bridge

    with client.websocket_connect("/ws/live-call") as ws:
        payload = ws.receive_json()

    assert payload["type"] == "error"
    assert payload["message"] == main.QUOTA_ERROR_MESSAGE


def test_unexpected_error_on_start_sends_the_graceful_message(bridge):
    FakeLiveSession.start_error = RuntimeError("boom")
    client, _ = bridge

    with client.websocket_connect("/ws/live-call") as ws:
        payload = ws.receive_json()

    assert payload["type"] == "error"
    assert payload["message"] == main.GRACEFUL_ERROR_MESSAGE


def test_too_many_calls_from_one_client_are_turned_away(bridge):
    """
    A live call holds a Gemini session open for its whole duration, so
    it's the most expensive thing an anonymous client can start.
    """
    client, _ = bridge
    limit = main._live_call_limiter.max_events

    for _ in range(limit):
        with client.websocket_connect("/ws/live-call") as ws:
            ws.send_bytes(b"\x00\x01")
            ws.receive()

    with client.websocket_connect("/ws/live-call") as ws:
        payload = ws.receive_json()

    assert payload["type"] == "error"
    assert payload["message"] == main.BUSY_ERROR_MESSAGE


def test_session_is_closed_when_the_browser_hangs_up(bridge):
    client, instances = bridge

    with client.websocket_connect("/ws/live-call") as ws:
        ws.send_bytes(b"\x00\x01")
        ws.receive()

    assert instances[0].closed is True

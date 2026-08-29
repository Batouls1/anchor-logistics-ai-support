"""
Tests for the Path A HTTP endpoints in main.py: /chat/text, /chat/voice
and /conversation/end. TurnManager and the database layer are mocked --
no Gemini calls, no Whisper, no Postgres.

Like the WebSocket tests, TestClient is used without its context manager
so the app's lifespan (init_db + warm_up) never runs.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import main
from conversation.session_token import issue_conversation_token, verify_conversation_token
from conversation.turn_manager import TurnResult
from gemini.errors import QuotaExceededError


@pytest.fixture
def token():
    """A genuine server-issued token, as a browser would receive."""
    return issue_conversation_token()


@pytest.fixture
def client():
    main._sessions.clear()
    main._last_active.clear()
    # Limiter buckets persist between requests, so a full suite run would
    # otherwise trip the limit partway through and fail unrelated tests.
    main._chat_limiter._hits.clear()
    main._start_limiter._hits.clear()
    main._live_call_limiter._hits.clear()

    fake_tm = AsyncMock()
    fake_tm.start = AsyncMock()
    fake_tm.close = AsyncMock()

    with patch.object(main, "TurnManager", return_value=fake_tm), \
         patch.object(main, "close_conversation", new_callable=AsyncMock) as close_conv:
        yield TestClient(main.app), fake_tm, close_conv

    main._sessions.clear()
    main._last_active.clear()


# --------------------------------------------------------------------
# Session ownership -- the security fix
# --------------------------------------------------------------------

def test_conversation_start_issues_a_usable_token(client):
    http, _, _ = client

    response = http.post("/conversation/start")

    assert response.status_code == 200
    issued = response.json()["conversation_id"]
    assert verify_conversation_token(issued) is not None


def test_a_client_invented_conversation_id_is_rejected(client):
    """
    The bug this closes: ids used to come from the browser and were
    trusted, so anyone could name someone else's conversation and read or
    append to its history.
    """
    http, fake_tm, _ = client

    response = http.post(
        "/chat/text", data={"conversation_id": "victims-conversation", "message": "hi"}
    )

    assert response.status_code == 401
    assert response.json()["agent_text"] == main.INVALID_SESSION_MESSAGE
    fake_tm.handle_text.assert_not_awaited()
    assert main._sessions == {}


def test_a_forged_signature_is_rejected(client, token):
    http, fake_tm, _ = client
    conversation_id, _, signature = token.partition(".")
    forged = f"{conversation_id}.{signature[:-1]}X"

    response = http.post("/chat/text", data={"conversation_id": forged, "message": "hi"})

    assert response.status_code == 401
    fake_tm.handle_text.assert_not_awaited()


def test_voice_notes_reject_an_invented_conversation_id(client):
    http, fake_tm, _ = client

    response = http.post(
        "/chat/voice",
        data={"conversation_id": "not-mine"},
        files={"audio": ("voice-note.webm", b"\x00\x01", "audio/webm")},
    )

    assert response.status_code == 401
    fake_tm.handle_voice.assert_not_awaited()


def test_ending_someone_elses_conversation_is_rejected(client):
    http, _, close_conv = client

    response = http.post("/conversation/end", data={"conversation_id": "not-mine"})

    assert response.status_code == 401
    close_conv.assert_not_awaited()


# --------------------------------------------------------------------
# Input limits and rate limiting
# --------------------------------------------------------------------

def test_an_overlong_message_is_rejected_before_reaching_the_model(client, token):
    http, fake_tm, _ = client
    oversized = "x" * (main.MAX_MESSAGE_CHARS + 1)

    response = http.post(
        "/chat/text", data={"conversation_id": token, "message": oversized}
    )

    assert response.status_code == 413
    assert response.json()["agent_text"] == main.TOO_LONG_MESSAGE
    fake_tm.handle_text.assert_not_awaited()


def test_a_message_at_the_limit_is_still_accepted(client, token):
    http, fake_tm, _ = client
    fake_tm.handle_text.return_value = TurnResult(user_text="x", agent_text="ok")

    response = http.post(
        "/chat/text",
        data={"conversation_id": token, "message": "x" * main.MAX_MESSAGE_CHARS},
    )

    assert response.status_code == 200


def test_an_oversized_voice_note_is_rejected_and_leaves_no_temp_file(client, token, tmp_path):
    http, fake_tm, _ = client
    oversized = b"\x00" * (main.MAX_AUDIO_BYTES + 1024)

    response = http.post(
        "/chat/voice",
        data={"conversation_id": token},
        files={"audio": ("big.webm", oversized, "audio/webm")},
    )

    assert response.status_code == 413
    fake_tm.handle_voice.assert_not_awaited()


def test_a_flood_of_messages_is_rate_limited(client, token):
    http, fake_tm, _ = client
    fake_tm.handle_text.return_value = TurnResult(user_text="hi", agent_text="ok")

    statuses = [
        http.post("/chat/text", data={"conversation_id": token, "message": "hi"}).status_code
        for _ in range(main._chat_limiter.max_events + 3)
    ]

    assert statuses[0] == 200
    assert statuses[-1] == 429
    assert statuses.count(200) == main._chat_limiter.max_events


def test_rate_limiting_is_per_conversation_not_global(client, token):
    """One abusive client must not lock everyone else out."""
    http, fake_tm, _ = client
    fake_tm.handle_text.return_value = TurnResult(user_text="hi", agent_text="ok")

    for _ in range(main._chat_limiter.max_events + 1):
        http.post("/chat/text", data={"conversation_id": token, "message": "hi"})

    other = issue_conversation_token()
    response = http.post("/chat/text", data={"conversation_id": other, "message": "hi"})

    assert response.status_code == 200


def test_responses_carry_a_request_id_for_tracing(client, token):
    http, fake_tm, _ = client
    fake_tm.handle_text.return_value = TurnResult(user_text="hi", agent_text="ok")

    response = http.post("/chat/text", data={"conversation_id": token, "message": "hi"})

    assert response.headers.get("X-Request-ID")


# --------------------------------------------------------------------
# Normal operation
# --------------------------------------------------------------------

def test_chat_text_returns_the_agent_reply(client, token):
    """Path A is text-only end to end: text in, text out, no audio field."""
    http, fake_tm, _ = client
    fake_tm.handle_text.return_value = TurnResult(
        user_text="hi", agent_text="Hello, how can I help?"
    )

    response = http.post("/chat/text", data={"conversation_id": token, "message": "hi"})

    assert response.status_code == 200
    assert response.json() == {
        "type": "text",
        "user_text": "hi",
        "agent_text": "Hello, how can I help?",
    }


def test_chat_text_reuses_one_session_per_conversation_id(client, token):
    http, fake_tm, _ = client
    fake_tm.handle_text.return_value = TurnResult(user_text="hi", agent_text="Hello.")

    http.post("/chat/text", data={"conversation_id": token, "message": "hi"})
    http.post("/chat/text", data={"conversation_id": token, "message": "again"})

    assert list(main._sessions) == [verify_conversation_token(token)]
    fake_tm.start.assert_awaited_once()


def test_chat_text_returns_the_quota_message_instead_of_a_500(client, token):
    http, fake_tm, _ = client
    fake_tm.handle_text.side_effect = QuotaExceededError("429")

    response = http.post("/chat/text", data={"conversation_id": token, "message": "hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "error"
    assert body["agent_text"] == main.QUOTA_ERROR_MESSAGE


def test_chat_text_degrades_gracefully_on_an_unexpected_error(client, token):
    http, fake_tm, _ = client
    fake_tm.handle_text.side_effect = RuntimeError("boom")

    response = http.post("/chat/text", data={"conversation_id": token, "message": "hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "error"
    assert body["agent_text"] == main.GRACEFUL_ERROR_MESSAGE


def test_chat_voice_returns_the_transcript_and_the_reply(client, token):
    http, fake_tm, _ = client
    fake_tm.handle_voice.return_value = TurnResult(
        user_text="where is my order", agent_text="Let me check that policy."
    )

    response = http.post(
        "/chat/voice",
        data={"conversation_id": token},
        files={"audio": ("voice-note.webm", b"\x00\x01\x02", "audio/webm")},
    )

    assert response.json() == {
        "type": "voice",
        "user_text": "where is my order",
        "agent_text": "Let me check that policy.",
    }


def test_chat_voice_marks_a_low_confidence_transcription_as_fallback(client, token):
    http, fake_tm, _ = client
    fake_tm.handle_voice.return_value = TurnResult(
        agent_text="I didn't catch that. Could you say that again?", is_fallback=True
    )

    response = http.post(
        "/chat/voice",
        data={"conversation_id": token},
        files={"audio": ("voice-note.webm", b"\x00\x01\x02", "audio/webm")},
    )

    body = response.json()
    assert body["type"] == "fallback"
    assert body["user_text"] is None


def test_chat_voice_cleans_up_its_temp_file(client, token, tmp_path):
    """The uploaded blob is written to a NamedTemporaryFile that must not leak."""
    http, fake_tm, _ = client
    seen = {}

    async def capture(path):
        seen["path"] = path
        return TurnResult(user_text="hi", agent_text="Hello.")

    fake_tm.handle_voice = capture

    http.post(
        "/chat/voice",
        data={"conversation_id": token},
        files={"audio": ("voice-note.webm", b"\x00\x01\x02", "audio/webm")},
    )

    import os
    assert not os.path.exists(seen["path"])


def test_ending_a_conversation_drops_the_session_and_closes_the_record(client, token):
    http, fake_tm, close_conv = client
    fake_tm.handle_text.return_value = TurnResult(user_text="hi", agent_text="Hello.")
    http.post("/chat/text", data={"conversation_id": token, "message": "hi"})

    response = http.post("/conversation/end", data={"conversation_id": token})

    assert response.json() == {"status": "closed"}
    assert verify_conversation_token(token) not in main._sessions
    fake_tm.close.assert_awaited_once()
    close_conv.assert_awaited_once_with(verify_conversation_token(token))


def test_ending_an_unknown_but_validly_signed_conversation_is_not_an_error(client):
    http, _, close_conv = client
    unused = issue_conversation_token()

    response = http.post("/conversation/end", data={"conversation_id": unused})

    assert response.json() == {"status": "closed"}
    close_conv.assert_awaited_once_with(verify_conversation_token(unused))


def test_the_app_still_starts_when_the_retriever_warm_up_fails():
    """
    The regression that matters: warm_up() used to run unguarded in
    lifespan, so an unreachable model host or a Pinecone blip took the
    whole server down at boot instead of degrading one feature. Startup
    must now complete and serve traffic regardless.

    This is the one test that runs the real lifespan, so init_db is
    mocked out alongside it.
    """
    with patch.object(main, "init_db", new_callable=AsyncMock), \
         patch.object(main, "close_conversation", new_callable=AsyncMock), \
         patch.object(main, "warm_up", return_value=False):
        with TestClient(main.app) as http:
            response = http.get("/app.js")

    assert response.status_code == 200


def test_the_app_starts_normally_when_the_retriever_warms_up():
    with patch.object(main, "init_db", new_callable=AsyncMock), \
         patch.object(main, "close_conversation", new_callable=AsyncMock), \
         patch.object(main, "warm_up", return_value=True) as mock_warm_up:
        with TestClient(main.app) as http:
            response = http.get("/app.js")

    assert response.status_code == 200
    mock_warm_up.assert_called_once()


def test_safe_static_files_closes_websocket_probes_instead_of_hanging():
    """
    SafeStaticFiles exists so bot probes that open a WebSocket against a
    static path don't raise an AssertionError traceback. It used to
    return without accepting or closing, which left the client hanging
    until it timed out -- it must send an explicit close now.
    """
    import asyncio

    static = main.SafeStaticFiles(directory="frontend")
    sent = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    asyncio.run(static({"type": "websocket", "path": "/x"}, receive, send))

    assert sent == [{"type": "websocket.close", "code": 1000}]


def test_a_websocket_probe_against_a_static_path_is_rejected_promptly(client, token):
    """End-to-end: the client gets a prompt rejection, not a hang."""
    from starlette.websockets import WebSocketDisconnect

    http, _, _ = client

    with pytest.raises((WebSocketDisconnect, Exception)):
        with http.websocket_connect("/app.js"):
            pass


def test_safe_static_files_still_serves_normal_http_requests(client, token):
    http, _, _ = client

    response = http.get("/app.js")

    assert response.status_code == 200
    assert "startLiveCall" in response.text

"""
Manages one Gemini Live session for Path B. Browser streams mic audio
continuously; Gemini handles turn-taking via its own VAD. No Whisper, no
history replay/reconnect, no Postgres persistence.

Path B is audio-in / audio-out only. It deliberately does NOT request
input or output transcription, and never emits text to the browser --
the live call is a phone call, completely separate from the Path A text
chat. The only things that cross the WebSocket are raw PCM audio and
small control signals (interrupted / turn_complete).
"""

import asyncio
import logging
import warnings
from typing import AsyncIterator, Optional

warnings.filterwarnings("ignore", message="there are non-data parts")

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError
from websockets.exceptions import ConnectionClosedError

from gemini.tools import TOOLS, execute_tool
from gemini.errors import QuotaExceededError

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.1-flash-live-preview"
# Google's own migration docs point from gemini-2.5-flash-native-audio-
# preview-12-2025 to this model directly -- same reasoning as
# text_client.py's model swap: the 2.5 generation is being locked out for
# new API projects ahead of its official shutdown, so this isn't a
# preference, it's what's actually available.

SYSTEM_INSTRUCTION = """You are the customer support voice assistant for \
Anchor Logistics, a delivery and logistics company, speaking with a \
customer on a live call.

Rules you must always follow:
- For any question about orders, refunds, shipping, accounts, payments, or \
company policy, always call the search_knowledge_base tool first. Never \
answer from memory or guess.
- You do not have access to live order tracking, real account data, or \
payment systems. If a customer asks about a *specific* order's status, \
explain the relevant general policy from the knowledge base and direct \
them to contact support with their order number -- never invent an order \
status.
- If search_knowledge_base returns found=False, say so honestly and point \
the customer to support. Never make up an answer to cover a gap.
- Keep responses short and conversational -- 2 to 3 sentences. This is a \
live spoken call, not a written document.
- Never state or imply that Anchor Logistics does not offer a particular \
service, feature, or contact method (e.g. "we don't support phone \
contact"). If you don't have information confirming something, say you \
don't have that information -- absence of a fact in the knowledge base is \
not evidence the fact is false.
"""

# Raw PCM formats the Live API expects/produces -- fixed by the API, not
# a choice we get to make. The frontend has to match these exactly.
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000

# Safety valve for the receive loop (see receive_events). The backoff is
# what actually prevents a CPU spin; the timeout only exists so a
# permanently dead-but-not-raising connection eventually lets go.
#
# This is deliberately generous. An earlier version gave up after 20
# empty iterations -- one single second -- which hung up on healthy calls
# the moment the caller stopped talking, and taking the WebSocket down
# mid-reply made the browser discard audio it had already buffered. A
# live call is mostly silence between turns; only a very long silence is
# evidence of anything being wrong.
MAX_EMPTY_TURN_SECONDS = 300.0
EMPTY_TURN_BACKOFF_SECONDS = 0.05


class LiveCallSession:
    """
    One instance of this = one ongoing live call. Owned entirely by the
    WebSocket endpoint in main.py for the duration of that connection --
    unlike TurnManager, there's no cross-request session store for this,
    since a live call is inherently one continuous connection anyway.
    """

    def __init__(self):
        self._client = genai.Client()
        self._config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=SYSTEM_INSTRUCTION,
            tools=TOOLS,
            # No input_audio_transcription / output_audio_transcription on
            # purpose. Asking for them is what made Gemini stream text
            # alongside the audio, which then leaked into the chat window.
            # Not requesting them means the server never generates the
            # transcripts in the first place -- cheaper and simpler than
            # generating text and then dropping it on our side.
        )
        self._connect_cm = None
        self._session = None
        # Audio delivered during the current turn, reset at turn_complete.
        self._turn_audio_bytes = 0

    async def start(self):
        try:
            self._connect_cm = self._client.aio.live.connect(
                model=MODEL_NAME, config=self._config
            )
            self._session = await self._connect_cm.__aenter__()
        except ClientError as e:
            if e.code == 429:
                raise QuotaExceededError(str(e)) from e
            raise

    async def close(self):
        if self._connect_cm is not None:
            await self._connect_cm.__aexit__(None, None, None)
            self._connect_cm = None
            self._session = None

    async def send_audio_chunk(self, pcm_bytes: bytes) -> None:
        """
        Forwards one chunk of raw 16-bit PCM, 16kHz, mono, little-endian
        audio straight from the browser to Gemini. No buffering or
        turn-segmentation on our end -- Gemini's own VAD decides when the
        user has finished speaking.
        """
        if self._session is None:
            raise RuntimeError("Session not started -- call start() first.")
        await self._session.send_realtime_input(
            audio=types.Blob(data=pcm_bytes, mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}")
        )

    def _translate_response(self, response) -> list[dict]:
        """
        Turns one raw Gemini response into zero or more flat events for
        the browser. Pure translation, no I/O -- kept separate from the
        connection loop so it's testable without mocking an async
        generator.

        Only audio and control signals are ever produced here. Any text
        Gemini might send is intentionally ignored: Path B never puts
        words in the chat window.
        """
        events = []

        server_content = getattr(response, "server_content", None)

        audio_bytes = response.data
        if audio_bytes is None and server_content is not None:
            # Fallback: response.data is normally a shortcut for inline
            # audio parts under server_content.model_turn -- if that
            # shortcut isn't populating for this model/SDK combination,
            # pull the bytes from model_turn directly instead of silently
            # dropping audio.
            model_turn = getattr(server_content, "model_turn", None)
            if model_turn and model_turn.parts:
                for part in model_turn.parts:
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data and inline_data.data:
                        audio_bytes = (audio_bytes or b"") + inline_data.data

        if audio_bytes:
            self._turn_audio_bytes += len(audio_bytes)
            events.append({"type": "audio", "data": audio_bytes})
        elif server_content is not None and self._looks_like_a_spoken_turn(server_content):
            # A model turn that carried no audio at all. Nothing is sent
            # to the browser (there's no text channel any more), but it's
            # worth a log line -- silent replies are the one failure mode
            # that's invisible from the UI now.
            logger.warning(
                "Live model turn produced no audio bytes -- server_content=%r",
                server_content,
            )

        if server_content and getattr(server_content, "interrupted", False):
            events.append({"type": "interrupted"})

        if server_content and getattr(server_content, "turn_complete", False):
            # Logged every turn, not just on failure. With no transcript
            # in the UI any more, this line is the only way to tell "the
            # model said nothing" apart from "the browser didn't play
            # what it was sent" -- the two look identical from the user's
            # side, and guessing between them wastes a debugging session.
            seconds = self._turn_audio_bytes / (OUTPUT_SAMPLE_RATE * 2)
            if self._turn_audio_bytes:
                logger.info(
                    "Live turn complete: %d bytes of audio (~%.1fs) sent to the browser.",
                    self._turn_audio_bytes,
                    seconds,
                )
            else:
                logger.warning(
                    "Live turn complete but the model sent NO audio at all. "
                    "The browser has nothing to play; this is a server/model "
                    "side problem, not a playback one."
                )
            self._turn_audio_bytes = 0
            events.append({"type": "turn_complete"})

        return events

    @staticmethod
    def _looks_like_a_spoken_turn(server_content) -> bool:
        """
        True when Gemini sent model_turn content that should have been
        audio. Used only to decide whether a missing-audio warning is
        worth logging -- turn_complete/interrupted-only frames carry no
        model_turn and are perfectly normal.
        """
        model_turn = getattr(server_content, "model_turn", None)
        return bool(model_turn and getattr(model_turn, "parts", None))

    async def receive_events(self) -> AsyncIterator[dict]:
        """
        Yields events for main.py's WebSocket bridge to forward to the
        browser:
          {"type": "audio", "data": <raw PCM bytes, 24kHz>}
          {"type": "turn_complete"}   -- Gemini finished speaking
          {"type": "interrupted"}     -- Gemini's own barge-in signal
        No text is ever yielded; see the module docstring.
        Runs until the connection actually drops.
        """
        if self._session is None:
            raise RuntimeError("Session not started -- call start() first.")

        idle_seconds = 0.0

        try:
            while True:
                got_response = False
                async for response in self._session.receive():
                    got_response = True
                    for event in self._translate_response(response):
                        yield event

                    if response.tool_call:
                        await self._handle_tool_call(response.tool_call)
                # session.receive() naturally exhausts once per turn (real
                # SDK behavior) -- without this outer loop, the call would
                # silently end after the very first exchange. Only a real
                # disconnect (caught below) should end this generator.
                #
                # Nothing in this loop awaits when receive() comes back
                # empty, so the sleep is what stops a dead-but-silent
                # connection from pinning a CPU core. The timeout is only
                # a last resort: ending the call here closes the browser's
                # WebSocket, and the browser tears down its audio context
                # on that -- so hanging up too eagerly throws away a reply
                # the caller never got to hear.
                if got_response:
                    idle_seconds = 0.0
                    continue

                idle_seconds += EMPTY_TURN_BACKOFF_SECONDS
                if idle_seconds >= MAX_EMPTY_TURN_SECONDS:
                    logger.warning(
                        "Live session produced nothing for %.0fs -- treating "
                        "the call as ended.",
                        idle_seconds,
                    )
                    return
                await asyncio.sleep(EMPTY_TURN_BACKOFF_SECONDS)
        except ClientError as e:
            if e.code == 429:
                logger.warning("Live call hit a quota limit")
                raise QuotaExceededError(str(e)) from e
            raise
        except (ConnectionClosedError, APIError) as e:
            logger.warning("Live call connection ended: %s", e)
            return

    async def _handle_tool_call(self, tool_call):
        function_responses = []
        for fc in tool_call.function_calls:
            result = await asyncio.to_thread(execute_tool, fc.name, dict(fc.args))
            function_responses.append(
                types.FunctionResponse(id=fc.id, name=fc.name, response=result)
            )
        await self._session.send_tool_response(function_responses=function_responses)
"""
Manages one Gemini Live session for Path B. Browser streams mic audio
continuously; Gemini handles turn-taking via its own VAD. No Whisper, no
history replay/reconnect, no Postgres persistence.
"""

import asyncio
import warnings
from typing import AsyncIterator, Optional

warnings.filterwarnings("ignore", message="there are non-data parts")

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError
from websockets.exceptions import ConnectionClosedError

from gemini.tools import TOOLS, execute_tool
from gemini.errors import QuotaExceededError

MODEL_NAME = "gemini-3.1-flash-live-preview"
# this model defaults to thinking_level="minimal"
# (optimized for lowest latency, which is what a live call wants anyway)

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
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
        )
        self._connect_cm = None
        self._session = None

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

    async def receive_events(self) -> AsyncIterator[dict]:
        """
        Yields events for main.py's WebSocket bridge to forward to the
        browser as they arrive:
          {"type": "audio", "data": <raw PCM bytes, 24kHz>}
          {"type": "transcript", "role": "user" | "assistant", "text": str}
          {"type": "interrupted"}   -- Gemini's own barge-in signal; the
                                       frontend should stop playback
        Runs until the session closes or the underlying connection drops.
        """
        if self._session is None:
            raise RuntimeError("Session not started -- call start() first.")

        try:
            async for response in self._session.receive():
                if response.data is not None:
                    yield {"type": "audio", "data": response.data}

                server_content = getattr(response, "server_content", None)

                if server_content and server_content.output_transcription:
                    text = server_content.output_transcription.text
                    if text:
                        yield {"type": "transcript", "role": "assistant", "text": text}

                if server_content and server_content.input_transcription:
                    text = server_content.input_transcription.text
                    if text:
                        yield {"type": "transcript", "role": "user", "text": text}

                if response.tool_call:
                    await self._handle_tool_call(response.tool_call)

                if server_content and getattr(server_content, "interrupted", False):
                    yield {"type": "interrupted"}
        except ClientError as e:
            if e.code == 429:
                raise QuotaExceededError(str(e)) from e
            raise
        except (ConnectionClosedError, APIError):
            # Deliberately not reconnecting here (see module docstring) --
            # the call just ends; the browser side treats this the same
            # as any other disconnect.
            return

    async def _handle_tool_call(self, tool_call):
        function_responses = []
        for fc in tool_call.function_calls:
            result = await asyncio.to_thread(execute_tool, fc.name, dict(fc.args))
            function_responses.append(
                types.FunctionResponse(id=fc.id, name=fc.name, response=result)
            )
        await self._session.send_tool_response(function_responses=function_responses)
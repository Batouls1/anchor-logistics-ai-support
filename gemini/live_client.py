"""
Manages one Gemini Live session per conversation.

We send text in (the Whisper transcript, or typed text) and get audio out --
we never send raw audio to Gemini, Whisper is the only "ears" in this system.
The session stays open across multiple turns so conversation context persists,
instead of reconnecting fresh for every message.

This path is used for VOICE turns only -- typed messages go through
gemini/text_client.py instead, since Live's response_modalities is session-
wide and would otherwise force audio synthesis on every text turn too.
"""

import asyncio
import warnings
from typing import Awaitable, Callable, Optional

warnings.filterwarnings("ignore", message="there are non-data parts")

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError
from websockets.exceptions import ConnectionClosedError

from gemini.tools import TOOLS, execute_tool
from gemini.errors import QuotaExceededError

MODEL_NAME = "gemini-2.5-flash-native-audio-preview-12-2025"

SYSTEM_INSTRUCTION = """You are the customer support voice assistant for \
Anchor Logistics, a delivery and logistics company.

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
spoken conversation, not a written document.
- Never state or imply that Anchor Logistics does not offer a particular \
service, feature, or contact method (e.g. "we don't support phone \
contact"). If you don't have information confirming something, say you \
don't have that information -- absence of a fact in the knowledge base is \
not evidence the fact is false.
"""

# A history replay entry looks like {"user_text": str | None, "agent_text": str | None}
HistoryProvider = Callable[[], Awaitable[list[dict]]]


class LiveSession:
    """One instance of this = one ongoing voice conversation."""

    def __init__(self, history_provider: Optional[HistoryProvider] = None):
        # Optional callback that fetches recent turns from the database.
        # Called only on reconnect so a fresh Gemini-side session isn't starting 
        # completely blank after an idle-timeout disconnect. 
        self._history_provider = history_provider

        self._client = genai.Client()
        self._config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=SYSTEM_INSTRUCTION,
            tools=TOOLS,
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )
        self._connect_cm = None
        self._session = None

    async def start(self):
        self._connect_cm = self._client.aio.live.connect(
            model=MODEL_NAME, config=self._config
        )
        self._session = await self._connect_cm.__aenter__()

    async def close(self):
        if self._connect_cm is not None:
            await self._connect_cm.__aexit__(None, None, None)
            self._connect_cm = None
            self._session = None

    async def send_message(self, text: str) -> dict:
        """
        Sends one text turn (e.g. a Whisper transcript) and returns both the
        spoken response and its text transcript:
          {"audio": <raw PCM bytes, 16-bit/24kHz/mono>, "text": <str>}
        """
        try:
            return await self._send_once(text)
        except ClientError as e:
            if e.code == 429:
                raise QuotaExceededError(str(e)) from e
            raise
        except (ConnectionClosedError, APIError):
            # Gemini Live closes idle WebSocket connections after a period
            # of inactivity (e.g. a keepalive ping timeout). Reconnect once
            # and retry, rather than permanently failing every subsequent
            # turn on this conversation.
            await self.close()
            await self.start()

            history = None
            if self._history_provider is not None:
                history = await self._history_provider()

            return await self._send_once(text, history=history)

    async def _send_once(self, text: str, history: list[dict] | None = None) -> dict:
        if self._session is None:
            raise RuntimeError("Session not started -- call start() first.")

        turns = self._build_turns(text, history)
        await self._session.send_client_content(turns=turns)

        audio_chunks = []
        text_parts = []

        async for response in self._session.receive():
            if response.data is not None:
                audio_chunks.append(response.data)

            server_content = getattr(response, "server_content", None)
            if server_content and server_content.output_transcription:
                text_parts.append(server_content.output_transcription.text)

            if response.tool_call:
                await self._handle_tool_call(response.tool_call)

            if server_content and getattr(server_content, "turn_complete", False):
                break

        return {
            "audio": b"".join(audio_chunks),
            "text": "".join(text_parts),
        }

    @staticmethod
    def _build_turns(text: str, history: list[dict] | None) -> list[dict]:
        """
        Builds the turns list for send_client_content. Without history,
        this is just the one new user turn (original behavior, unchanged).
        With history (only populated on reconnect), prior turns are
        prepended as alternating user/model content so Gemini has some
        awareness of the conversation before this point, instead of
        starting completely fresh after a reconnect.
        """
        turns = []
        if history:
            for turn in history:
                if turn.get("user_text"):
                    turns.append(
                        {"role": "user", "parts": [{"text": turn["user_text"]}]}
                    )
                if turn.get("agent_text"):
                    turns.append(
                        {"role": "model", "parts": [{"text": turn["agent_text"]}]}
                    )
        turns.append({"role": "user", "parts": [{"text": text}]})
        return turns

    async def _handle_tool_call(self, tool_call):
        function_responses = []
        for fc in tool_call.function_calls:
            result = await asyncio.to_thread(execute_tool, fc.name, dict(fc.args))
            function_responses.append(
                types.FunctionResponse(id=fc.id, name=fc.name, response=result)
            )
        await self._session.send_tool_response(function_responses=function_responses)


if __name__ == "__main__":
    import wave

    async def repl():
        session = LiveSession()
        await session.start()
        turn = 0
        try:
            while True:
                text = input("\nYou ('q' to quit): ")
                if text.lower() == "q":
                    break

                try:
                    result = await session.send_message(text)
                except QuotaExceededError:
                    print(
                        "\n[Quota exceeded for this model. Wait for the "
                        "reset window, or switch models, then try again.]"
                    )
                    continue

                turn += 1
                filename = f"response_{turn}.wav"

                with wave.open(filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(result["audio"])

                print(f"Gemini (text): {result['text']}")
                print(f"Gemini responded -- saved audio to {filename}")
        finally:
            await session.close()

    asyncio.run(repl())
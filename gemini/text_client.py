"""
Non-Live path for typed messages and voice notes -- both share this
TextSession/history. Plain generate_content, not Live, since audio
output is session-wide, not per-turn.
"""

import asyncio

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from gemini.tools import TOOLS, execute_tool
from gemini.errors import QuotaExceededError

MODEL_NAME = "gemini-3.1-flash-lite"

SYSTEM_INSTRUCTION = """You are the customer support assistant for \
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
- Keep responses short and conversational -- 2 to 3 sentences.
- Never state or imply that Anchor Logistics does not offer a particular \
service, feature, or contact method (e.g. "we don't support phone \
contact"). If you don't have information confirming something, say you \
don't have that information -- absence of a fact in the knowledge base is \
not evidence the fact is false.
"""


class TextSession:
    """
    One instance of this = one ongoing Path A conversation (typed
    messages and voice-note transcripts alike). Owned by TurnManager,
    one per conversation_id.
    """

    def __init__(self):
        self._client = genai.Client()
        self._history: list[types.Content] = []
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=TOOLS,
        )

    def prime_history(self, turns: list[tuple[str, str]]) -> None:
        """
        Replays previously stored exchanges into this session's history,
        so a conversation picked up by a different process (or after a
        restart) still remembers what was said.

        Only the plain text of each turn is replayed -- tool calls and
        their responses aren't reconstructed. The model doesn't need the
        old retrieval steps, just what was actually said, and it calls
        the tool again for anything it needs to look up.
        """
        for user_text, agent_text in turns:
            self._history.append(
                types.Content(role="user", parts=[types.Part(text=user_text)])
            )
            self._history.append(
                types.Content(role="model", parts=[types.Part(text=agent_text)])
            )

    async def send_message(self, text: str) -> str:
        self._history.append(
            types.Content(role="user", parts=[types.Part(text=text)])
        )

        response = await self._generate()

        # Manual function-calling loop: Gemini may request the RAG tool
        # one or more times before giving a final text answer.
        while response.function_calls:
            self._history.append(response.candidates[0].content)

            function_response_parts = []
            for fc in response.function_calls:
                result = await asyncio.to_thread(execute_tool, fc.name, dict(fc.args))
                function_response_parts.append(
                    types.Part.from_function_response(name=fc.name, response=result)
                )

            self._history.append(
                types.Content(role="user", parts=function_response_parts)
            )

            response = await self._generate()

        text_out = response.text or ""
        self._history.append(
            types.Content(role="model", parts=[types.Part(text=text_out)])
        )
        return text_out

    async def _generate(self):
        try:
            return await self._client.aio.models.generate_content(
                model=MODEL_NAME, contents=self._history, config=self._config
            )
        except ClientError as e:
            if e.code == 429:
                raise QuotaExceededError(str(e)) from e
            raise


if __name__ == "__main__":
    # Minimal manual test harness
    async def repl():
        session = TextSession()
        while True:
            text = input("\nYou ('q' to quit): ")
            if text.lower() == "q":
                break
            try:
                reply = await session.send_message(text)
                print(f"Gemini: {reply}")
            except QuotaExceededError:
                print(
                    "\n[Quota exceeded for this model. Wait for the reset "
                    "window, or switch models, then try again.]"
                )

    asyncio.run(repl())
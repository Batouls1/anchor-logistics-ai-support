"""
Gemini function-calling tool wrapping rag/retriever.py, shared by
TextSession and LiveCallSession. Retriever is built lazily on first call,
not at import time, so importing this module doesn't require live
Pinecone credentials.
"""

import logging

from rag.retriever import Retriever

logger = logging.getLogger(__name__)

_retriever: Retriever | None = None


def warm_up() -> bool:
    """
    Forces the Retriever to build now instead of on first tool call.
    Called once at server startup -- without this, the first real tool
    call (cold model load + Pinecone connect, several seconds) can land
    mid-conversation, which is what broke the live call: Gemini Live
    won't wait that long for a function response.

    Best-effort by design: returns True if the retriever is ready, False
    if it couldn't be built, and never raises. Warming up is an
    optimisation, so a failure here must degrade one feature rather than
    stop the whole server from starting -- an unreachable model host or a
    Pinecone blip used to take the entire app down at boot. _retriever is
    left as None on failure, so the next tool call simply retries.
    """
    try:
        _get_retriever()
        return True
    except Exception:
        logger.exception(
            "Retriever warm-up failed -- knowledge base lookups will retry "
            "lazily on the first tool call."
        )
        return False


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


RAG_TOOL_DECLARATION = {
    "name": "search_knowledge_base",
    "description": (
        "Search Anchor Logistics' knowledge base for information about orders, "
        "refunds, cancellations, shipping, delivery, accounts, payments, and "
        "company policies. Always call this before answering any question "
        "about the company or its policies -- never answer from memory."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "The customer's question, rephrased as a clear, "
                                "standalone search query.",
            }
        },
        "required": ["query"],
    },
}

TOOLS = [{"function_declarations": [RAG_TOOL_DECLARATION]}]


def execute_tool(name: str, args: dict) -> dict:
    """
    Called by text_client.py / live_client.py whenever Gemini invokes a
    tool. Returns a plain dict -- this becomes the tool's function
    response, which Gemini reads before composing its reply.
    """
    if name != "search_knowledge_base":
        return {"error": f"Unknown tool: {name}"}

    query = args.get("query", "").strip()
    if not query:
        return {"found": False, "message": "No query provided."}

    try:
        results = _get_retriever().search(query)
    except Exception:
        # Reaching here means the retriever still can't be built (or the
        # search itself failed) after warm-up already gave up. Returning a
        # tool response instead of raising matters most on a live call:
        # an exception here would propagate out of the receive loop and
        # drop the call mid-sentence. This way the model just explains it
        # can't look something up, and the conversation continues.
        logger.exception("Knowledge base search failed for query %r", query)
        return {
            "found": False,
            "message": (
                "The knowledge base is temporarily unavailable. Tell the "
                "customer honestly that you can't look that up right now "
                "and suggest they contact support directly. Do NOT state "
                "or imply that a service, feature, or contact method "
                "doesn't exist -- this is a temporary technical problem on "
                "your side, not an answer about the company."
            ),
        }

    if not results:
        # Honesty fallback
        return {
            "found": False,
            "message": (
                "No relevant information was found in the knowledge base. "
                "Tell the customer honestly that you don't currently have "
                "that specific information, and suggest contacting support "
                "directly. Do NOT state or imply that a service, feature, "
                "or contact method doesn't exist or isn't offered -- you "
                "only know that you don't have the answer right now, not "
                "that the answer is 'no'."
            ),
        }

    return {
        "found": True,
        "answers": [r["answer"] for r in results],
    }
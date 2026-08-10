"""
Gemini function-calling tool wrapping rag/retriever.py, shared by
TextSession and LiveCallSession. Retriever is built lazily on first call,
not at import time, so importing this module doesn't require live
Pinecone credentials.
"""

from rag.retriever import Retriever

_retriever: Retriever | None = None


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

    results = _get_retriever().search(query)

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
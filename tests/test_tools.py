"""
Tests for the shared RAG tool (gemini/tools.py). The Retriever is mocked
-- no Pinecone connection, no embedding model load.
"""

from unittest.mock import MagicMock, patch

import gemini.tools as tools


def _reset_retriever():
    tools._retriever = None


def test_execute_tool_returns_answers_when_the_retriever_finds_matches():
    _reset_retriever()
    retriever = MagicMock()
    retriever.search.return_value = [
        {"answer": "Refunds take 5 business days."},
        {"answer": "Refunds go back to the original payment method."},
    ]

    with patch.object(tools, "Retriever", return_value=retriever):
        result = tools.execute_tool("search_knowledge_base", {"query": "refund policy"})

    assert result["found"] is True
    assert result["answers"] == [
        "Refunds take 5 business days.",
        "Refunds go back to the original payment method.",
    ]
    retriever.search.assert_called_once_with("refund policy")


def test_execute_tool_reports_not_found_without_claiming_the_answer_is_no():
    """
    The honesty rule: an empty knowledge base result must not let the
    model tell a customer that a service doesn't exist.
    """
    _reset_retriever()
    retriever = MagicMock()
    retriever.search.return_value = []

    with patch.object(tools, "Retriever", return_value=retriever):
        result = tools.execute_tool("search_knowledge_base", {"query": "do you ship to Mars"})

    assert result["found"] is False
    assert "don't" in result["message"] or "not" in result["message"]


def test_execute_tool_rejects_an_unknown_tool_name():
    _reset_retriever()
    with patch.object(tools, "Retriever") as retriever_cls:
        result = tools.execute_tool("delete_everything", {"query": "x"})

    assert "error" in result
    retriever_cls.assert_not_called()


def test_execute_tool_handles_a_blank_query_without_searching():
    _reset_retriever()
    with patch.object(tools, "Retriever") as retriever_cls:
        result = tools.execute_tool("search_knowledge_base", {"query": "   "})

    assert result["found"] is False
    retriever_cls.assert_not_called()


def test_warm_up_builds_the_retriever_once_and_is_idempotent():
    """
    warm_up() exists so the first real tool call isn't a multi-second cold
    start mid-call -- Gemini Live won't wait that long for a function
    response. It must build eagerly, and only once.
    """
    _reset_retriever()
    with patch.object(tools, "Retriever", return_value=MagicMock()) as retriever_cls:
        tools.warm_up()
        assert retriever_cls.call_count == 1

        tools.warm_up()
        tools.execute_tool("search_knowledge_base", {"query": "refunds"})
        assert retriever_cls.call_count == 1

    _reset_retriever()


def test_warm_up_reports_failure_instead_of_raising():
    """
    A retriever that can't be built (unreachable model host, Pinecone
    blip) must not take the server down at boot -- warm_up reports the
    failure and lets startup continue.
    """
    _reset_retriever()
    with patch.object(tools, "Retriever", side_effect=RuntimeError("HF unreachable")):
        assert tools.warm_up() is False

    _reset_retriever()


def test_warm_up_reports_success():
    _reset_retriever()
    with patch.object(tools, "Retriever", return_value=MagicMock()):
        assert tools.warm_up() is True

    _reset_retriever()


def test_a_failed_warm_up_is_retried_on_the_next_tool_call():
    """
    warm_up leaves _retriever as None on failure, so the app self-heals:
    the first lookup after the outage builds it for real.
    """
    _reset_retriever()

    with patch.object(tools, "Retriever", side_effect=RuntimeError("down")):
        assert tools.warm_up() is False

    retriever = MagicMock()
    retriever.search.return_value = [{"answer": "Refunds take 5 business days."}]
    with patch.object(tools, "Retriever", return_value=retriever):
        result = tools.execute_tool("search_knowledge_base", {"query": "refunds"})

    assert result["found"] is True
    assert result["answers"] == ["Refunds take 5 business days."]

    _reset_retriever()


def test_execute_tool_degrades_gracefully_when_the_retriever_cannot_be_built():
    """
    On a live call an exception here would propagate out of the receive
    loop and drop the call mid-sentence. It must come back as a normal
    tool response instead.
    """
    _reset_retriever()
    with patch.object(tools, "Retriever", side_effect=RuntimeError("still down")):
        result = tools.execute_tool("search_knowledge_base", {"query": "refunds"})

    assert result["found"] is False
    assert "message" in result
    _reset_retriever()


def test_execute_tool_degrades_gracefully_when_the_search_itself_fails():
    _reset_retriever()
    retriever = MagicMock()
    retriever.search.side_effect = RuntimeError("pinecone timeout")

    with patch.object(tools, "Retriever", return_value=retriever):
        result = tools.execute_tool("search_knowledge_base", {"query": "refunds"})

    assert result["found"] is False
    assert "message" in result
    _reset_retriever()


def test_the_outage_message_never_implies_the_company_lacks_a_service():
    """
    Same honesty rule as the empty-results path: "I can't check right
    now" must not become "we don't offer that".
    """
    _reset_retriever()
    with patch.object(tools, "Retriever", side_effect=RuntimeError("down")):
        result = tools.execute_tool("search_knowledge_base", {"query": "phone support"})

    message = result["message"].lower()
    assert "temporar" in message
    assert "do not state" in message or "don't state" in message
    _reset_retriever()


def test_tool_declaration_shape_is_what_gemini_expects():
    assert tools.TOOLS == [{"function_declarations": [tools.RAG_TOOL_DECLARATION]}]
    assert tools.RAG_TOOL_DECLARATION["name"] == "search_knowledge_base"
    assert tools.RAG_TOOL_DECLARATION["parameters"]["required"] == ["query"]

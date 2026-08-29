"""
Tests for signed conversation tokens.

The property that matters: a client cannot name a conversation the server
didn't issue. Before this existed, conversation ids were generated in the
browser and trusted, so anyone could read or append to another
conversation just by supplying its id.
"""

from conversation.session_token import (
    issue_conversation_token,
    verify_conversation_token,
)


def test_an_issued_token_verifies_and_yields_its_conversation_id():
    token = issue_conversation_token()

    conversation_id = verify_conversation_token(token)

    assert conversation_id
    assert token.startswith(conversation_id + ".")


def test_every_token_is_unique():
    tokens = {issue_conversation_token() for _ in range(200)}

    assert len(tokens) == 200


def test_a_made_up_id_is_rejected():
    """The whole point: a client can't invent its own conversation id."""
    assert verify_conversation_token("some-conversation-i-made-up") is None


def test_another_conversations_id_cannot_be_borrowed_with_a_valid_signature():
    """
    Taking a real token and swapping in a different id must fail -- the
    signature covers the id, so it won't match.
    """
    victim = verify_conversation_token(issue_conversation_token())
    attacker_token = issue_conversation_token()
    _attacker_id, _, attacker_signature = attacker_token.partition(".")

    forged = f"{victim}.{attacker_signature}"

    assert verify_conversation_token(forged) is None


def test_a_tampered_signature_is_rejected():
    conversation_id, _, signature = issue_conversation_token().partition(".")
    tampered = signature[:-1] + ("A" if signature[-1] != "A" else "B")

    assert verify_conversation_token(f"{conversation_id}.{tampered}") is None


def test_malformed_input_is_rejected_rather_than_raising():
    """Anything can arrive over the network; none of it should blow up."""
    for value in ["", ".", "no-dot", "a.", ".b", None, 12345, [], {}]:
        assert verify_conversation_token(value) is None

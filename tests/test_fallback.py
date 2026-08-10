"""
Tests for conversation/fallback.py's retry-counting and escalating-message
logic. No mocking needed -- FallbackHandler has no external dependencies.
"""

from conversation.fallback import FallbackHandler


def test_first_failure_returns_soft_retry_message():
    handler = FallbackHandler(max_retries=1)

    message = handler.record_failure()

    assert message == "I didn't catch that. Could you say that again?"


def test_repeated_failure_beyond_max_retries_escalates():
    handler = FallbackHandler(max_retries=1)

    handler.record_failure()               # 1st failure -- within max_retries
    message = handler.record_failure()     # 2nd failure -- exceeds max_retries

    assert message == (
        "I'm still having trouble understanding. Please type your question "
        "so I can help you best."
    )


def test_max_retries_of_zero_escalates_on_first_failure():
    handler = FallbackHandler(max_retries=0)

    message = handler.record_failure()

    assert message == (
        "I'm still having trouble understanding. Please type your question "
        "so I can help you best."
    )


def test_reset_clears_retry_count_after_success():
    handler = FallbackHandler(max_retries=1)

    handler.record_failure()   # retry_count == 1
    handler.reset()

    message = handler.record_failure()   # should behave like a fresh 1st failure

    assert message == "I didn't catch that. Could you say that again?"
    assert handler.retry_count == 1


def test_retry_count_is_per_instance_not_shared():
    handler_a = FallbackHandler(max_retries=1)
    handler_b = FallbackHandler(max_retries=1)

    handler_a.record_failure()
    handler_a.record_failure()

    assert handler_a.retry_count == 2
    assert handler_b.retry_count == 0
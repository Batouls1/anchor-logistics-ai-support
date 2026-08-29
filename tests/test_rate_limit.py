"""
Tests for the sliding-window rate limiter. Time is controlled by patching
the clock rather than sleeping, so the suite stays fast and doesn't
depend on wall-clock timing.
"""

from unittest.mock import patch

from conversation.rate_limit import RateLimiter


def test_requests_under_the_limit_are_allowed():
    limiter = RateLimiter(max_events=3, window_seconds=60)

    assert [limiter.allow("client") for _ in range(3)] == [True, True, True]


def test_the_request_over_the_limit_is_blocked():
    limiter = RateLimiter(max_events=3, window_seconds=60)
    for _ in range(3):
        limiter.allow("client")

    assert limiter.allow("client") is False


def test_each_key_gets_its_own_budget():
    """One noisy client must not rate-limit everyone else."""
    limiter = RateLimiter(max_events=2, window_seconds=60)

    limiter.allow("noisy")
    limiter.allow("noisy")

    assert limiter.allow("noisy") is False
    assert limiter.allow("quiet") is True


def test_the_window_slides_so_a_blocked_client_recovers():
    limiter = RateLimiter(max_events=2, window_seconds=60)

    with patch("conversation.rate_limit.time.monotonic", return_value=1000.0):
        assert limiter.allow("client") is True
        assert limiter.allow("client") is True
        assert limiter.allow("client") is False

    # Past the window: the earlier hits have aged out.
    with patch("conversation.rate_limit.time.monotonic", return_value=1061.0):
        assert limiter.allow("client") is True


def test_only_hits_inside_the_window_count():
    limiter = RateLimiter(max_events=2, window_seconds=60)

    with patch("conversation.rate_limit.time.monotonic", return_value=1000.0):
        limiter.allow("client")
    with patch("conversation.rate_limit.time.monotonic", return_value=1030.0):
        limiter.allow("client")
    with patch("conversation.rate_limit.time.monotonic", return_value=1070.0):
        # The 1000.0 hit has expired, the 1030.0 one hasn't.
        assert limiter.allow("client") is True
        assert limiter.allow("client") is False


def test_pruning_drops_inactive_keys():
    """
    Buckets outlive their requests, so without pruning a public endpoint
    accumulates one dict entry per client forever.
    """
    limiter = RateLimiter(max_events=5, window_seconds=60)

    with patch("conversation.rate_limit.time.monotonic", return_value=1000.0):
        limiter.allow("old-client")
    with patch("conversation.rate_limit.time.monotonic", return_value=1100.0):
        limiter.allow("current-client")
        removed = limiter.prune()

    assert removed == 1
    assert "old-client" not in limiter._hits
    assert "current-client" in limiter._hits


def test_pruning_keeps_budgets_intact_for_active_keys():
    limiter = RateLimiter(max_events=2, window_seconds=60)
    limiter.allow("client")
    limiter.allow("client")

    limiter.prune()

    assert limiter.allow("client") is False

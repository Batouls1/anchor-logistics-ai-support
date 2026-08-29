"""
A small in-process sliding-window rate limiter.

Deliberately not Redis-backed: the limits here exist to stop one client
hammering the Gemini quota or the Whisper worker pool, and a per-process
limit is enough for that. Behind N processes each one allows the
configured rate, so treat the effective ceiling as N x the limit. If this
ever needs to be exact across replicas, the same interface can be backed
by Redis without touching the call sites.
"""

import time
from collections import defaultdict, deque


class RateLimiter:
    """
    Allows `max_events` per `window_seconds` for each key. Keys are
    whatever the caller wants to bucket by -- a conversation id, a client
    address, or both.
    """

    def __init__(self, max_events: int, window_seconds: float):
        self.max_events = max_events
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        """Records an attempt; returns False if the key is over its limit."""
        now = time.monotonic()
        hits = self._hits[key]

        cutoff = now - self.window_seconds
        while hits and hits[0] < cutoff:
            hits.popleft()

        if len(hits) >= self.max_events:
            return False

        hits.append(now)
        return True

    def prune(self) -> int:
        """
        Drops keys with no recent activity. Without this the dict grows
        once per client forever, which is a slow memory leak on a public
        endpoint. Called from the same sweep that expires idle sessions.
        """
        cutoff = time.monotonic() - self.window_seconds
        stale = [key for key, hits in self._hits.items() if not hits or hits[-1] < cutoff]
        for key in stale:
            del self._hits[key]
        return len(stale)

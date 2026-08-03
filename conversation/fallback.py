"""
Tracks failed voice-transcription attempts within a single conversation
and escalates the message after repeated failures.

One instance of this belongs to one TurnManager (one conversation) --
retry count is per-conversation, not global.
"""


class FallbackHandler:
    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries
        self.retry_count = 0

    def record_failure(self) -> str:
        """
        Call this exactly once per failed/low-confidence transcription.
        Increments the retry count and returns the message for this attempt.
        """
        self.retry_count += 1
        if self.retry_count > self.max_retries:
            return "I'm still having trouble understanding. Please type your question so I can help you best."
        return "I didn't catch that. Could you say that again?"

    def reset(self):
        """Call after any successful turn, so a later failure starts fresh."""
        self.retry_count = 0
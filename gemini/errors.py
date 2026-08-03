"""
Shared error types for the Gemini clients (live_client.py, text_client.py).
"""


class QuotaExceededError(RuntimeError):
    """
    Raised when a Gemini API call hits a 429 (rate limit / quota exceeded)
    response. Caught separately from generic errors in main.py so the user
    gets an honest "try again shortly" message instead of the generic
    "something went wrong" one.
    """
    pass
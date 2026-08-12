"""
Shared resilience helper: retries a function call with backoff when it
hits a RATE LIMIT or a TRANSIENT server error, instead of letting the
whole program crash on the first hiccup.

Why this matters for a real system: with real usage, external API calls
WILL occasionally fail -- rate limits, brief network issues, momentary
server overload on the provider's end. A production-grade assistant
treats this as an expected, recoverable situation, not a fatal bug. This
is true regardless of which LLM/embedding provider you use, or whether
one user is making requests or many at once.

What this deliberately does NOT do: retry on errors that retrying can't
fix (bad API key, invalid request, etc.) -- those get raised immediately
so you see the real problem instead of waiting through pointless retries.
"""

import time

# Substrings that show up across different providers' error messages when
# the failure is rate-limiting or a temporary server issue (i.e., worth
# retrying) rather than something wrong with the request itself.
RETRYABLE_MARKERS = [
    "429",
    "resource_exhausted",
    "rate limit",
    "rate_limit",
    "503",
    "500",
    "overloaded",
    "unavailable",
    "timeout",
]


def _is_retryable(exception: Exception) -> bool:
    message = str(exception).lower()
    return any(marker in message for marker in RETRYABLE_MARKERS)


def call_with_retry(func, *args, max_retries: int = 3, base_delay_seconds: int = 15, **kwargs):
    """
    Calls func(*args, **kwargs). If it fails with a retryable error
    (rate limit / transient server issue), waits and tries again with
    increasing delays (15s, 30s, 60s by default). If it fails with a
    non-retryable error (e.g. bad request, invalid API key), raises
    immediately -- no point waiting on something retrying won't fix.

    Raises a clear, final error if all retries are exhausted.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not _is_retryable(e):
                raise  # Not something a retry can fix -- surface it immediately

            last_exception = e
            if attempt < max_retries:
                wait_seconds = base_delay_seconds * (2**attempt)
                print(
                    f"  [INFO] Hit a rate limit or temporary issue. "
                    f"Retrying in {wait_seconds}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Still failing after {max_retries} retries due to rate limits or "
        f"a temporary provider issue. Last error: {last_exception}"
    ) from last_exception

"""
Shared resilience layer for LLM API calls.

Why this needs to be centralized instead of handled per-script: every
part of this project that calls an LLM (retrieval-answer generation,
question rewriting, tool-calling, the future orchestrator) can hit the
exact same kind of failure -- a rate limit. Real usage isn't uniform:
one user asks one question a day, another fires off twenty in a row
testing things. The CODE needs to handle both gracefully, without us
having to remember to wrap every single call site by hand.

What this does on a rate-limit error (HTTP 429):
  - If it looks like a short, transient limit, wait (using Google's own
    suggested retry delay when available) and retry a couple of times.
  - If retries are exhausted, stop trying and let the caller show a
    clear, human message instead of a raw stack trace.
"""

import re
import time

MAX_RETRIES = 2
DEFAULT_BACKOFF_SECONDS = 15.0

FRIENDLY_RATE_LIMIT_MESSAGE = (
    "I've hit the free-tier request limit for the AI service right now, so I can't respond "
    "at this exact moment. This isn't a bug -- free API keys have a daily/per-minute cap. "
    "It resets automatically; try again shortly. (If this happens often, see .env.example for "
    "how to switch to a different model or provider.)"
)


class RateLimitExceeded(Exception):
    """Raised when we've retried a rate-limited call and it still didn't succeed."""


def _is_rate_limit_error(error: Exception) -> bool:
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


def _extract_retry_delay(error: Exception) -> float | None:
    """Pulls Google's suggested wait time (e.g. "retryDelay": "29s") out of the error text."""
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", str(error))
    return float(match.group(1)) if match else None


def call_with_retry(fn, *args, **kwargs):
    """
    Calls fn(*args, **kwargs), automatically retrying on rate-limit errors.
    Raises RateLimitExceeded if retries are exhausted. Re-raises any
    OTHER kind of error immediately -- we only want to swallow/retry
    rate limits, not hide real bugs.
    """
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise  # a real bug -- don't mask it, fail loudly like normal

            last_error = e
            if attempt < MAX_RETRIES:
                wait = _extract_retry_delay(e) or DEFAULT_BACKOFF_SECONDS
                print(f"\n  [!] Hit a request rate limit. Waiting {wait:.0f}s before retrying "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})...\n")
                time.sleep(wait)

    raise RateLimitExceeded() from last_error

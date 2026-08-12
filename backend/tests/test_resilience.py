"""
Unit tests for the retry/resilience wrapper. Patches time.sleep so these
tests run in milliseconds instead of actually waiting 15+ seconds.

    pytest backend/tests/test_resilience.py -v
"""

import pytest

import backend.common.resilience as resilience_module
from backend.common.resilience import call_with_retry


def test_succeeds_immediately_when_no_error(monkeypatch):
    monkeypatch.setattr(resilience_module.time, "sleep", lambda s: None)

    result = call_with_retry(lambda: "ok")
    assert result == "ok"


def test_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr(resilience_module.time, "sleep", lambda s: None)

    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise Exception("429 RESOURCE_EXHAUSTED")
        return "success"

    result = call_with_retry(flaky, max_retries=3)

    assert result == "success"
    assert calls["count"] == 3


def test_non_retryable_error_raises_immediately(monkeypatch):
    monkeypatch.setattr(resilience_module.time, "sleep", lambda s: None)

    calls = {"count": 0}

    def bad_request():
        calls["count"] += 1
        raise Exception("400 invalid API key")

    with pytest.raises(Exception, match="invalid API key"):
        call_with_retry(bad_request, max_retries=3)

    assert calls["count"] == 1  # should NOT have retried


def test_raises_clear_error_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(resilience_module.time, "sleep", lambda s: None)

    def always_fails():
        raise Exception("503 service unavailable")

    with pytest.raises(RuntimeError, match="Still failing after"):
        call_with_retry(always_fails, max_retries=2)

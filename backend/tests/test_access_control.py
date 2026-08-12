"""
Unit tests for the role-based access control policy.

    pytest backend/tests/test_access_control.py -v
"""

from backend.common.access_control import get_allowed_access_levels


def test_employee_sees_only_public():
    assert get_allowed_access_levels("employee") == ["public"]


def test_hr_sees_public_and_hr():
    levels = get_allowed_access_levels("hr")
    assert "public" in levels
    assert "hr" in levels
    assert "engineering" not in levels


def test_admin_sees_everything():
    levels = get_allowed_access_levels("admin")
    assert set(levels) == {"public", "hr", "engineering"}


def test_unknown_role_fails_closed_to_employee_level():
    # Security-critical: an unrecognized/typo'd role must NOT accidentally
    # grant broad access. It should fall back to the most restrictive policy.
    levels = get_allowed_access_levels("some_made_up_role")
    assert levels == ["public"]


def test_empty_string_role_fails_closed():
    assert get_allowed_access_levels("") == ["public"]

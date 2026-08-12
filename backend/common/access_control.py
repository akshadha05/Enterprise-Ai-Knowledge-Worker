"""
Role-based access control policy: which document access levels each
user role is allowed to search.

This is intentionally a simple, hardcoded mapping for this stage of the
project -- a real enterprise system would look this up from a proper
user/identity database (or an SSO provider's claims), not a Python dict.
But the ENFORCEMENT mechanism this plugs into (VectorStore.query's
allowed_access_levels filter) is the same either way -- only the source
of "what role is this user" would change.
"""

DEFAULT_ROLE = "employee"

ROLE_ACCESS_LEVELS: dict[str, list[str]] = {
    "employee": ["public"],
    "hr": ["public", "hr"],
    "admin": ["public", "hr", "engineering"],
}


def get_allowed_access_levels(role: str) -> list[str]:
    """Returns the list of document access levels this role may search.
    Unknown roles fall back to the most restrictive (employee) policy --
    fail closed, not open."""
    return ROLE_ACCESS_LEVELS.get(role, ROLE_ACCESS_LEVELS[DEFAULT_ROLE])

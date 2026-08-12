"""
The "toolbox" -- real actions the assistant can take, not just answers
it can give.

These are MOCK implementations for now (they print what would happen and
log it to a local file, instead of actually sending a real email or
hitting a real calendar API). This is intentional and matches how you'd
build this for real: prove the LLM calls the right tool with the right
arguments FIRST, using safe mocks, before ever wiring up something that
sends a real email to a real person.

Swapping a mock for the real thing later means changing what happens
INSIDE these functions -- the LLM-facing interface (function name,
arguments, docstring) stays the same.

IMPORTANT: the docstrings below aren't just documentation -- the Gemini
SDK reads them to figure out what each tool does and what arguments it
needs. Vague docstrings = the model calling tools incorrectly.
"""

import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).parent.parent / "data" / "tool_actions_log.json"


def _log_action(action_type: str, details: dict) -> int:
    """Appends an action to the local log file and returns its ID number."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if LOG_FILE.exists():
        records = json.loads(LOG_FILE.read_text())

    record_id = len(records) + 1
    records.append(
        {
            "id": record_id,
            "type": action_type,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **details,
        }
    )
    LOG_FILE.write_text(json.dumps(records, indent=2))
    return record_id


def send_email(to: str, subject: str, body: str) -> str:
    """Sends an email to a colleague or external contact.

    Args:
        to: The recipient's email address.
        subject: The email subject line.
        body: The full email message content.
    """
    # Basic sanity check -- don't silently "send" to something that isn't
    # even shaped like an email address. Better to tell the LLM (and thus
    # the user) that something's off than to fake a successful send.
    if "@" not in to or "." not in to.split("@")[-1]:
        return (
            f"Could not send email: '{to}' doesn't look like a valid email address. "
            f"Ask the user to confirm the correct recipient."
        )

    record_id = _log_action("email", {"to": to, "subject": subject, "body": body})

    print(f"\n  [ACTION] Sending email (#{record_id})")
    print(f"    To: {to}")
    print(f"    Subject: {subject}")
    print(f"    Body: {body}\n")

    return f"Email #{record_id} sent successfully to {to}."


def create_ticket(title: str, description: str, priority: str = "Medium") -> str:
    """Creates a support or task ticket in the ticketing system.

    Args:
        title: A short title summarizing the issue or task.
        description: A detailed description of the issue or task.
        priority: One of "Low", "Medium", or "High". Defaults to "Medium".
    """
    record_id = _log_action(
        "ticket", {"title": title, "description": description, "priority": priority}
    )

    print(f"\n  [ACTION] Creating ticket (#{record_id})")
    print(f"    Title: {title}")
    print(f"    Priority: {priority}")
    print(f"    Description: {description}\n")

    return f"Ticket #{record_id} created: '{title}' (priority: {priority})."


def schedule_meeting(attendees: str, date: str, time: str, subject: str) -> str:
    """Schedules a meeting on the calendar.

    Args:
        attendees: Comma-separated names or email addresses of attendees.
        date: The meeting date, e.g. "2026-08-05".
        time: The meeting time, e.g. "3:00 PM".
        subject: A short subject/title for the meeting.
    """
    record_id = _log_action(
        "meeting", {"attendees": attendees, "date": date, "time": time, "subject": subject}
    )

    print(f"\n  [ACTION] Scheduling meeting (#{record_id})")
    print(f"    Subject: {subject}")
    print(f"    When: {date} at {time}")
    print(f"    Attendees: {attendees}\n")

    return f"Meeting #{record_id} '{subject}' scheduled for {date} at {time}."


def list_recent_actions(action_type: str = "all", limit: int = 5) -> str:
    """Looks up recently created tickets, sent emails, or scheduled meetings.

    Use this when the user asks things like "what tickets have been created?",
    "did we schedule that meeting?", or "show me recent emails sent."

    Args:
        action_type: One of "ticket", "email", "meeting", or "all" (default).
        limit: Maximum number of recent records to return. Defaults to 5.
    """
    if not LOG_FILE.exists():
        return "No actions have been logged yet."

    records = json.loads(LOG_FILE.read_text())

    if action_type != "all":
        records = [r for r in records if r["type"] == action_type]

    if not records:
        return f"No records of type '{action_type}' found."

    recent = records[-limit:]
    lines = []
    for r in recent:
        summary = {k: v for k, v in r.items() if k not in ("type",)}
        lines.append(f"#{r['id']} ({r['type']}): {summary}")

    return "\n".join(lines)


# The list the LLM will actually be given access to.
AVAILABLE_TOOLS = [send_email, create_ticket, schedule_meeting, list_recent_actions]

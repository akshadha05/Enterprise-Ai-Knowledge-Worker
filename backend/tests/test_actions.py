"""
Unit tests for the action tools (send_email, create_ticket, etc).
Redirects the log file to a temp path so these tests never touch your
real backend/data/tool_actions_log.json.

    pytest backend/tests/test_actions.py -v
"""

import json

import backend.tools.actions as actions_module


def test_send_email_rejects_invalid_address(tmp_path, monkeypatch):
    monkeypatch.setattr(actions_module, "LOG_FILE", tmp_path / "log.json")

    result = actions_module.send_email(to="not-an-email", subject="Hi", body="Test")

    assert "Could not send" in result
    assert not (tmp_path / "log.json").exists()  # nothing should have been logged


def test_send_email_accepts_valid_address(tmp_path, monkeypatch):
    log_file = tmp_path / "log.json"
    monkeypatch.setattr(actions_module, "LOG_FILE", log_file)

    result = actions_module.send_email(to="hr@company.com", subject="Hi", body="Test")

    assert "sent successfully" in result
    assert log_file.exists()
    records = json.loads(log_file.read_text())
    assert records[0]["to"] == "hr@company.com"


def test_create_ticket_logs_correctly(tmp_path, monkeypatch):
    log_file = tmp_path / "log.json"
    monkeypatch.setattr(actions_module, "LOG_FILE", log_file)

    result = actions_module.create_ticket(title="Bug", description="It's broken", priority="High")

    assert "Ticket #1" in result
    assert "High" in result
    records = json.loads(log_file.read_text())
    assert records[0]["priority"] == "High"


def test_list_recent_actions_filters_by_type(tmp_path, monkeypatch):
    log_file = tmp_path / "log.json"
    monkeypatch.setattr(actions_module, "LOG_FILE", log_file)

    actions_module.create_ticket(title="Ticket A", description="...", priority="Low")
    actions_module.send_email(to="a@b.com", subject="Subj", body="Body")
    actions_module.create_ticket(title="Ticket B", description="...", priority="High")

    tickets_only = actions_module.list_recent_actions(action_type="ticket")

    assert "Ticket A" in tickets_only
    assert "Ticket B" in tickets_only
    assert "Subj" not in tickets_only  # the email shouldn't leak into ticket results


def test_list_recent_actions_respects_limit(tmp_path, monkeypatch):
    log_file = tmp_path / "log.json"
    monkeypatch.setattr(actions_module, "LOG_FILE", log_file)

    for i in range(5):
        actions_module.create_ticket(title=f"Ticket {i}", description="...", priority="Low")

    result = actions_module.list_recent_actions(action_type="ticket", limit=2)
    assert result.count("#") == 2  # only 2 records returned


def test_list_recent_actions_with_no_log_file(tmp_path, monkeypatch):
    monkeypatch.setattr(actions_module, "LOG_FILE", tmp_path / "does_not_exist.json")

    result = actions_module.list_recent_actions()

    assert "No actions have been logged yet" in result

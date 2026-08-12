"""
Short-term (in-session) conversational memory.

Keeps a running list of question/answer pairs so far in the current
session. This is the "remembers previous conversations" piece -- it's
what lets a follow-up question like "what was the F1 score?" be
understood in the context of "we were just talking about ChurnWise."

This is intentionally simple: an in-memory Python list. It disappears
when the program exits. Persisting this across sessions (so it remembers
you tomorrow too) is a further upgrade -- see the note at the bottom.
"""

from dataclasses import dataclass, field


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class ConversationMemory:
    turns: list[Turn] = field(default_factory=list)

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def format_recent(self, max_turns: int = 4) -> str:
        """
        Returns the last few turns as plain text, suitable for dropping
        into a prompt. We only keep a handful of recent turns (not the
        whole history) to keep prompts short and cheap -- for a real
        enterprise deployment with long sessions, older turns would
        eventually get summarized instead of dropped entirely.
        """
        recent = self.turns[-max_turns:]
        lines = []
        for t in recent:
            lines.append(f"User: {t.question}")
            lines.append(f"Assistant: {t.answer}")
        return "\n".join(lines)


# NOTE on going further (not built yet, for when you're ready):
# To remember a user across SEPARATE runs of the program (not just within
# one session), you'd persist `turns` to a small database (SQLite is
# perfect for this) keyed by a user/session ID, and load it back in at
# startup instead of starting with an empty ConversationMemory().

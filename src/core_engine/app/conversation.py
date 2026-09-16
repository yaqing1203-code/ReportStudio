"""Conversation history + context-window management for chat mode.

The desktop app is single-user, so one in-memory Conversation holds the running
chat. Each turn is appended; before every LLM call the full history is rendered
into a transcript and passed in the request payload so the assistant remembers
prior turns.

Context guard: when the accumulated history exceeds `summarize_word_limit` words
(default 100k), the OLDER turns are compressed into a single summary paragraph via
the LLM, and only that summary + the most recent turns are retained. This keeps us
comfortably under model context limits while preserving the gist of the discussion.

Nothing here is persisted to disk — history lives for the process lifetime, which
matches a desktop chat session. Clearing is a restart (or the New Report button).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Turn:
    role: str          # "user" | "assistant"
    content: str

    def words(self) -> int:
        return len(self.content.split())


@dataclass
class Conversation:
    """In-memory chat history with automatic summarization of older turns."""

    turns: list[Turn] = field(default_factory=list)
    # A rolling summary of turns that were compressed away. Prepended to the
    # transcript so their gist survives even after they're dropped.
    summary: str = ""

    def add_user(self, content: str) -> None:
        self.turns.append(Turn("user", content))

    def add_assistant(self, content: str) -> None:
        self.turns.append(Turn("assistant", content))

    def word_count(self) -> int:
        total = len(self.summary.split())
        return total + sum(t.words() for t in self.turns)

    def transcript(self) -> str:
        """Render summary + turns into a plain-text transcript for the LLM payload."""
        parts: list[str] = []
        if self.summary:
            parts.append(f"[Summary of earlier conversation]\n{self.summary}")
        for t in self.turns:
            label = "User" if t.role == "user" else "Assistant"
            parts.append(f"{label}: {t.content}")
        return "\n\n".join(parts)

    def clear(self) -> None:
        self.turns.clear()
        self.summary = ""


SUMMARIZER_SYSTEM_PROMPT = (
    "You compress an older segment of a chat transcript into a single concise "
    "paragraph. Preserve concrete facts, decisions, names, numbers, and the user's "
    "goals. Drop pleasantries and redundancy. Output ONLY the summary paragraph, no "
    "preamble."
)


async def maybe_summarize(convo: Conversation, llm, *, word_limit: int,
                          keep_recent: int = 6) -> bool:
    """If the conversation exceeds `word_limit` words, summarize everything except
    the `keep_recent` most-recent turns into `convo.summary` and drop those turns.

    Returns True if summarization ran. `llm` must expose async complete(system, user).
    Best-effort: if the LLM call fails we keep the history intact (a slightly long
    context is better than losing it), so this never breaks a chat turn.
    """
    if convo.word_count() <= word_limit:
        return False
    if len(convo.turns) <= keep_recent:
        return False  # nothing old enough to compress

    old = convo.turns[:-keep_recent]
    recent = convo.turns[-keep_recent:]

    old_text = "\n\n".join(
        f"{'User' if t.role == 'user' else 'Assistant'}: {t.content}" for t in old
    )
    prior = f"[Existing summary]\n{convo.summary}\n\n" if convo.summary else ""
    try:
        new_summary = await llm.complete(
            system=SUMMARIZER_SYSTEM_PROMPT,
            user=f"{prior}[Transcript to compress]\n{old_text}",
        )
    except Exception:
        return False  # keep history rather than lose it

    convo.summary = new_summary.strip()
    convo.turns = recent
    return True

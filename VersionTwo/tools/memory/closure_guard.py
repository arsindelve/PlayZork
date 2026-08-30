"""Refusing a closure the game's own transcript contradicts.

The IssueClosedAgent closed the ESCAPE POD — the objective — in five of six
Planetfall runs, always shortly after the turn-2 refusal "Why open the door to
the emergency escape pod if there's no emergency?". It reads a TEMPORAL refusal
("not yet") as resolution ("done"). pf4 escaped with every one of its four
issues closed, so the memory system contributed nothing to either success.

A FIRST ATTEMPT AT THIS GUARD WAS INSUFFICIENT, and the reason is worth
recording. It deferred closures on any turn that changed nothing — no move, no
score, no inventory change — which correctly caught the turn-2 refusal. But the
closer simply re-stages the same closure next turn, and the very next turn was
a move, so the closure went through anyway and the pod was lost on turn 3
instead of turn 2. A TURN-level guard cannot fix an ISSUE-level misjudgement;
deferral only helps if the closer changes its mind, and it does not.

So the question this asks is about the issue, not the turn: has the action this
issue asks for already been TRIED in the room it applies to, and accomplished
nothing? If so, the transcript positively contradicts the claim that it is
resolved, and the closure is refused rather than delayed.

This is deliberately evidence-based rather than a prohibition. It cannot block
a closure the game has not already disproved, so an issue that really was
resolved still closes normally.
"""
from typing import Any, Iterable, Optional, Sequence

_SEPARATORS = ("—", " - ", "--")

# Words that carry no identifying weight when matching a criteria to a command.
_NOISE = frozenset({
    "the", "a", "an", "and", "or", "it", "its", "to", "of", "in", "on", "at",
    "for", "with", "from", "into", "onto", "then", "if", "is", "are", "be",
    "this", "that", "them", "there", "here", "find", "way", "try", "get",
    "some", "any", "all", "out", "up", "down", "back", "again", "reason",
})


def _criteria_of(content: str) -> str:
    """The acceptance criteria — the half after the em-dash."""
    for separator in _SEPARATORS:
        if separator in content:
            return content.partition(separator)[2]
    return content


def _significant(text: Optional[str]) -> set:
    return {w for w in (text or "").lower().replace(",", " ").split()
            if w not in _NOISE and len(w) > 2}


def _same_room(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a and b and a.strip().casefold() == b.strip().casefold())


def closure_is_contradicted(
    content: Optional[str],
    target_location: Optional[str],
    turns: Optional[Sequence[Any]] = None,
    overlap_required: int = 2,
) -> Optional[str]:
    """The response that disproves this closure, or None.

    A closure is contradicted when a recent turn, IN THE ISSUE'S TARGET ROOM,
    ran a command that clearly attempts the issue's acceptance criteria and
    neither moved us nor scored. "OPEN escape pod bulkhead" against the
    criteria "open bulkhead and examine escape pod" overlaps on four
    significant words; a move like "GO WEST" overlaps on none.

    `overlap_required` is 2 rather than 1 so a single shared noun cannot
    condemn an unrelated command: "drop lamp" must not count as an attempt at
    "take the lamp and light it".
    """
    if not content or not turns:
        return None

    wanted = _significant(_criteria_of(content))
    if len(wanted) < overlap_required:
        return None

    previous = None
    for turn in turns:
        command_words = _significant(getattr(turn, "player_command", ""))
        if len(wanted & command_words) < overlap_required:
            previous = turn
            continue
        if target_location and not _same_room(
                getattr(turn, "location", None), target_location):
            previous = turn
            continue
        # Did it accomplish anything? Same test the rest of the world model
        # uses. Without a predecessor we cannot tell, so we do not claim a
        # contradiction — never block a closure on missing information.
        if previous is None:
            previous = turn
            continue
        moved = getattr(turn, "location", None) != getattr(previous, "location", None)
        scored = (getattr(turn, "score", 0) or 0) > (getattr(previous, "score", 0) or 0)
        if not moved and not scored:
            return (getattr(turn, "game_response", "") or "").strip()
        previous = turn
    return None

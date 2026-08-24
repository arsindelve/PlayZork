"""Where an issue actually points, as opposed to where it was noticed.

A Memory's `location` field is documented as "Where we were when we learned
this" — the SIGHTING room. That is the useful room surprisingly often: "White
house at West Of House" is seen from the very place you must return to, and the
grating at the Clearing is both seen and acted on there.

It is wrong exactly when an issue is ABOUT somewhere else. Planetfall run
pf5-20260824 stored:

    "Ensign Blather at Reactor Lobby — return to Deck Nine as ordered"
                     ^ sighting room                  ^ the actual target

`location` was Reactor Lobby, so pathfinding routed the agent to where it
already stood, returned NO PATH, and the IssueAgent had nothing to offer. Both
escape runs fell back to the model inventing "RETURN TO DECK NINE" as a raw
command — which Planetfall's parser happened to accept and Zork's would not.

The target is recovered from the issue text using the MAP as the vocabulary:
only room names we have actually seen can be matched, so this cannot invent a
destination, and anything it fails to match simply falls back to today's
behaviour. Deterministic, no LLM call.
"""
import re
from typing import Iterable, Optional

# Issues are written as "<subject> at <Location> — <acceptance criteria>".
_CRITERIA_SEPARATORS = ("—", " - ", "--")


def _split_criteria(content: str) -> tuple:
    """(subject, criteria). Criteria is "" when the issue has no separator."""
    for separator in _CRITERIA_SEPARATORS:
        if separator in content:
            subject, _, criteria = content.partition(separator)
            return subject, criteria
    return content, ""


def _find_location(text: str, known_locations: Iterable[str]) -> str:
    """The longest known room name mentioned in `text`, or ""."""
    if not text:
        return ""
    # Longest first so "North of House" wins over "House".
    for name in sorted((n for n in known_locations if n), key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
            return name
    return ""


def resolve_issue_target(
    content: Optional[str],
    sighting_location: Optional[str],
    known_locations: Optional[Iterable[str]] = None,
) -> str:
    """The room this issue should route the agent toward.

    The ACCEPTANCE CRITERIA wins over the subject, because that is the half
    describing what must be done: in "Ensign Blather at Reactor Lobby — return
    to Deck Nine", the criteria names Deck Nine while the subject names the
    room we were standing in.

    Falls back to the sighting location whenever nothing is matched, so an
    issue whose text names no known room behaves exactly as it does today.
    """
    sighting = (sighting_location or "").strip()
    if not content or not known_locations:
        return sighting

    known = list(known_locations)
    subject, criteria = _split_criteria(content)

    from_criteria = _find_location(criteria, known)
    if from_criteria:
        return from_criteria

    from_subject = _find_location(subject, known)
    if from_subject:
        return from_subject

    return sighting

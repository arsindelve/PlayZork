"""Location-name vocabulary (GitHub issue #7).

The game does not always report a room name — some cutscenes and transitional
states return an empty LocationName. (Death is NOT one of them: probing the
live backend shows a death turn reports the *respawn* room, e.g. "Forest".
That case is handled by the death gate in `response_signals`, not here.)

Several call sites papered over that with `LocationName or "Unknown"`, which
turned a *missing* value into a *fabricated room*. Downstream code then treated "Unknown" as a real place:
the explorer claimed all ten directions were unexplored from it, IssueAgents
asked the pathfinder to route out of it (always NO PATH), and memories were
stored anchored to it forever.

The rule this module encodes: "Unknown" is acceptable as PROSE shown to the
LLM, never as DATA. Anything that indexes the map, stores a location, or
computes a route must ask `is_known_location()` first.

Deliberately dependency-free.
"""
from typing import Optional

# Human-readable stand-in for "the game did not tell us where we are". Safe to
# interpolate into a prompt; never safe to store or look up.
UNKNOWN_LOCATION = "Unknown"


def normalize_location(location: Optional[str]) -> str:
    """Canonical comparison key for a room name (GitHub issue #13).

    Casing is not consistent even within one backend: the live Zork API returns
    "West Of House" but "North of House" and "South of House". A model that has
    seen the first writes the siblings by analogy, and Zork's own printed name
    ("West of House") disagrees with the API as well, so the model's prior
    knowledge of the game is wrong too. Location lookups therefore compare on
    this key, while storage and display keep the backend's own spelling.

    Internal whitespace is collapsed because models emit "West  Of House".
    """
    return " ".join((location or "").split()).casefold()


def is_known_location(location: Optional[str]) -> bool:
    """True when `location` names a real room we can map, store or route from."""
    if not location:
        return False
    return location.strip().casefold() != UNKNOWN_LOCATION.casefold()

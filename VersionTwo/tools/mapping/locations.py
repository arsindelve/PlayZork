"""Location-name vocabulary (GitHub issue #7).

The game does not always report a room name — darkness, some cutscenes, and
death sequences all return an empty LocationName. Several call sites papered
over that with `LocationName or "Unknown"`, which turned a *missing* value into
a *fabricated room*. Downstream code then treated "Unknown" as a real place:
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


def is_known_location(location: Optional[str]) -> bool:
    """True when `location` names a real room we can map, store or route from."""
    if not location:
        return False
    return location.strip().casefold() != UNKNOWN_LOCATION.casefold()

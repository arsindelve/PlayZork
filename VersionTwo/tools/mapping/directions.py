"""Canonical direction vocabulary shared by the mapper and the explorer.

The game and the LLM both use abbreviations ("N") and full names ("NORTH")
interchangeably, and several prompts actively push the model toward the
shortest form. The mapper stored whatever arrived, while the explored-check
compared against full names only — so a move recorded as "N" left NORTH
looking unexplored forever and the ExplorerAgent re-proposed it every turn
(GitHub issue #9). `map_transitions` is UNIQUE(session_id, from_location,
direction), so "N" and "NORTH" are two distinct rows: the constraint cannot
dedupe them.

Deliberately dependency-free so `decision_graph` can import it without
pulling in DatabaseManager.
"""
from typing import Optional

CANONICAL_DIRECTIONS = [
    "NORTH", "SOUTH", "EAST", "WEST",
    "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST",
    "UP", "DOWN",
]

DIRECTION_ABBREVIATIONS = {
    "N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST",
    "NE": "NORTHEAST", "NW": "NORTHWEST", "SE": "SOUTHEAST", "SW": "SOUTHWEST",
    "U": "UP", "D": "DOWN",
}


def normalize_direction(direction: Optional[str]) -> str:
    """Map any direction token to its canonical full name.

    Unknown tokens pass through uppercased and stripped, so this is safe to
    apply to arbitrary stored strings (including "BLOCKED" destinations and
    non-cardinal edges).
    """
    token = (direction or "").strip().upper()
    return DIRECTION_ABBREVIATIONS.get(token, token)

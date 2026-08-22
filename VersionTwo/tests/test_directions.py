"""Canonical direction handling (GitHub issue #9).

A move issued as "N" was stored as "N" while the explored-check compared
against "NORTH", so the direction looked unexplored forever and the
ExplorerAgent re-proposed it every turn. `map_transitions` is
UNIQUE(session_id, from_location, direction), so "N" and "NORTH" are two
distinct rows and the constraint cannot dedupe them.
"""
from typing import List, Tuple

import pytest

from tools.mapping.directions import (
    CANONICAL_DIRECTIONS,
    DIRECTION_ABBREVIATIONS,
    normalize_direction,
)
from tools.mapping.mapper_state import MapperState
from tools.mapping.pathfinder import PathFinder


# ---------------------------------------------------------------------------
# normalize_direction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("abbrev,full", sorted(DIRECTION_ABBREVIATIONS.items()))
def test_every_abbreviation_maps_to_its_full_name(abbrev, full):
    assert normalize_direction(abbrev) == full
    assert full in CANONICAL_DIRECTIONS


@pytest.mark.parametrize("direction", CANONICAL_DIRECTIONS)
def test_canonical_directions_are_idempotent(direction):
    assert normalize_direction(direction) == direction


def test_case_and_whitespace_are_handled():
    assert normalize_direction("  ne ") == "NORTHEAST"
    assert normalize_direction("north") == "NORTH"


def test_unknown_tokens_pass_through_uppercased():
    """Must be safe on arbitrary stored strings, including non-cardinal edges."""
    assert normalize_direction("MAILBOX") == "MAILBOX"
    assert normalize_direction("CLIMB TREE") == "CLIMB TREE"
    assert normalize_direction(None) == ""
    assert normalize_direction("") == ""


# ---------------------------------------------------------------------------
# MapperState write + read boundaries
# ---------------------------------------------------------------------------


class MockDatabase:
    """Same shape as test_pathfinder.MockDatabase, with no UNIQUE constraint."""

    def __init__(self):
        self.transitions: List[Tuple[str, str, str, int]] = []

    def add_map_transition(self, session_id, from_location, to_location, direction, turn_number):
        self.transitions.append((from_location, to_location, direction, turn_number))
        return True

    def get_all_transitions(self, session_id):
        return self.transitions

    def get_transitions_from_location(self, session_id, location):
        return [
            (direction, to_loc)
            for from_loc, to_loc, direction, turn in self.transitions
            if from_loc == location
        ]


@pytest.fixture
def mock_db():
    return MockDatabase()


@pytest.fixture
def mapper_state(mock_db):
    return MapperState(session_id="test_session", db=mock_db)


def test_abbreviated_command_records_canonical_direction(mapper_state):
    mapper_state.previous_location = "West Of House"
    mapper_state.update_from_turn("North Of House", "N", 1)

    assert mapper_state.get_exits_from("West Of House") == [("NORTH", "North Of House")]


def test_abbreviated_and_full_command_are_one_edge(mapper_state):
    mapper_state.previous_location = "West Of House"
    mapper_state.update_from_turn("North Of House", "N", 1)
    mapper_state.previous_location = "West Of House"
    mapper_state.update_from_turn("North Of House", "GO NORTH", 3)

    exits = mapper_state.get_exits_from("West Of House")
    assert exits == [("NORTH", "North Of House")], "abbreviation created a duplicate edge"


def test_explored_check_sees_an_abbreviated_move(mapper_state):
    """The exact loop from #9: after moving N, NORTH must not look unexplored."""
    mapper_state.previous_location = "West Of House"
    mapper_state.update_from_turn("North Of House", "N", 1)

    known = {direction for direction, _ in mapper_state.get_exits_from("West Of House")}
    unexplored = [d for d in CANONICAL_DIRECTIONS if d not in known]

    assert "NORTH" not in unexplored


# ---------------------------------------------------------------------------
# Legacy rows written before the fix (resumed sessions self-heal on read)
# ---------------------------------------------------------------------------


def test_legacy_abbreviated_rows_are_normalized_on_read(mapper_state, mock_db):
    mock_db.transitions.append(("A", "B", "N", 1))

    assert mapper_state.get_exits_from("A") == [("NORTH", "B")]
    assert mapper_state.get_all_transitions()[0].direction == "NORTH"


def test_legacy_duplicate_rows_prefer_the_real_passage(mapper_state, mock_db):
    """A stale BLOCKED row must not hide a known passage after collapsing."""
    mock_db.transitions.append(("A", "BLOCKED", "N", 2))
    mock_db.transitions.append(("A", "B", "NORTH", 5))

    assert mapper_state.get_exits_from("A") == [("NORTH", "B")]
    assert PathFinder(mapper_state).find_path("A", "B") == ["NORTH"]


def test_blocked_rows_still_count_as_explored(mapper_state, mock_db):
    """Collapsing must not silently drop BLOCKED edges — #11 owns that decision."""
    mock_db.transitions.append(("A", "BLOCKED", "E", 2))

    assert mapper_state.get_exits_from("A") == [("EAST", "BLOCKED")]


def test_pathfinder_emits_canonical_directions_for_legacy_rows(mapper_state, mock_db):
    mock_db.transitions.append(("A", "B", "N", 1))
    mock_db.transitions.append(("B", "C", "E", 2))

    assert PathFinder(mapper_state).find_path("A", "C") == ["NORTH", "EAST"]

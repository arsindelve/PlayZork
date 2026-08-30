"""Non-cardinal movement is mapped as a raw-command edge (GitHub issue #14).

Zork moves the player with plain commands as often as with compass directions:
CLIMB TREE, ENTER HOUSE, IN, OUT, CROSS BRIDGE, TOUCH MIRROR, PRAY. The mapper
only understood cardinals, so those destinations became orphan nodes — rooms
the map knows exist but can never route *to*.
"""
from typing import List, Tuple

import pytest

from tools.mapping.directions import (
    CANONICAL_DIRECTIONS,
    is_probable_movement_command,
    normalize_movement_command,
)
from tools.mapping.mapper_state import MapperState
from tools.mapping.pathfinder import PathFinder


class MockDatabase:
    """Mirrors the real UNIQUE(session_id, from_location, direction)."""

    def __init__(self):
        self.transitions: List[Tuple[str, str, str, int]] = []

    def add_map_transition(self, session_id, from_location, to_location, direction, turn_number):
        for index, (from_loc, _, existing_dir, _) in enumerate(self.transitions):
            if (from_loc, existing_dir) == (from_location, direction):
                self.transitions[index] = (from_location, to_location, direction, turn_number)
                return True
        self.transitions.append((from_location, to_location, direction, turn_number))
        return True

    def get_all_transitions(self, session_id):
        return self.transitions

    def get_transitions_from_location(self, session_id, location):
        return [
            (direction, to_loc)
            for from_loc, to_loc, direction, _ in self.transitions
            if from_loc == location
        ]


@pytest.fixture
def mock_db():
    return MockDatabase()


@pytest.fixture
def mapper_state(mock_db):
    return MapperState(session_id="test_session", db=mock_db)


def _move(state, frm, to, command, turn=1, response="You move."):
    state.previous_location = frm
    state.update_from_turn(to, command, turn, game_response=response)


# ---------------------------------------------------------------------------
# Label canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command,expected", [
    ("climb tree", "CLIMB TREE"),
    ("  CROSS   BRIDGE  ", "CROSS BRIDGE"),
    ("GO IN", "IN"),
    ("WALK ACROSS BRIDGE", "ACROSS BRIDGE"),
    ("GO", "GO"),
    ("", ""),
    (None, ""),
])
def test_movement_labels_are_canonicalized(command, expected):
    assert normalize_movement_command(command) == expected


def test_go_in_and_in_describe_one_passage():
    """Otherwise UNIQUE(from, direction) stores the same passage twice."""
    assert normalize_movement_command("GO IN") == normalize_movement_command("IN")


# ---------------------------------------------------------------------------
# Recognition rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", [
    "CLIMB TREE", "CLIMB UP", "ENTER HOUSE", "ENTER", "IN", "OUT", "EXIT",
    "CROSS BRIDGE", "GET IN BOAT", "GET OUT OF BOAT", "BOARD BOAT", "LAUNCH",
    "DISEMBARK", "GO IN",
    # Game-specific teleports no verb allow-list would ever contain:
    "TOUCH MIRROR", "PRAY", "ODYSSEUS",
])
def test_movement_shaped_commands_are_mapped(command):
    assert is_probable_movement_command(command) is True


@pytest.mark.parametrize("command", [
    "TAKE LAMP", "GET LAMP", "MOVE RUG", "EXAMINE MAILBOX", "READ LEAFLET",
    "OPEN WINDOW", "DROP SWORD", "ATTACK TROLL WITH SWORD",
    "INVENTORY", "WAIT", "SAVE", "SCORE", "DIAGNOSE", "", None,
    "PUT THE SMALL BRASS LANTERN INSIDE THE TROPHY CASE NOW",
])
def test_non_movement_commands_are_rejected(command):
    assert is_probable_movement_command(command) is False


# ---------------------------------------------------------------------------
# Mapper integration
# ---------------------------------------------------------------------------


def test_climb_tree_records_a_raw_command_edge(mapper_state):
    _move(mapper_state, "Forest Path", "Up A Tree", "CLIMB TREE")

    assert mapper_state.get_exits_from("Forest Path") == [("CLIMB TREE", "Up A Tree")]


@pytest.mark.parametrize("command,label", [
    ("ENTER HOUSE", "ENTER HOUSE"),
    ("IN", "IN"),
    ("OUT", "OUT"),
    ("CROSS BRIDGE", "CROSS BRIDGE"),
    ("GET IN BOAT", "GET IN BOAT"),
    ("GO IN", "IN"),
])
def test_non_cardinal_commands_record_their_raw_label(mapper_state, command, label):
    _move(mapper_state, "A", "B", command)

    assert mapper_state.get_exits_from("A") == [(label, "B")]


def test_the_reported_symptom_up_a_tree_is_routable(mapper_state):
    """The issue's exact repro: CLIMB TREE up, DOWN back."""
    _move(mapper_state, "Forest Path", "Up A Tree", "CLIMB TREE", turn=1)
    _move(mapper_state, "Up A Tree", "Forest Path", "DOWN", turn=2)

    pathfinder = PathFinder(mapper_state)
    assert pathfinder.find_path("Forest Path", "Up A Tree") == ["CLIMB TREE"]
    assert pathfinder.find_path("Up A Tree", "Forest Path") == ["DOWN"]


def test_multi_hop_route_mixes_cardinals_and_raw_commands(mapper_state):
    _move(mapper_state, "West Of House", "North Of House", "NORTH", turn=1)
    _move(mapper_state, "North Of House", "Forest Path", "EAST", turn=2)
    _move(mapper_state, "Forest Path", "Up A Tree", "CLIMB TREE", turn=3)

    assert PathFinder(mapper_state).find_path("West Of House", "Up A Tree") == [
        "NORTH", "EAST", "CLIMB TREE",
    ]


def test_the_pathfinder_hands_back_an_executable_command(mapper_state):
    """A raw label needs no translation by the agent; "SOUTH" does."""
    _move(mapper_state, "Forest Path", "Up A Tree", "CLIMB TREE")

    assert PathFinder(mapper_state).get_next_step("Forest Path", "Up A Tree") == "CLIMB TREE"


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["EXAMINE MAILBOX", "TAKE LEAFLET", "INVENTORY", "MOVE RUG"])
def test_raw_commands_never_record_blocked(mapper_state, mock_db, command):
    """The BLOCKED branch fires on "room unchanged", which is true of every
    non-movement command — one junk row per turn, and get_map feeds every row
    to the LLM."""
    mapper_state.previous_location = "Living Room"
    mapper_state.update_from_turn("Living Room", command, 1, game_response="Taken.")

    assert mock_db.transitions == []


def test_a_failed_non_cardinal_move_records_nothing(mapper_state, mock_db):
    mapper_state.previous_location = "Forest Path"
    mapper_state.update_from_turn(
        "Forest Path", "CLIMB TREE", 1, game_response="You can't go that way."
    )

    assert mock_db.transitions == []


def test_a_death_on_a_movement_shaped_command_records_nothing(mapper_state, mock_db):
    """#14 widens #12's blast radius; the death gate must still win."""
    mapper_state.previous_location = "Troll Room"
    mapper_state.update_from_turn(
        "Forest", "ATTACK TROLL WITH SWORD", 1,
        game_response="The troll's axe removes your head.\n\n*** You have died ***",
    )

    assert mock_db.transitions == []


def test_a_raw_edge_does_not_mark_any_cardinal_explored(mapper_state):
    _move(mapper_state, "Forest Path", "Up A Tree", "CLIMB TREE")

    known = {direction for direction, _ in mapper_state.get_exits_from("Forest Path")}
    unexplored = [d for d in CANONICAL_DIRECTIONS if d not in known]

    assert len(unexplored) == 10


def test_repeating_a_raw_move_stays_one_row(mapper_state, mock_db):
    _move(mapper_state, "Forest Path", "Up A Tree", "CLIMB TREE", turn=1)
    _move(mapper_state, "Forest Path", "Up A Tree", "climb tree", turn=5)

    assert len(mock_db.transitions) == 1

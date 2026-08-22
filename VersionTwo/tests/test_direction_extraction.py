"""Tokenized direction extraction and refusal-gated BLOCKED (issues #10, #15).

The old extractor substring-matched single letters under a
`startswith("MOVE ")` guard, so the "E" inside the verb MOVE itself matched:
EVERY `MOVE <noun>` reported an EAST-ish move. Object manipulation leaves the
room unchanged, so the mapper then wrote `<room> --[E]--> BLOCKED` -- deleting
a real exit that had never been tried.

Failure phrases below are quoted from live probes of the Zork I and Planetfall
backends on 2026-08-22, not from the published Infocom text.
"""
from typing import List, Tuple

import pytest

from tools.mapping.directions import (
    MOVEMENT_VERBS,
    extract_direction,
    is_direction,
)
from tools.mapping.mapper_state import MapperState
from tools.mapping.response_signals import is_movement_refusal


class MockDatabase:
    """Same shape as test_directions.MockDatabase, with no UNIQUE constraint."""

    def __init__(self):
        self.transitions: List[Tuple[str, str, str, int]] = []

    def add_map_transition(self, session_id, from_location, to_location, direction, turn_number):
        self.transitions.append((from_location, to_location, direction, turn_number))
        return True

    def get_all_transitions(self, session_id):
        return self.transitions

    def get_transitions_from_location(self, session_id, location):
        return [(d, to) for f, to, d, _ in self.transitions if f == location]


@pytest.fixture
def mock_db():
    return MockDatabase()


@pytest.fixture
def mapper(mock_db):
    return MapperState(session_id="test_session", db=mock_db)


# ---------------------------------------------------------------------------
# extract_direction -- the #10 regression table
# ---------------------------------------------------------------------------

EXTRACTION_TABLE = [
    # (command, expected)
    ("MOVE RUG", None),          # the bug: "E" from the verb MOVE
    ("MOVE ROCK", None),
    ("MOVE LEAVES", None),
    ("MOVE SOFA", None),
    ("MOVE MIRROR", None),
    ("MOVE TROPHY CASE", None),
    ("OPEN DOOR", None),
    ("EXAMINE SWORD", None),
    ("TAKE LEAFLET", None),
    ("READ NEWSPAPER", None),
    ("PUSH BUTTON", None),
    ("DROP SWORD", None),
    ("LOOK", None),
    ("WEST OF HOUSE", None),     # a room name, not a command
    ("GO IN", None),             # IN is #14's vocabulary, not #10's
    ("GO INSIDE", None),
    ("GO TO THE KITCHEN", None),
    ("GO THROUGH WINDOW", None),
    ("WALK AROUND HOUSE", None), # the "SE" inside HOUSE
    ("GO BACK", None),
    ("ENTER HOUSE", None),       # #14 owns ENTER
    ("CLIMB TREE", None),        # #14 owns CLIMB
    # genuine movement
    ("NORTH", "NORTH"),
    ("N", "NORTH"),
    ("SOUTHWEST", "SOUTHWEST"),
    ("SW", "SOUTHWEST"),
    ("DOWN", "DOWN"),
    ("U", "UP"),
    ("GO NORTH", "NORTH"),
    ("WALK EAST", "EAST"),
    ("HEAD SOUTH", "SOUTH"),
    ("RUN WEST", "WEST"),
    ("GO SW", "SOUTHWEST"),
]


@pytest.mark.parametrize("command,expected", EXTRACTION_TABLE)
def test_extraction_table(command, expected):
    assert extract_direction(command) == expected


@pytest.mark.parametrize("noun", [
    "RUG", "ROCK", "LEAVES", "SOFA", "MIRROR", "PAINTING", "CASE", "SAND",
    "DOOR", "COFFIN", "TROPHY CASE", "BOAT", "TABLE", "GRATING", "PILE OF LEAVES",
])
def test_move_noun_is_never_a_direction(noun):
    """MOVE RUG is a REQUIRED Zork action; it must never touch the map."""
    assert extract_direction(f"MOVE {noun}") is None


@pytest.mark.parametrize("command", [
    "TAKE NORTH STAR", "EXAMINE WESTERN WALL", "READ NORTH SIGN",
    "OPEN WEST DOOR", "PUT SWORD IN CASE", "LOOK UNDER RUG",
])
def test_direction_words_used_as_objects_are_rejected(command):
    assert extract_direction(command) is None


@pytest.mark.parametrize("command,expected", [
    ("north.", "NORTH"), ("  Go   North  ", "NORTH"), ("n", "NORTH"),
    ("go north!", "NORTH"), ("NoRtH", "NORTH"),
])
def test_case_whitespace_and_punctuation(command, expected):
    assert extract_direction(command) == expected


def test_move_is_not_a_movement_verb():
    """The heart of #10 -- documented so nobody re-adds it."""
    assert "MOVE" not in MOVEMENT_VERBS


@pytest.mark.parametrize("command", ["GO UP THE STAIRS", "WALK TO THE NORTH"])
def test_filler_word_phrasings_are_deliberately_rejected(command):
    """Accepted precision trade: a wrong edge is permanent, a missing one is not.

    #14's raw-command edge label is the intended recovery for these.
    """
    assert extract_direction(command) is None


def test_is_direction_rejects_substrings():
    assert is_direction("N") and is_direction("north")
    assert not is_direction("NORTHERN") and not is_direction("MOVE")


# ---------------------------------------------------------------------------
# is_movement_refusal -- phrases quoted from the live backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("response", [
    "You cannot go that way.",                    # Zork + Planetfall, verbatim
    "You can't go that way.",                     # Zork, same session, verbatim
    "You can't get there from here.",             # Zork "enter house", verbatim
    "You cannot go that way.\n\nThis action or command has no effect on the game.",
    "There is a wall there.",
])
def test_refusals_are_recognized(response):
    assert is_movement_refusal(response) is True


@pytest.mark.parametrize("response", [
    "",
    None,
    "Forest",                                      # successful move, brief mode
    "Forest\nThe forest thins out, revealing impassible mountains.",
    "With a great effort, the rug is moved to one side of the room.",
    "This action or command has no effect on the game.",
    # Temporary obstacles: accurate, but must NOT become permanent map facts.
    "The trap door is closed.",
    "The escape pod bulkhead is closed.",
    "The troll fends you off with a menacing gesture.",
    "The door is boarded and you can't remove the boards.",
])
def test_non_refusals_are_not_recognized(response):
    assert is_movement_refusal(response) is False


# ---------------------------------------------------------------------------
# End-to-end through update_from_turn
# ---------------------------------------------------------------------------

def test_move_rug_records_nothing(mapper, mock_db):
    """#10 scenario 1: MOVE RUG used to write Living Room --[EAST]--> BLOCKED,
    permanently burning the real EAST exit to the Kitchen."""
    mapper.previous_location = "Living Room"
    mapper.update_from_turn(
        current_location="Living Room",
        player_command="MOVE RUG",
        turn_number=7,
        game_response="With a great effort, the rug is moved to one side of the "
                      "room. With the rug moved, the dusty cover of a closed "
                      "trap door appears.",
    )
    assert mock_db.transitions == []


def test_east_to_kitchen_still_recordable_after_moving_the_rug(mapper, mock_db):
    """The exit MOVE RUG used to destroy must survive."""
    mapper.previous_location = "Living Room"
    mapper.update_from_turn("Living Room", "MOVE RUG", 7, "the rug is moved")
    mapper.update_from_turn("Kitchen", "EAST", 8, "Kitchen\nYou are in the kitchen")

    assert mapper.get_exits_from("Living Room") == [("EAST", "Kitchen")]


def test_refused_move_still_records_blocked(mapper, mock_db):
    """The feature must survive the fix: real walls are still learned."""
    mapper.previous_location = "South of House"
    mapper.update_from_turn("South of House", "NORTHEAST", 12, "You cannot go that way.")

    assert mapper.get_exits_from("South of House") == [("NORTHEAST", "BLOCKED")]


def test_same_named_rooms_do_not_fabricate_a_wall(mapper, mock_db):
    """#15, reproduced from a live probe: EAST from one Forest reaches a
    different Forest. Name-equality alone used to call that a wall."""
    mapper.previous_location = "Forest"
    mapper.update_from_turn(
        current_location="Forest",
        player_command="EAST",
        turn_number=11,
        game_response="Forest\nThe forest thins out, revealing impassible mountains.",
    )
    assert mock_db.transitions == []


def test_temporary_obstacle_is_not_frozen_into_the_map(mapper, mock_db):
    """DOWN with the trap door shut is a puzzle state, not topology (#11)."""
    mapper.previous_location = "Living Room"
    mapper.update_from_turn("Living Room", "DOWN", 8, "The trap door is closed.")

    assert mock_db.transitions == []


def test_missing_response_suppresses_blocked(mapper, mock_db):
    """Default-safe: no evidence of refusal ⇒ no edge."""
    mapper.previous_location = "Forest"
    mapper.update_from_turn("Forest", "EAST", 11)

    assert mock_db.transitions == []


def test_successful_move_records_regardless_of_response_text(mapper, mock_db):
    """The refusal gate must apply ONLY to the same-name branch."""
    mapper.previous_location = "West Of House"
    mapper.update_from_turn("North of House", "N", 1, "North of House")

    assert mapper.get_exits_from("West Of House") == [("NORTH", "North of House")]


def test_non_movement_command_never_records_even_on_refusal_text(mapper, mock_db):
    mapper.previous_location = "Living Room"
    mapper.update_from_turn("Living Room", "OPEN DOOR", 9, "You cannot go that way.")

    assert mock_db.transitions == []

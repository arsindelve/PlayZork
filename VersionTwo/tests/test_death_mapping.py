"""Death/respawn must not become a map edge (GitHub issue #12).

Zork kills the player, prints an obituary and teleports them to the Forest,
but reports the RESPAWN room as the turn's LocationName. The state-only rule
"the name changed, so the command was a passage" therefore fabricates an edge
from where we died to where we respawned, and BFS routes future journeys
through the fatal move.

The death text below is verbatim from the live Zork backend (2026-08-22).
"""
import pytest

from tools.database import DatabaseManager
from tools.mapping.mapper_state import MapperState
from tools.mapping.pathfinder import PathFinder
from tools.mapping.response_signals import looks_like_death

# Captured live: POST /Prod/ZorkOne, command "north" from the Cellar.
# locationName came back "Forest" (NOT empty), score 35 -> 25.
LIVE_DEATH_RESPONSE = (
    "It is pitch black. You are likely to be eaten by a grue.\n\n"
    "The troll's axe removes your head.  It appears that that last blow was too "
    "much for you. I'm afraid you are dead. \n\n"
    "\t*** You have died ***\n\n"
    "Now, let's take a look here... Well, you probably deserve another chance. "
    "I can't quite fix you up completely, but you can't have everything.\n\n"
    "Forest\nThis is a forest, with trees in all directions. To the east, there "
    "appears to be sunlight.\n"
)

ORDINARY_RESPONSE = (
    "The Troll Room\nThis is a small room with passages to the east and south "
    "and a forbidding hole leading west.\n"
)


class RecordingDb:
    """Minimal stand-in; records whatever the mapper asks it to write."""

    def __init__(self):
        self.transitions = []

    def add_map_transition(self, **kwargs):
        self.transitions.append(kwargs)
        return True

    def get_all_transitions(self, session_id):
        return [
            (t["from_location"], t["to_location"], t["direction"], t["turn_number"])
            for t in self.transitions
        ]

    def get_transitions_from_location(self, session_id, location):
        return [
            (t["direction"], t["to_location"])
            for t in self.transitions
            if t["from_location"] == location
        ]


def _mapper(previous_location="Cellar"):
    db = RecordingDb()
    state = MapperState(session_id="s", db=db)
    state.previous_location = previous_location
    return state, db


# --- the predicate ----------------------------------------------------------

@pytest.mark.parametrize("text", [
    LIVE_DEATH_RESPONSE,
    "\t*** You have died ***",
    "    ****  You have died  ****",
    "Oh, no! A lurking grue slithered into the room and devoured you!\n\n"
    "\t*** You have died ***",
    "You have been killed by the thief.",
    "Your adventure is over.",
])
def test_death_text_is_detected(text):
    assert looks_like_death(text) is True


@pytest.mark.parametrize("text", [
    None,
    "",
    ORDINARY_RESPONSE,
    "The troll is dead.  His body is gone.",              # someone ELSE died
    "This is a dead end.",
    "You have entered the Land of the Living Dead.",
    "It is pitch black. You are likely to be eaten by a grue.",  # warning only
    "Your score is 25 (total of 350 points), in 20 moves.",
])
def test_ordinary_text_is_not_a_death(text):
    assert looks_like_death(text) is False


# --- the mapper gate --------------------------------------------------------

def test_death_turn_records_no_edge():
    """#12: Cellar --NORTH--> Forest is a teleport, not a passage."""
    state, db = _mapper()
    state.update_from_turn("Forest", "north", 40, game_response=LIVE_DEATH_RESPONSE)
    assert db.transitions == []


def test_death_turn_keeps_the_chain_at_the_respawn_room():
    """Unlike the #7 unnamed-room guard, the player really IS in the respawn
    room, so the next turn's move out of it is a genuine edge."""
    state, _ = _mapper()
    state.update_from_turn("Forest", "north", 40, game_response=LIVE_DEATH_RESPONSE)
    assert state.previous_location == "Forest"


def test_move_after_a_death_is_still_mapped():
    state, db = _mapper()
    state.update_from_turn("Forest", "north", 40, game_response=LIVE_DEATH_RESPONSE)
    state.update_from_turn("Clearing", "north", 41, game_response="Clearing\n...")
    assert db.transitions == [dict(
        session_id="s", from_location="Forest", to_location="Clearing",
        direction="NORTH", turn_number=41,
    )]


def test_death_into_a_same_named_room_records_no_blocked_edge():
    """The gate covers the BLOCKED branch too: dying in the Forest and
    respawning in the Forest is not a wall to the north."""
    state, db = _mapper(previous_location="Forest")
    state.update_from_turn("Forest", "north", 40, game_response=LIVE_DEATH_RESPONSE)
    assert db.transitions == []


def test_ordinary_movement_is_unaffected():
    state, db = _mapper()
    state.update_from_turn("The Troll Room", "north", 20, game_response=ORDINARY_RESPONSE)
    assert db.transitions[0]["to_location"] == "The Troll Room"


def test_missing_game_response_preserves_legacy_behaviour():
    """game_response is optional; without it the gate simply cannot fire."""
    state, db = _mapper()
    state.update_from_turn("Forest", "north", 40)
    assert db.transitions[0]["to_location"] == "Forest"


# --- the consequence the issue is actually about ----------------------------

def test_pathfinder_never_routes_through_the_fatal_move():
    state, db = _mapper()
    state.update_from_turn("Forest", "north", 40, game_response=LIVE_DEATH_RESPONSE)
    state.update_from_turn("Clearing", "north", 41, game_response="Clearing\n...")
    assert PathFinder(state).find_path("Cellar", "Clearing") is None


def test_death_does_not_overwrite_a_known_destination(tmp_path):
    """Since #11 a real destination overwrites what is stored, so an ungated
    death turn DESTROYS the correct edge rather than merely being ignored.

    Needs a REAL DatabaseManager: RecordingDb has no UNIQUE constraint and so
    cannot observe upsert behaviour (same reasoning as test_map_corrections).
    """
    db = DatabaseManager(db_path=str(tmp_path / "test.db"))
    state = MapperState(session_id="s", db=db)

    state.previous_location = "Cellar"
    state.update_from_turn("The Troll Room", "north", 20, game_response=ORDINARY_RESPONSE)
    assert state.get_exits_from("Cellar") == [("NORTH", "The Troll Room")]

    state.previous_location = "Cellar"
    state.update_from_turn("Forest", "north", 40, game_response=LIVE_DEATH_RESPONSE)

    assert state.get_exits_from("Cellar") == [("NORTH", "The Troll Room")]
    assert state.pathfinder.find_path("Cellar", "Forest") is None

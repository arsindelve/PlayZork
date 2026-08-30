"""Case-insensitive location lookups (GitHub issue #13).

Casing is not consistent even within one backend. Probed live on 2026-08-22,
the Zork API returns "West Of House" (capital "Of") but "North of House" and
"South of House" (lowercase "of"). Two consequences:

  * a model that has seen "West Of House" in the map writes the siblings by
    analogy as "North Of House", which is wrong; and
  * Zork's own printed name is "West of House", so the model's memorized game
    knowledge disagrees with this backend too.

Location arguments reach the mapper tools straight from a tool-calling LLM
(`bind_tools(..., tool_choice="any")`), so both mistakes are routine. Before
the fix each produced "NO PATH"/"no known exits" for a fully mapped room, and
`prompt_library.get_issue_agent_human_prompt` turns NO PATH into confidence 0.

Lookups are case-insensitive; DISPLAY keeps the backend's own spelling.
"""
from typing import List, Tuple

import pytest

from tools.mapping.locations import normalize_location
from tools.mapping.mapper_state import MapperState
from tools.mapping.mapper_tools import (
    find_path_between_locations,
    get_direction_to_location,
    get_exits_from_location,
    initialize_mapper_tools,
)
from tools.mapping.pathfinder import PathFinder

# The exact strings the live backend returns. Do not "tidy" the casing:
# the inconsistency between "Of" and "of" is the whole point of this module.
WEST = "West Of House"
NORTH = "North of House"
SOUTH = "South of House"
BEHIND = "Behind House"
KITCHEN = "Kitchen"


class MockDatabase:
    """Mirrors test_pathfinder.MockDatabase, but emulates COLLATE NOCASE.

    `get_transitions_from_location` filters in SQL, so a mock that compares
    case-sensitively would hide the very bug under test.
    """

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
            for from_loc, to_loc, direction, _ in self.transitions
            if from_loc.casefold() == location.casefold()  # COLLATE NOCASE
        ]


@pytest.fixture
def mapper_state():
    db = MockDatabase()
    state = MapperState("test-session", db)
    for turn, (a, b, d) in enumerate(
        [
            (WEST, NORTH, "NORTH"),
            (NORTH, BEHIND, "EAST"),
            (BEHIND, SOUTH, "SOUTH"),
            (SOUTH, WEST, "WEST"),
            (BEHIND, KITCHEN, "IN"),
        ],
        start=1,
    ):
        state.record_movement(a, b, d, turn)
    return state


@pytest.fixture
def tools(mapper_state):
    initialize_mapper_tools(mapper_state)
    return mapper_state


# ---------------------------------------------------------------------------
# normalize_location
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", [
    "West Of House", "West of House", "west of house",
    "WEST OF HOUSE", "  West Of House  ", "West  Of  House",
])
def test_all_casings_share_one_key(variant):
    assert normalize_location(variant) == normalize_location(WEST)


def test_distinct_rooms_keep_distinct_keys():
    assert normalize_location(WEST) != normalize_location(NORTH)


def test_empty_and_none_are_safe():
    assert normalize_location(None) == ""
    assert normalize_location("") == ""
    assert normalize_location("   ") == ""


def test_is_idempotent():
    once = normalize_location(WEST)
    assert normalize_location(once) == once


# ---------------------------------------------------------------------------
# The regression the issue reports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("typed", ["North Of House", "north of house", "NORTH OF HOUSE"])
def test_pathfinder_resolves_a_mistyped_destination(mapper_state, typed):
    """'North Of House' is the analogy error the backend's own casing invites."""
    assert PathFinder(mapper_state).find_path(WEST, typed) == ["NORTH"]


@pytest.mark.parametrize("typed", ["west of house", "West of House", "WEST OF HOUSE"])
def test_pathfinder_resolves_a_mistyped_origin(mapper_state, typed):
    assert PathFinder(mapper_state).find_path(typed, KITCHEN) == ["NORTH", "EAST", "IN"]


def test_exits_lookup_is_case_insensitive(mapper_state):
    assert mapper_state.get_exits_from("north of house") == [("EAST", BEHIND)]


def test_same_room_in_different_casings_is_one_graph_node(mapper_state):
    """A room entered under one casing and left under another must not split."""
    mapper_state.record_movement("kitchen", "Attic", "UP", 6)
    assert PathFinder(mapper_state).find_path(WEST, "Attic") == ["NORTH", "EAST", "IN", "UP"]


def test_already_there_ignores_casing(tools):
    assert get_direction_to_location.invoke(
        {"from_location": WEST, "to_location": "west of house"}
    ) == "ALREADY THERE"


def test_no_path_still_means_no_path(mapper_state):
    """Case-insensitivity must not invent routes to rooms that are unmapped."""
    assert PathFinder(mapper_state).find_path(WEST, "Cyclops Room") is None


def test_blocked_is_still_excluded(mapper_state):
    mapper_state.record_movement(KITCHEN, "BLOCKED", "DOWN", 7)
    assert PathFinder(mapper_state).find_path(WEST, "blocked") is None


# ---------------------------------------------------------------------------
# Display fidelity: lookups fold case, output must not
# ---------------------------------------------------------------------------


def test_exits_output_echoes_the_backend_spelling(tools):
    """The model must learn the real name, not see its own guess reflected."""
    out = get_exits_from_location.invoke({"location": "NORTH OF HOUSE"})
    assert f"KNOWN EXITS FROM '{NORTH}'" in out
    assert "NORTH OF HOUSE" not in out


def test_destination_names_keep_their_casing(tools):
    out = get_exits_from_location.invoke({"location": "behind house"})
    assert SOUTH in out and KITCHEN in out


def test_get_map_is_untouched_by_normalization(tools):
    from tools.mapping.mapper_tools import get_map
    out = get_map.invoke({})
    for room in (WEST, NORTH, SOUTH, BEHIND, KITCHEN):
        assert f"'{room}'" in out


def test_transitions_keep_original_casing(mapper_state):
    stored = {t.from_location for t in mapper_state.get_all_transitions()}
    assert WEST in stored and NORTH in stored


def test_unknown_room_echoes_what_the_caller_typed(tools):
    """Nothing to correct it to, so don't silently rewrite the caller's text."""
    out = get_exits_from_location.invoke({"location": "Cyclops Room"})
    assert "Cyclops Room" in out


# ---------------------------------------------------------------------------
# Legacy rows written before the fix must self-heal on read
# ---------------------------------------------------------------------------


def test_case_variant_rows_merge_on_read(mapper_state):
    """A resumed session may hold both casings; read-side collapse merges them."""
    mapper_state.record_movement("west of house", "BLOCKED", "NORTH", 8)
    exits = mapper_state.get_exits_from(WEST)
    assert exits == [("NORTH", NORTH)], "the real passage must beat the BLOCKED row"


def test_tool_path_matches_python_path(tools):
    assert find_path_between_locations.invoke(
        {"from_location": "west of house", "to_location": "kitchen"}
    ) == "NORTH, EAST, IN"


# ---------------------------------------------------------------------------
# The SQL predicates, against a real SQLite file.
#
# The MockDatabase above emulates COLLATE NOCASE, which is exactly what makes
# it useless for verifying the SQL. These two exercise DatabaseManager itself.
# ---------------------------------------------------------------------------


@pytest.fixture
def real_db(tmp_path):
    from tools.database import DatabaseManager
    db = DatabaseManager(str(tmp_path / "t.db"))
    db.create_session("s")
    return db


def test_sql_exits_predicate_folds_case(real_db):
    real_db.add_map_transition("s", WEST, NORTH, "NORTH", 1)
    assert real_db.get_transitions_from_location("s", "west of house") == [("NORTH", NORTH)]
    assert real_db.get_transitions_from_location("s", "WEST OF HOUSE") == [("NORTH", NORTH)]


def test_sql_memory_predicate_folds_case(real_db):
    real_db.add_memory("s", 1, "The mailbox holds a leaflet", 500, WEST, 0, 0)
    found = real_db.get_location_memories("s", "west of house")
    assert len(found) == 1 and found[0][0] == "The mailbox holds a leaflet"


def test_sql_predicate_does_not_over_match(real_db):
    """Folding case must not collapse genuinely different rooms."""
    real_db.add_map_transition("s", WEST, NORTH, "NORTH", 1)
    assert real_db.get_transitions_from_location("s", "North of House") == []

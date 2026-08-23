"""The backend already knows what the agents were inferring (GitHub issue #30).

`PreviousLocationName`, `LastMovementDirection`, `Exits` and `Inventory` are
populated on both hosted backends but were declared-and-unread or not declared
at all, so the mapper re-derived direction by parsing English and the
InventoryAnalyzer guessed at an inventory the server was reporting exactly.
"""
from types import SimpleNamespace
from typing import List, Tuple

import pytest

from tools.agent_graph.decision_graph import create_persist_node
from tools.mapping.mapper_state import MapperState
from zork.zork_api_response import ZorkApiResponse


# ---------------------------------------------------------------------------
# The response model carries the fields
# ---------------------------------------------------------------------------


def test_response_model_declares_the_previously_unread_fields():
    payload = {
        "response": "Kitchen\nYou are in the kitchen.",
        "locationName": "Kitchen",
        "previousLocationName": "Behind House",
        "lastMovementDirection": "In",
        "exits": [10, 3, 2, 9],
        "inventory": ["brass lantern", "sword"],
        "actionsAvailableFromLocation": {"window": ["open window"]},
        "actionsAvailableFromInventory": {},
        "moves": 6,
        "score": 0,
        "time": 0,
    }
    r = ZorkApiResponse(**payload)

    assert r.PreviousLocationName == "Behind House"
    assert r.LastMovementDirection == "In"
    assert r.Exits == [10, 3, 2, 9]
    assert r.Inventory == ["brass lantern", "sword"]
    assert r.ActionsAvailableFromLocation == {"window": ["open window"]}


def test_missing_new_fields_stay_none():
    """Backends that do not send them (e.g. the local Escape Room) must not break."""
    r = ZorkApiResponse(response="ok", locationName="Kitchen", moves=1, score=0)

    assert r.Exits is None
    assert r.Inventory is None
    assert r.LastMovementDirection is None


# ---------------------------------------------------------------------------
# The mapper prefers the backend's own direction
# ---------------------------------------------------------------------------


class MockDatabase:
    def __init__(self):
        self.transitions: List[Tuple[str, str, str, int]] = []

    def add_map_transition(self, session_id, from_location, to_location, direction, turn_number):
        self.transitions.append((from_location, to_location, direction, turn_number))
        return True

    def get_all_transitions(self, session_id):
        return self.transitions

    def get_transitions_from_location(self, session_id, location):
        return [(d, t) for f, t, d, _ in self.transitions if f == location]


@pytest.fixture
def mapper(mock_db=None):
    return MapperState(session_id="s", db=MockDatabase())


@pytest.mark.parametrize("command,api_dir,expected", [
    ("CLIMB TREE", "Up", "UP"),        # canonical AND executable, beats the raw label
    ("ENTER WINDOW", "In", "IN"),
    ("GO UP THE STAIRS", "Up", "UP"),  # the tokenizer deliberately returns None here
])
def test_backend_direction_is_used_when_the_command_is_not_a_bare_direction(mapper, command, api_dir, expected):
    mapper.previous_location = "A"
    mapper.update_from_turn("B", command, 1, game_response="You move.", api_direction=api_dir)

    assert mapper.get_exits_from("A") == [(expected, "B")]


def test_an_explicit_direction_command_still_wins(mapper):
    """The tokenizer is exact for bare directions; no need to defer."""
    mapper.previous_location = "A"
    mapper.update_from_turn("B", "NORTH", 1, game_response="ok", api_direction="N")

    assert mapper.get_exits_from("A") == [("NORTH", "B")]


def test_a_sticky_direction_cannot_map_a_non_movement_command(mapper):
    """LastMovementDirection keeps its old value on turns that attempt no move.
    If a timed event relocated us during TAKE LAMP, the stale direction must
    not become an edge."""
    mapper.previous_location = "A"
    mapper.update_from_turn("B", "TAKE LAMP", 1, game_response="Taken.", api_direction="W")

    assert mapper.get_exits_from("A") == []


def test_no_api_direction_falls_back_to_the_raw_command_label(mapper):
    """Backends without the field keep #14's behaviour."""
    mapper.previous_location = "Forest Path"
    mapper.update_from_turn("Up A Tree", "CLIMB TREE", 1, game_response="ok", api_direction=None)

    assert mapper.get_exits_from("Forest Path") == [("CLIMB TREE", "Up A Tree")]


# ---------------------------------------------------------------------------
# Inventory comes from the game, not from an LLM
# ---------------------------------------------------------------------------


class FakeInventoryState:
    def __init__(self):
        self.synced = None

    def sync_with_game(self, items, turn_number):
        self.synced = (list(items), turn_number)

    def get_items(self):
        return list(self.synced[0]) if self.synced else []

    def add_item(self, *a): pass
    def remove_item(self, *a): pass


def _persist(zork_response, inventory_state, analyzer_calls):
    import tools.inventory as inv_mod

    class RecordingAnalyzer:
        def __init__(self, llm): pass

        def analyze_turn(self, player_command, game_response, current_inventory=None):
            analyzer_calls.append(player_command)
            return SimpleNamespace(items_added=[], items_removed=[], reasoning="")

    original = inv_mod.InventoryAnalyzer
    inv_mod.InventoryAnalyzer = RecordingAnalyzer
    try:
        persist = create_persist_node(
            SimpleNamespace(state=SimpleNamespace(remove_memory=lambda i: True),
                            add_memory=lambda **k: True),
            SimpleNamespace(state=inventory_state),
            {"current": 4},
        )
        persist({
            "game_response": zork_response,
            "player_command": "TAKE SWORD",
            "decision": SimpleNamespace(command="LOOK"),
            "observer_response": None,
            "pending_closures": [],
            "issue_closed_response": None,
        })
    finally:
        inv_mod.InventoryAnalyzer = original


def test_persist_syncs_from_the_game_inventory_and_skips_the_llm():
    state = FakeInventoryState()
    calls = []
    response = SimpleNamespace(
        Response="Taken.", LocationName="Living Room", Score=0, Moves=4,
        Inventory=["sword", "brass lantern"],
    )

    _persist(response, state, calls)

    assert state.synced == (["sword", "brass lantern"], 4)
    assert calls == [], "the LLM analyzer must not run when the game reports inventory"


def test_persist_falls_back_to_the_analyzer_when_the_backend_omits_inventory():
    state = FakeInventoryState()
    calls = []
    response = SimpleNamespace(
        Response="Taken.", LocationName="Living Room", Score=0, Moves=4,
        Inventory=None,
    )

    _persist(response, state, calls)

    assert state.synced is None
    assert calls == ["TAKE SWORD"]

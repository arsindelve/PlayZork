"""A missing LocationName must not become a fabricated room (GitHub issue #7).

The backend returns no LocationName in darkness, some cutscenes and death
sequences. Call sites papered over that with `LocationName or "Unknown"`,
turning a missing value into a room that does not exist: the explorer claimed
all ten directions unexplored from it (its maximum possible EV, reached
exactly when we know least), every IssueAgent pathfound out of it and got
NO PATH -> mandatory confidence 0, and memories were stored anchored to it.

"Unknown" survives as prose in prompts; it must never be data.
"""
import asyncio
from types import SimpleNamespace

import tools.agent_graph.decision_graph as dg
from tools.agent_graph.decision_graph import create_persist_node, create_spawn_agents_node
from tools.mapping.mapper_state import MapperState


class FakeInventoryState:
    def add_item(self, item_name, turn_number):
        pass

    def remove_item(self, item_name, turn_number):
        pass

    def get_items(self):
        return []


def _persist_with(location_name, monkeypatch):
    captured = {}

    class FakeInventoryAnalyzer:
        def __init__(self, llm):
            pass

        def analyze_turn(self, player_command, game_response):
            return SimpleNamespace(items_added=[], items_removed=[], reasoning="none")

    monkeypatch.setattr("tools.inventory.InventoryAnalyzer", FakeInventoryAnalyzer)
    monkeypatch.setattr("config.get_cheap_llm", lambda temperature=0: object())

    def fake_add_memory(**kwargs):
        captured.update(kwargs)
        return True

    persist = create_persist_node(
        SimpleNamespace(add_memory=fake_add_memory),
        SimpleNamespace(state=FakeInventoryState()),
        {"current": 3},
    )
    state = {
        "game_response": SimpleNamespace(
            Response="It is pitch black.", LocationName=location_name, Score=0, Moves=1
        ),
        "player_command": "NORTH",
        "decision": SimpleNamespace(command="LOOK"),
        "observer_response": SimpleNamespace(
            remember="a grue lurks in the dark", rememberImportance=800, item=""
        ),
    }
    persist(state)
    return captured


def test_persist_does_not_fabricate_a_location_name(monkeypatch):
    """#7: a missing LocationName must not be stored as the fake room 'Unknown'."""
    captured = _persist_with(None, monkeypatch)
    assert captured["location"] == ""


def test_persist_still_stores_a_real_location(monkeypatch):
    captured = _persist_with("West Of House", monkeypatch)
    assert captured["location"] == "West Of House"


# --- spawn node -------------------------------------------------------------

class FakeExplorerAgent:
    def __init__(self, current_location, unexplored_directions, mentioned_directions, turn_number):
        self.current_location = current_location
        self.unexplored_directions = unexplored_directions
        self.mentioned_directions = mentioned_directions
        self.best_direction = "NORTH"
        self.proposed_action = None
        self.reason = None
        self.confidence = None
        self.tool_calls_history = []

    async def propose(self, **kwargs):
        self.proposed_action = self.best_direction
        self.confidence = 95


class FakeInteractionAgent:
    def __init__(self):
        self.proposed_action = "nothing"
        self.reason = ""
        self.confidence = 0
        self.detected_objects = []
        self.inventory_items = []
        self.tool_calls_history = []

    async def propose(self, **kwargs):
        self.confidence = 0


def _run_spawn(monkeypatch, location_name):
    monkeypatch.setattr(dg, "ExplorerAgent", FakeExplorerAgent)
    monkeypatch.setattr(dg, "InteractionAgent", FakeInteractionAgent)
    monkeypatch.setattr("tools.analysis.get_analysis_tools", lambda: [])

    empty_tools = SimpleNamespace(get_tools=lambda: [])
    memory_toolkit = SimpleNamespace(state=SimpleNamespace(get_top_memories=lambda **kw: []))
    mapper_toolkit = SimpleNamespace(
        get_tools=lambda: [],
        state=SimpleNamespace(get_exits_from=lambda loc: []),  # nothing mapped yet
    )
    node = create_spawn_agents_node(
        memory_toolkit,
        mapper_toolkit,
        empty_tools,
        decision_llm=object(),
        history_toolkit=empty_tools,
        turn_number_ref={"current": 7},
    )
    state = {
        "game_response": SimpleNamespace(
            LocationName=location_name,
            Response="It is pitch black. You are likely to be eaten by a grue.",
            Score=0,
            Moves=1,
        ),
    }
    return asyncio.run(node(state))


def test_no_explorer_agent_when_current_location_is_unknown(monkeypatch):
    """#7: with no LocationName there is no map node to explore *from*."""
    result = _run_spawn(monkeypatch, None)
    assert result["explorer_agent"] is None


def test_explorer_agent_still_spawns_when_location_is_known(monkeypatch):
    result = _run_spawn(monkeypatch, "West Of House")
    assert result["explorer_agent"] is not None
    assert result["explorer_agent"].current_location == "West Of House"


# --- mapper -----------------------------------------------------------------

class RecordingDb:
    def __init__(self):
        self.transitions = []

    def add_map_transition(self, **kwargs):
        self.transitions.append(kwargs)
        return True


def test_mapper_records_no_transition_into_an_unknown_location():
    """#7 adjacent: to_location is NOT NULL; the insert is doomed and the
    IntegrityError is swallowed as 'already known'."""
    db = RecordingDb()
    state = MapperState(session_id="s", db=db)
    state.previous_location = "West Of House"

    state.update_from_turn(current_location=None, player_command="NORTH", turn_number=5)

    assert db.transitions == []
    assert state.previous_location is None

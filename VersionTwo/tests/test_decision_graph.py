from types import SimpleNamespace

from tools.agent_graph.decision_graph import create_persist_node


class FakeInventoryState:
    def add_item(self, item_name, turn_number):
        pass

    def remove_item(self, item_name, turn_number):
        pass

    def get_items(self):
        return []


def test_persist_analyzes_command_that_produced_game_response(monkeypatch):
    captured = {}

    class FakeInventoryAnalyzer:
        def __init__(self, llm):
            pass

        def analyze_turn(self, player_command, game_response):
            captured["player_command"] = player_command
            captured["game_response"] = game_response
            return SimpleNamespace(items_added=[], items_removed=[], reasoning="none")

    monkeypatch.setattr("tools.inventory.InventoryAnalyzer", FakeInventoryAnalyzer)
    monkeypatch.setattr("config.get_cheap_llm", lambda temperature=0: object())

    memory_toolkit = SimpleNamespace(add_memory=lambda **kwargs: True)
    inventory_toolkit = SimpleNamespace(state=FakeInventoryState())
    persist = create_persist_node(
        memory_toolkit,
        inventory_toolkit,
        {"current": 3},
    )
    state = {
        "game_response": SimpleNamespace(
            Response="Response to OPEN MAILBOX",
            LocationName="West Of House",
            Score=0,
            Moves=1,
        ),
        "player_command": "OPEN MAILBOX",
        "decision": SimpleNamespace(command="TAKE LEAFLET"),
        "observer_response": SimpleNamespace(
            remember="",
            rememberImportance=None,
            item="",
        ),
    }

    persist(state)

    assert captured == {
        "player_command": "OPEN MAILBOX",
        "game_response": "Response to OPEN MAILBOX",
    }


# ---------------------------------------------------------------------------
# Failure containment on the turn path (GitHub issue #1)
# ---------------------------------------------------------------------------

import asyncio

import tools.agent_graph.decision_graph as dg
from tools.agent_graph.decision_graph import (
    create_close_issues_node,
    create_observe_node,
    create_spawn_agents_node,
)


ALL_DIRECTIONS = [
    "NORTH", "SOUTH", "EAST", "WEST",
    "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST",
    "UP", "DOWN",
]


class FakeMemory:
    def __init__(self, mem_id, content, location="West Of House"):
        self.id = mem_id
        self.content = content
        self.location = location
        self.importance = 500
        self.turn_number = 1
        self.score = 0
        self.moves = 1


def _spawn_node_fixtures(monkeypatch, memories, agent_behaviors):
    """Wire a spawn node whose IssueAgents behave per `agent_behaviors`.

    agent_behaviors maps issue content -> None (succeed) or an Exception to raise.
    """
    created = []

    class FakeIssueAgent:
        def __init__(self, memory):
            self.memory = memory
            self.issue_content = memory.content
            self.importance = memory.importance
            self.location = memory.location
            self.proposed_action = None
            self.reason = None
            self.confidence = None
            self.tool_calls_history = []
            created.append(self)

        async def research_and_propose(self, **kwargs):
            behavior = agent_behaviors.get(self.issue_content)
            if isinstance(behavior, Exception):
                raise behavior
            self.proposed_action = f"SOLVE {self.issue_content}"
            self.reason = "because"
            self.confidence = 70

    class FakeInteractionAgent:
        def __init__(self):
            self.proposed_action = "nothing"
            self.reason = ""
            self.confidence = 0
            self.detected_objects = []
            self.inventory_items = []
            self.tool_calls_history = []

        async def research_and_propose(self, **kwargs):
            self.proposed_action = "OPEN MAILBOX"
            self.confidence = 80

    monkeypatch.setattr(dg, "IssueAgent", FakeIssueAgent)
    monkeypatch.setattr(dg, "InteractionAgent", FakeInteractionAgent)
    monkeypatch.setattr("tools.analysis.get_analysis_tools", lambda: [])

    empty_tools = SimpleNamespace(get_tools=lambda: [])
    memory_toolkit = SimpleNamespace(
        state=SimpleNamespace(get_top_memories=lambda **kwargs: memories)
    )
    mapper_toolkit = SimpleNamespace(
        get_tools=lambda: [],
        # All directions already known => no ExplorerAgent spawned.
        state=SimpleNamespace(
            get_exits_from=lambda loc: [(d, "Somewhere") for d in ALL_DIRECTIONS]
        ),
    )

    node = create_spawn_agents_node(
        memory_toolkit,
        mapper_toolkit,
        empty_tools,
        research_agent=object(),
        decision_llm=object(),
        history_toolkit=empty_tools,
        turn_number_ref={"current": 7},
    )
    state = {
        "game_response": SimpleNamespace(
            LocationName="West Of House",
            Response="You are standing in an open field.",
            Score=0,
            Moves=1,
        ),
    }
    return node, state, created


def test_one_failing_agent_does_not_kill_the_turn(monkeypatch):
    """Issue #1: gather() without return_exceptions cancelled every sibling
    agent and ended the session."""
    memories = [FakeMemory(1, "open the mailbox"), FakeMemory(2, "enter the house")]
    behaviors = {"open the mailbox": ValueError("to_location Field required")}
    node, state, created = _spawn_node_fixtures(monkeypatch, memories, behaviors)

    result = asyncio.run(node(state))

    failed = next(a for a in created if a.issue_content == "open the mailbox")
    survivor = next(a for a in created if a.issue_content == "enter the house")

    # The sibling agent still produced its proposal.
    assert survivor.proposed_action == "SOLVE enter the house"
    assert survivor.confidence == 70
    # The failed agent cannot advocate...
    assert failed.proposed_action is None
    assert failed.confidence is None
    # ...but explains itself in the report.
    assert "failed during research" in failed.reason
    # Both agents remain in state; the always-on InteractionAgent still ran.
    assert result["issue_agents"] == created
    assert result["interaction_agent"].proposed_action == "OPEN MAILBOX"


def test_failed_agents_are_excluded_from_proposals(monkeypatch):
    memories = [FakeMemory(1, "open the mailbox")]
    behaviors = {"open the mailbox": RuntimeError("model returned garbage")}
    node, state, _ = _spawn_node_fixtures(monkeypatch, memories, behaviors)

    result = asyncio.run(node(state))

    proposals = dg._format_agent_proposals(
        result["issue_agents"],
        result["explorer_agent"],
        result["loop_detection_agent"],
        result["interaction_agent"],
    )
    assert "open the mailbox" not in proposals


def test_close_issues_failure_does_not_discard_the_decision(monkeypatch):
    class ExplodingIssueCloser:
        def analyze(self, **kwargs):
            raise RuntimeError("structured output parse failed")

    monkeypatch.setattr(dg, "IssueClosedAgent", ExplodingIssueCloser)

    node = create_close_issues_node(
        decision_llm=object(),
        history_toolkit=SimpleNamespace(get_tools=lambda: []),
        memory_toolkit=object(),
    )
    state = {
        "game_response": SimpleNamespace(
            LocationName="West Of House", Response="ok", Score=0, Moves=1
        ),
        "decision": SimpleNamespace(command="OPEN MAILBOX"),
    }

    result = node(state)

    assert result["issue_closed_response"] is None
    assert result["decision"].command == "OPEN MAILBOX"


def test_observe_failure_does_not_discard_the_decision(monkeypatch):
    class ExplodingObserver:
        def observe(self, **kwargs):
            raise RuntimeError("timed out after 5 attempts")

    monkeypatch.setattr(dg, "ObserverAgent", ExplodingObserver)

    node = create_observe_node(
        decision_llm=object(),
        research_agent=object(),
        history_toolkit=SimpleNamespace(get_tools=lambda: []),
        memory_toolkit=object(),
    )
    state = {
        "game_response": SimpleNamespace(
            LocationName="West Of House", Response="ok", Score=0, Moves=1
        ),
        "decision": SimpleNamespace(command="OPEN MAILBOX"),
    }

    result = node(state)

    assert result["observer_response"] is None
    assert result["decision"].command == "OPEN MAILBOX"


def test_persist_handles_missing_observer_response(monkeypatch):
    """observe_node may now yield None; persist must still update inventory."""
    captured = {}

    class FakeInventoryAnalyzer:
        def __init__(self, llm):
            pass

        def analyze_turn(self, player_command, game_response):
            captured["player_command"] = player_command
            return SimpleNamespace(items_added=[], items_removed=[], reasoning="none")

    monkeypatch.setattr("tools.inventory.InventoryAnalyzer", FakeInventoryAnalyzer)
    monkeypatch.setattr("config.get_cheap_llm", lambda temperature=0: object())

    persist = create_persist_node(
        SimpleNamespace(add_memory=lambda **kwargs: True),
        SimpleNamespace(state=FakeInventoryState()),
        {"current": 3},
    )
    state = {
        "game_response": SimpleNamespace(
            Response="Opened.", LocationName="West Of House", Score=0, Moves=1
        ),
        "player_command": "OPEN MAILBOX",
        "decision": SimpleNamespace(command="TAKE LEAFLET"),
        "observer_response": None,
    }

    result = persist(state)

    assert result["memory_persisted"] is False
    assert captured["player_command"] == "OPEN MAILBOX"


def test_persist_survives_inventory_analysis_failure(monkeypatch):
    class ExplodingInventoryAnalyzer:
        def __init__(self, llm):
            pass

        def analyze_turn(self, player_command, game_response):
            raise RuntimeError("ollama connection reset")

    monkeypatch.setattr("tools.inventory.InventoryAnalyzer", ExplodingInventoryAnalyzer)
    monkeypatch.setattr("config.get_cheap_llm", lambda temperature=0: object())

    persist = create_persist_node(
        SimpleNamespace(add_memory=lambda **kwargs: True),
        SimpleNamespace(state=FakeInventoryState()),
        {"current": 3},
    )
    state = {
        "game_response": SimpleNamespace(
            Response="Opened.", LocationName="West Of House", Score=0, Moves=1
        ),
        "player_command": "OPEN MAILBOX",
        "decision": SimpleNamespace(command="TAKE LEAFLET"),
        "observer_response": SimpleNamespace(
            remember="the mailbox is open", rememberImportance=600, item=""
        ),
    }

    result = persist(state)

    # The memory was still stored even though inventory analysis blew up.
    assert result["memory_persisted"] is True

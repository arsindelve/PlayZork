"""The turn's facts are fetched in code, not requested from a model (#25).

Every agent used to open with an LLM "research" round-trip that named the
tools to call, ran them once and never iterated — 176s of a 445s measured
turn. TurnContext replaces that. These tests pin both halves: the data is
right, and the calls are actually gone.
"""
from types import SimpleNamespace

import pytest

from tools.agent_graph.turn_context import TurnContext, build_turn_context
from tools.mapping.mapper_state import MapperState


class MockDb:
    def __init__(self, transitions=None):
        self.t = transitions or []

    def add_map_transition(self, *a, **k):
        return True

    def get_all_transitions(self, s):
        return self.t

    def get_transitions_from_location(self, s, loc):
        return [(d, to) for f, to, d, _ in self.t if f == loc]


def _toolkits(transitions=None, inventory=None):
    mapper = SimpleNamespace(state=MapperState(session_id="s", db=MockDb(transitions)))
    history = SimpleNamespace(state=SimpleNamespace(
        get_full_summary=lambda: "recent summary",
        get_long_running_summary=lambda: "story so far",
        get_recent_turns=lambda n: [
            SimpleNamespace(turn_number=1, player_command="LOOK", game_response="West Of House")
        ],
    ))
    inv = SimpleNamespace(state=SimpleNamespace(get_items=lambda: list(inventory or [])))
    return history, mapper, inv


def _response(**kw):
    base = dict(LocationName="West Of House", Response="You are here.", Score=0, Moves=3)
    base.update(kw)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def test_inventory_prefers_the_game_over_our_own_tracking():
    """#30: the backend reports it exactly; our tracking is only a fallback."""
    history, mapper, inv = _toolkits(inventory=["stale item"])

    ctx = build_turn_context(
        game_response=_response(Inventory=["brass lantern", "sword"]),
        history_toolkit=history, mapper_toolkit=mapper, inventory_toolkit=inv,
    )

    assert ctx.inventory == ["brass lantern", "sword"]


def test_inventory_falls_back_when_the_backend_omits_it():
    history, mapper, inv = _toolkits(inventory=["tracked item"])

    ctx = build_turn_context(
        game_response=_response(),  # no Inventory attribute
        history_toolkit=history, mapper_toolkit=mapper, inventory_toolkit=inv,
    )

    assert ctx.inventory == ["tracked item"]


def test_directions_are_precomputed_for_every_spawned_issue():
    transitions = [("West Of House", "North of House", "NORTH", 1),
                   ("North of House", "Kitchen", "EAST", 2)]
    history, mapper, inv = _toolkits(transitions)

    ctx = build_turn_context(
        game_response=_response(), history_toolkit=history,
        mapper_toolkit=mapper, inventory_toolkit=inv,
        issue_locations=["Kitchen"],
    )

    assert ctx.direction_to("Kitchen") == "NORTH"
    assert ctx.direction_to("kitchen") == "NORTH", "lookups must be casefolded (#13)"


def test_unreachable_and_unnamed_targets_are_distinguished():
    history, mapper, inv = _toolkits()
    ctx = build_turn_context(
        game_response=_response(), history_toolkit=history,
        mapper_toolkit=mapper, inventory_toolkit=inv, issue_locations=["Nowhere"],
    )

    assert ctx.direction_to("Nowhere") == "NO PATH"
    # An issue the game never gave a room name is not the same as unreachable.
    assert ctx.direction_to("Unknown") == "NOT AVAILABLE"
    assert ctx.direction_to(None) == "NOT AVAILABLE"


def test_no_exits_are_looked_up_without_a_room_name():
    """#7: 'Unknown' is not a map node."""
    history, mapper, inv = _toolkits()

    ctx = build_turn_context(
        game_response=_response(LocationName=None),
        history_toolkit=history, mapper_toolkit=mapper, inventory_toolkit=inv,
    )

    assert ctx.exits == []


def test_a_broken_source_degrades_one_field_not_the_turn():
    """Each read is guarded independently (#1). A toolkit missing `.state`
    must not cost the whole context."""
    history = SimpleNamespace()          # no .state at all
    _, mapper, inv = _toolkits()

    ctx = build_turn_context(
        game_response=_response(Inventory=["lamp"]),
        history_toolkit=history, mapper_toolkit=mapper, inventory_toolkit=inv,
    )

    assert ctx.full_summary == ""        # degraded
    assert ctx.inventory == ["lamp"]     # unaffected
    assert ctx.location == "West Of House"


def test_empty_inventory_never_renders_as_a_blank_string():
    ctx = TurnContext(location="X", game_text="", score=0, moves=0)
    assert ctx.inventory_summary == "empty"
    assert "INVENTORY: empty" in ctx.research_context_for()


# ---------------------------------------------------------------------------
# The calls are actually gone
# ---------------------------------------------------------------------------


def test_no_agent_has_a_research_phase_any_more():
    from tools.agent_graph.explorer_agent import ExplorerAgent
    from tools.agent_graph.interaction_agent import InteractionAgent
    from tools.agent_graph.issue_agent import IssueAgent

    for cls in (IssueAgent, ExplorerAgent, InteractionAgent):
        assert not hasattr(cls, "research_and_propose"), cls.__name__
        assert hasattr(cls, "propose"), cls.__name__


def test_the_graph_no_longer_has_a_research_node():
    import tools.agent_graph.decision_graph as dg

    assert not hasattr(dg, "create_research_node")

    stub = SimpleNamespace(get_tools=lambda: [], state=SimpleNamespace())
    graph = dg.create_decision_graph(
        decision_chain=object(), decision_llm=object(),
        history_toolkit=stub, memory_toolkit=stub, mapper_toolkit=stub,
        inventory_toolkit=stub, turn_number_ref={"current": 0},
    )
    nodes = set(graph.get_graph().nodes)
    assert "research" not in nodes
    assert {"spawn_agents", "decide"} <= nodes


def test_the_observer_no_longer_researches():
    """It used to make a research round-trip whose results were executed
    against a map holding only 2 of the 8 bound tools (#5), so it routinely
    decided what to persist to long-term memory with no history at all."""
    import inspect

    from tools.agent_graph.observer_agent import ObserverAgent

    source = inspect.getsource(ObserverAgent.observe)
    assert "research_agent" not in source
    # exactly one LLM call left in the observer
    assert source.count("invoke_with_retry(") == 1

    params = inspect.signature(ObserverAgent.observe).parameters
    assert "context" in params


def test_every_agent_actually_runs_end_to_end(monkeypatch):
    """Executes each agent rather than inspecting its source.

    A source-only assertion missed a real NameError here: removing the
    research phase also removed the function-local `from llm_utils import ...`
    that the surviving proposal call depended on. The failure surfaced only in
    a live run, where #1's containment logged "OBSERVE failed" and silently
    disabled the Observer for the whole session.
    """
    import asyncio

    import llm_utils
    from langchain_core.runnables import RunnableLambda

    from tools.agent_graph.explorer_agent import ExplorerAgent
    from tools.agent_graph.interaction_agent import InteractionAgent
    from tools.agent_graph.issue_agent import IssueAgent
    from tools.agent_graph.observer_agent import ObserverAgent

    calls = []

    class Result:
        proposed_action = "NORTH"
        reason = "r"
        confidence = 80
        detected_objects: list = []
        inventory_items: list = []
        remember = ""
        rememberImportance = 500
        item = ""

    async def fake_a(chain, payload, operation_name="", **kw):
        calls.append(operation_name)
        return Result()

    def fake_s(chain, payload, operation_name="", **kw):
        calls.append(operation_name)
        return Result()

    monkeypatch.setattr(llm_utils, "ainvoke_with_retry", fake_a)
    monkeypatch.setattr(llm_utils, "invoke_with_retry", fake_s)

    llm = SimpleNamespace(with_structured_output=lambda schema: RunnableLambda(lambda _: None))
    ctx = TurnContext(location="West Of House", game_text="You are here.", score=0, moves=1,
                      inventory=["lamp"])

    memory = SimpleNamespace(id=1, content="open the mailbox", importance=500,
                             turn_number=1, location="West Of House", score=0, moves=1)
    asyncio.run(IssueAgent(memory).propose(decision_llm=llm, context=ctx))
    asyncio.run(ExplorerAgent("West Of House", ["NORTH"], [], 1).propose(decision_llm=llm, context=ctx))
    asyncio.run(InteractionAgent().propose(decision_llm=llm, context=ctx))
    ObserverAgent().observe(
        game_response="You are here.", location="West Of House", score=0, moves=1,
        decision_llm=llm,
        memory_toolkit=SimpleNamespace(state=SimpleNamespace(get_top_memories=lambda **k: [])),
        context=ctx,
    )

    # Four agents, four calls — no research round-trips left.
    assert len(calls) == 4, calls

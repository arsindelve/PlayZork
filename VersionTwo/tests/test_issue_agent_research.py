"""IssueAgent proposals are fed from the deterministic TurnContext (#25).

This file previously tested the agent's *research phase* — an LLM round-trip
that named the tools to call, ran them once, and never iterated (#6 lived
here: it named a tool that did not exist). That phase is gone; the same facts
now arrive from TurnContext. The guarantees it protected still matter, so they
are re-pinned against the new path — where they hold by construction rather
than by prompt discipline.
"""
import asyncio
from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda

import llm_utils
from tools.agent_graph.issue_agent import IssueAgent, IssueProposal
from tools.agent_graph.turn_context import TurnContext


def make_agent(issue_location="Exit Hallway"):
    memory = SimpleNamespace(
        id=1, content="Locked metal door at Exit Hallway - need to unlock",
        importance=800, turn_number=4, location=issue_location, score=0, moves=4,
    )
    return IssueAgent(memory)


def run_agent(monkeypatch, context, issue_location="Exit Hallway"):
    """Drive propose() and capture what reached the proposal prompt."""
    captured = {}

    async def fake_ainvoke_with_retry(chain, payload, operation_name="", **kwargs):
        captured["proposal_input"] = payload
        return IssueProposal(proposed_action="SOUTH", reason="r", confidence=90)

    monkeypatch.setattr(llm_utils, "ainvoke_with_retry", fake_ainvoke_with_retry)

    agent = make_agent(issue_location)
    decision_llm = SimpleNamespace(
        with_structured_output=lambda schema: RunnableLambda(lambda _: None)
    )
    asyncio.run(agent.propose(decision_llm=decision_llm, context=context))
    return captured


def _context(**kwargs):
    base = dict(location="Storage Closet", game_text="You are in a storage closet.",
                score=0, moves=4)
    base.update(kwargs)
    return TurnContext(**base)


def test_inventory_reaches_the_proposal_prompt(monkeypatch):
    ctx = _context(inventory=["brass key", "leaflet"])

    captured = run_agent(monkeypatch, ctx)

    assert captured["proposal_input"]["inventory_summary"] == "brass key, leaflet"


def test_empty_inventory_renders_as_empty_not_blank(monkeypatch):
    """A blank string in a prompt reads as a missing value, not as 'nothing'."""
    captured = run_agent(monkeypatch, _context(inventory=[]))

    assert captured["proposal_input"]["inventory_summary"] == "empty"


def test_precomputed_direction_reaches_the_proposal_prompt(monkeypatch):
    ctx = _context(directions={"exit hallway": "SOUTH"})

    captured = run_agent(monkeypatch, ctx)

    assert captured["proposal_input"]["navigation_direction"] == "SOUTH"


def test_unroutable_issue_reports_no_path(monkeypatch):
    captured = run_agent(monkeypatch, _context(directions={}))

    assert captured["proposal_input"]["navigation_direction"] == "NO PATH"


def test_an_issue_with_no_location_is_not_routed(monkeypatch):
    """Routing from or to a room the game never named is meaningless (#7)."""
    captured = run_agent(monkeypatch, _context(), issue_location="Unknown")

    assert captured["proposal_input"]["navigation_direction"] == "NOT AVAILABLE"


def test_the_agent_makes_exactly_one_llm_call(monkeypatch):
    """#25: two calls per agent became one. The research round-trip is gone."""
    calls = []

    async def counting(chain, payload, operation_name="", **kwargs):
        calls.append(operation_name)
        return IssueProposal(proposed_action="SOUTH", reason="r", confidence=90)

    monkeypatch.setattr(llm_utils, "ainvoke_with_retry", counting)

    agent = make_agent()
    decision_llm = SimpleNamespace(
        with_structured_output=lambda schema: RunnableLambda(lambda _: None)
    )
    asyncio.run(agent.propose(decision_llm=decision_llm, context=_context()))

    assert len(calls) == 1
    assert "Proposal" in calls[0]


def test_the_agent_no_longer_has_a_research_phase():
    """Guard against reintroducing it: the whole point of #25 is that this
    data is deterministic and needs no model."""
    assert not hasattr(IssueAgent, "research_and_propose")
    assert hasattr(IssueAgent, "propose")

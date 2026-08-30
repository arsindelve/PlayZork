"""InteractionAgent must not inject garbage into the arbiter (#16).

`_deterministic_parse` ran BEFORE the LLM and short-circuited it on a match,
so on precisely the turns its regexes misfired, nothing could correct them —
and they misfired on Zork's most common reply. Every proposal below was
produced by executing the parser.

Two changes: the parser is now a HINT that never short-circuits, and the game's
own list of accepted commands (#30) replaces guessing at what is interactable.
"""
import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda

import llm_utils
from tools.agent_graph.interaction_agent import InteractionAgent
from tools.agent_graph.turn_context import TurnContext


def parse(text, inventory=None):
    result = InteractionAgent()._deterministic_parse(text, inventory or [])
    return result["action"] if result else None


# ---------------------------------------------------------------------------
# The reported garbage, verbatim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,was", [
    ("You see nothing special about the mailbox.", "TAKE NOTHING @ conf 90"),
    ("You notice nothing unusual here.", "TAKE NOTHING @ conf 90"),
    ("You closed the wooden door.", "OPEN YOU @ conf 85"),
    ("You cannot see any button here.", "PRESS BUTTON @ conf 80"),
])
def test_the_reported_fabrications_are_gone(text, was):
    assert parse(text) is None, f"still emitting garbage (previously {was})"


def test_zorks_commonest_reply_no_longer_fires():
    """"You see nothing special about X." is the standard EXAMINE response —
    this fired routinely, every session."""
    assert parse("You see nothing special about the brass lantern.") is None


@pytest.mark.parametrize("text,expected", [
    ("You see a brass lantern and a sword here.", "TAKE LANTERN"),  # was TAKE BRASS
    ("There is a small mailbox here.", "TAKE MAILBOX"),             # was a miss
    ("The wooden door is closed.", "OPEN DOOR"),
    ("The grating is locked.", "EXAMINE GRATING"),
])
def test_head_nouns_are_captured_not_adjectives(text, expected):
    assert parse(text) == expected


def test_a_negated_sentence_never_produces_an_action():
    """A negation describes what is NOT here; acting on it is guaranteed to
    fail."""
    for text in ["You cannot see any lever here.",
                 "There isn't anything here.",
                 "You don't see a button."]:
        assert parse(text) is None, text


def test_the_unlock_path_still_works_when_a_key_is_held():
    assert parse("The grating is locked.", ["rusty key"]) == "UNLOCK GRATING WITH KEY"


# ---------------------------------------------------------------------------
# The parser can no longer bypass the LLM
# ---------------------------------------------------------------------------


def _run(monkeypatch, context):
    captured = {}

    async def fake(chain, payload, operation_name="", **kw):
        captured["inputs"] = payload
        return SimpleNamespace(proposed_action="OPEN MAILBOX", reason="r", confidence=70,
                               detected_objects=["mailbox"], inventory_items=[])

    monkeypatch.setattr(llm_utils, "ainvoke_with_retry", fake)
    agent = InteractionAgent()
    llm = SimpleNamespace(with_structured_output=lambda s: RunnableLambda(lambda _: None))
    asyncio.run(agent.propose(decision_llm=llm, context=context))
    return agent, captured


def test_the_llm_runs_even_when_the_parser_matches(monkeypatch):
    """The short-circuit is the core defect: on a parser match the LLM never
    ran, so nothing could correct it."""
    ctx = TurnContext(location="West Of House", game_text="There is a small mailbox here.",
                      score=0, moves=1)

    agent, captured = _run(monkeypatch, ctx)

    assert "inputs" in captured, "the LLM phase was skipped"
    assert agent.proposed_action == "OPEN MAILBOX", "the parser overrode the LLM"


def test_the_hint_is_passed_but_labelled_unreliable(monkeypatch):
    ctx = TurnContext(location="West Of House", game_text="There is a small mailbox here.",
                      score=0, moves=1)

    _, captured = _run(monkeypatch, ctx)

    assert captured["inputs"]["parser_hint"] == "TAKE MAILBOX"


def test_the_llm_receives_the_games_own_action_list(monkeypatch):
    """#30: the backend reports exactly which commands it will accept, so the
    agent no longer has to infer what is interactable from prose."""
    ctx = TurnContext(location="Kitchen", game_text="You are in the kitchen.",
                      score=0, moves=6,
                      available_actions={"glass bottle": ["open bottle", "take bottle"]})

    _, captured = _run(monkeypatch, ctx)

    assert "open bottle" in captured["inputs"]["available_actions"]


def test_already_tried_commands_are_filtered_from_the_offered_actions():
    """#18 and #16 compose: an authoritative command that has already done
    nothing here should not be offered again."""
    ctx = TurnContext(location="Kitchen", game_text="", score=0, moves=6,
                      available_actions={"window": ["open window", "close window"]},
                      unproductive={"OPEN WINDOW": "The window is already open."})

    summary = ctx.available_actions_summary

    assert "close window" in summary
    assert "open window" not in summary


def test_a_backend_without_the_field_still_works(monkeypatch):
    """The local Escape Room backend does not report actions; the agent must
    fall back to prose, not crash."""
    ctx = TurnContext(location="Cell", game_text="There is a small mailbox here.",
                      score=0, moves=1)

    _, captured = _run(monkeypatch, ctx)

    assert "did not report" in captured["inputs"]["available_actions"]

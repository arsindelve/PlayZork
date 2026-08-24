"""Two live-run defects, both fixed in code rather than in prompt text.

Observed on session analysis-run-20260824:

  1. The InteractionAgent proposed `close grating` at confidence 90, reasoning
     "the game explicitly lists 'close grating' as a command it will accept
     here" — undoing the grating it had just opened. The accepted-command list
     is a GRAMMAR; the agent read it as advice.

  2. 55% of IssueAgent calls returned "nothing", at ~3000 tokens each — the
     most expensive call in the system — every one of them saying a variant of
     "the grating is locked and I have no key".
"""
from types import SimpleNamespace

import pytest

from tools.agent_graph.decision_graph import _blocked_signature, _format_agent_proposals
from tools.agent_graph.turn_context import TurnContext, inverse_of


# ---------------------------------------------------------------------------
# 1. Self-undoing proposals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command,undo", [
    ("OPEN GRATING", "CLOSE GRATING"),
    ("close grating", "OPEN GRATING"),
    ("TAKE LEAVES", "DROP LEAVES"),
    ("turn on lamp", "TURN OFF LAMP"),
    ("UNLOCK DOOR", "LOCK DOOR"),
])
def test_inverse_commands_are_recognised(command, undo):
    assert inverse_of(command) == undo


@pytest.mark.parametrize("command", ["NORTH", "EXAMINE SWORD", "LOOK", ""])
def test_commands_without_an_inverse_are_left_alone(command):
    assert inverse_of(command) == ""


def test_the_observed_close_grating_proposal_is_demoted():
    """The exact live failure."""
    context = TurnContext(location="Clearing", game_text="", score=0, moves=10,
                          succeeded={"OPEN GRATING": "The grating is locked."})
    interaction = SimpleNamespace(proposed_action="close grating", confidence=90,
                                  reason="the game lists it", detected_objects=[],
                                  inventory_items=[])

    text = _format_agent_proposals([], None, None, interaction, context=context)

    assert "WOULD UNDO 'OPEN GRATING'" in text


def test_an_action_we_have_not_taken_is_not_treated_as_an_undo():
    context = TurnContext(location="Clearing", game_text="", score=0, moves=10,
                          succeeded={"TAKE LEAVES": "revealed a grating"})

    assert context.undoes_recent_progress("close grating") == ""


def test_undo_detection_is_scoped_to_this_room():
    """Closing a door in another room is not undoing anything here."""
    context = TurnContext(location="Forest", game_text="", score=0, moves=10)

    assert context.undoes_recent_progress("close grating") == ""


def test_the_prompt_frames_the_action_list_as_grammar_not_advice():
    """The framing that caused it: 'those commands are guaranteed to parse',
    which the model read as endorsement."""
    from adventurer.prompt_library import PromptLibrary

    prompt = PromptLibrary.get_interaction_agent_human_prompt()

    assert "GRAMMAR, not advice" in prompt
    assert "Never undo your own progress" in prompt
    assert "guaranteed to parse" not in prompt


# ---------------------------------------------------------------------------
# 2. Not re-asking an agent that has already said it cannot act
# ---------------------------------------------------------------------------


def _memory(mem_id=1):
    return SimpleNamespace(id=mem_id, content="Grating at Clearing — unlock it",
                           location="Clearing", importance=600)


def test_the_verdict_depends_on_location_and_inventory():
    """Those are what the observed blocking reasons actually turn on."""
    memory = _memory()
    here = TurnContext(location="Clearing", game_text="", score=0, moves=1,
                       inventory=["leaflet"])
    same = TurnContext(location="clearing", game_text="different room text",
                       score=0, moves=9, inventory=["Leaflet"])

    assert _blocked_signature(memory, here) == _blocked_signature(memory, same), \
        "room text alone must not invalidate the verdict"


@pytest.mark.parametrize("changed", [
    TurnContext(location="Forest Path", game_text="", score=0, moves=1, inventory=["leaflet"]),
    TurnContext(location="Clearing", game_text="", score=0, moves=1, inventory=["leaflet", "key"]),
    TurnContext(location="Clearing", game_text="", score=0, moves=1, inventory=[]),
])
def test_a_change_of_location_or_inventory_invalidates_the_verdict(changed):
    """Finding the key, or moving somewhere new, must let the agent try again —
    otherwise a cached 'no' becomes permanent, which is the #11 mistake."""
    memory = _memory()
    original = TurnContext(location="Clearing", game_text="", score=0, moves=1,
                           inventory=["leaflet"])

    assert _blocked_signature(memory, original) != _blocked_signature(memory, changed)


def test_different_issues_have_different_signatures():
    context = TurnContext(location="Clearing", game_text="", score=0, moves=1)

    assert _blocked_signature(_memory(1), context) != _blocked_signature(_memory(2), context)

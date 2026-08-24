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


# ---------------------------------------------------------------------------
# 3. The explorer chooses on evidence, not a fixed cardinal order
# ---------------------------------------------------------------------------

ALL_DIRECTIONS = ["NORTH", "SOUTH", "EAST", "WEST", "NORTHEAST", "NORTHWEST",
                  "SOUTHEAST", "SOUTHWEST", "UP", "DOWN"]


def _explorer(mentioned=None, game_exits=None, unexplored=None):
    from tools.agent_graph.explorer_agent import ExplorerAgent

    return ExplorerAgent("Somewhere", unexplored or list(ALL_DIRECTIONS),
                         mentioned or [], 1, game_exits=game_exits)


def test_the_games_own_exit_list_beats_the_cardinal_default():
    """Up A Tree has one exit. The old fixed order picked NORTH, into nothing;
    over 26 turns that northward bias walked the agent into the forest and it
    never came back — and the explorer won 64% of contested turns, so its bias
    was effectively the agent's policy."""
    assert _explorer(game_exits=["DOWN"]).best_direction == "DOWN"


def test_a_mentioned_direction_confirmed_by_the_game_wins():
    assert _explorer(mentioned=["EAST"], game_exits=["EAST", "NORTH"]).best_direction == "EAST"


def test_a_mentioned_direction_still_counts_without_a_game_exit_list():
    """Backends that report no exits must keep the old description-based
    behaviour rather than degrading."""
    assert _explorer(mentioned=["SOUTH"]).best_direction == "SOUTH"


def test_a_direction_the_game_omits_is_ranked_down_not_banned():
    """The exits array is not a perfect oracle — North of House advertises an
    exit that is then refused — so it ranks candidates rather than
    restricting them."""
    explorer = _explorer(game_exits=["UP"], unexplored=["NORTH", "UP"])

    assert explorer.best_direction == "UP"
    assert "NORTH" in explorer.unexplored_directions, "must remain selectable"


def test_choice_is_deterministic_for_reproducible_runs():
    first = _explorer(game_exits=["EAST", "WEST"]).best_direction
    second = _explorer(game_exits=["WEST", "EAST"]).best_direction

    assert first == second


# ---------------------------------------------------------------------------
# 4. The rolling summaries are bounded
# ---------------------------------------------------------------------------


def test_summaries_are_truncated_not_merely_asked_to_be_short():
    """Both feed every agent prompt every turn, so their length is multiplied
    by the per-turn call count. Measured over 26 turns the recent summary grew
    140 -> 1334 chars despite covering a fixed window."""
    from config import LONG_SUMMARY_MAX_CHARS
    from tools.history.history_summarizer import _cap

    oversized = "CURRENT STATE:\nLocation: X\n" + "\n".join(
        f"- room {i}: notes" for i in range(300))

    capped = _cap(oversized, LONG_SUMMARY_MAX_CHARS, "long-running summary")

    assert len(capped) <= LONG_SUMMARY_MAX_CHARS + 40
    assert capped.startswith("CURRENT STATE:"), "the head carries current state"
    assert "truncated" in capped


def test_a_summary_within_budget_is_left_alone():
    from tools.history.history_summarizer import _cap

    text = "CURRENT STATE:\nLocation: West Of House"
    assert _cap(text, 2500, "x") == text


def test_the_long_summary_prompt_states_what_to_drop_first():
    """A budget without a priority order invites the model to drop whatever is
    convenient — including the unsolved puzzles that matter most."""
    from adventurer.prompt_library import PromptLibrary

    prompt = PromptLibrary.get_long_running_summary_system_prompt()

    flat = " ".join(prompt.split())
    assert "LENGTH BUDGET" in flat
    assert "NEVER drop current state, inventory, or unsolved puzzles" in flat
    assert "resolved puzzles first" in flat

"""The agent must not re-propose what the game already refused (#18).

Observed live, session m3-checkpoint-20260822, turns 11-15:

    turn 11  EXAMINE PILE OF LEAVES -> "There is nothing special about it."
    turn 12  NORTH                  -> "The forest becomes impenetrable..."
    turn 13  EXAMINE PILE OF LEAVES -> repeat
    turn 14  NORTH                  -> repeat
    turn 15  EXAMINE PILE OF LEAVES -> repeat

Both failures were sitting in the agent's own recent history. It alternated
them until the session was stopped.
"""
from types import SimpleNamespace

import pytest

from tools.agent_graph.decision_graph import _format_agent_proposals
from tools.agent_graph.turn_context import (
    TurnContext,
    build_turn_context,
    normalize_command,
)


def _turn(n, command, response, location, score=0):
    return SimpleNamespace(turn_number=n, player_command=command, game_response=response,
                           location=location, score=score, moves=n)


def _context(turns, location="Clearing"):
    history = SimpleNamespace(state=SimpleNamespace(
        get_full_summary=lambda: "", get_long_running_summary=lambda: "",
        # honour n, like the real implementation — the window is what makes
        # suppression temporary rather than permanent
        get_recent_turns=lambda n: turns[-n:]))
    mapper = SimpleNamespace(state=SimpleNamespace(get_exits_from=lambda l: []))
    inventory = SimpleNamespace(state=SimpleNamespace(get_items=lambda: []))
    return build_turn_context(
        game_response=SimpleNamespace(LocationName=location, Response="", Score=0, Moves=20),
        history_toolkit=history, mapper_toolkit=mapper, inventory_toolkit=inventory,
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_the_observed_deadlock_is_detected():
    ctx = _context([
        _turn(9, "NORTH", "Forest Path", "Forest Path"),
        _turn(10, "GO NORTH", "Clearing", "Clearing"),
        _turn(11, "EXAMINE PILE OF LEAVES", "There is nothing special about the pile of leaves.", "Clearing"),
        _turn(12, "NORTH", "The forest becomes impenetrable to the north.", "Clearing"),
    ])

    assert ctx.is_unproductive("EXAMINE PILE OF LEAVES")
    assert ctx.is_unproductive("NORTH")


def test_the_command_that_moved_us_here_is_not_suppressed():
    """It changed the location, so it plainly did something — and the first
    turn in the window has no predecessor to prove otherwise."""
    ctx = _context([
        _turn(10, "GO NORTH", "Clearing", "Clearing"),
        _turn(11, "EXAMINE PILE OF LEAVES", "Nothing special.", "Clearing"),
    ])

    assert not ctx.is_unproductive("GO NORTH")


def test_a_scoring_command_is_never_suppressed():
    ctx = _context([
        _turn(10, "LOOK", "Clearing", "Clearing", score=0),
        _turn(11, "TAKE EGG", "Taken.", "Clearing", score=5),
    ])

    assert not ctx.is_unproductive("TAKE EGG")


def test_suppression_is_scoped_to_the_room():
    """"OPEN DOOR" failing in the Kitchen says nothing about the Cellar."""
    ctx = _context([
        _turn(10, "LOOK", "Kitchen", "Kitchen"),
        _turn(11, "OPEN DOOR", "The door is locked.", "Kitchen"),
    ], location="Cellar")

    assert not ctx.is_unproductive("OPEN DOOR")


def test_matching_is_case_and_space_insensitive_but_not_semantic():
    ctx = _context([
        _turn(10, "LOOK", "Clearing", "Clearing"),
        _turn(11, "EXAMINE  PILE OF LEAVES", "Nothing special.", "Clearing"),
    ])

    assert ctx.is_unproductive("examine pile of leaves")
    # Deliberately NOT semantic: a near-miss must not suppress a real attempt.
    assert not ctx.is_unproductive("EXAMINE THE PILE OF LEAVES")
    assert not ctx.is_unproductive("TAKE PILE OF LEAVES")


def test_suppression_ages_out_of_the_window():
    """The world changes — a door gets unlocked, a lamp gets lit. Permanent
    suppression would repeat #11's mistake of making an inference
    unfalsifiable."""
    old = [_turn(1, "LOOK", "Clearing", "Clearing"),
           _turn(2, "OPEN GRATING", "The grating is locked.", "Clearing")]
    recent = [_turn(n, "LOOK", "Clearing", "Clearing") for n in range(3, 40)]
    ctx = _context(old + recent)

    assert not ctx.is_unproductive("OPEN GRATING")


def test_normalize_command_is_conservative():
    assert normalize_command("  examine   mailbox ") == "EXAMINE MAILBOX"
    assert normalize_command(None) == ""


# ---------------------------------------------------------------------------
# The arbiter actually sees it
# ---------------------------------------------------------------------------


def _proposals(ctx):
    issue = SimpleNamespace(proposed_action="EXAMINE PILE OF LEAVES", confidence=85,
                            importance=500, issue_content="Pile of leaves", reason="investigate")
    explorer = SimpleNamespace(proposed_action="NORTH", confidence=95, best_direction="NORTH",
                               unexplored_directions=["NORTH", "SOUTH"], reason="unexplored")
    return _format_agent_proposals([issue], explorer, None, None, context=ctx)


def test_repeated_proposals_reach_the_arbiter_at_zero_expected_value():
    """Done in code, not by prompt instruction: the #21 inventory bug showed a
    14B model given a bare prohibition will invent its own way around it."""
    ctx = TurnContext(location="Clearing", game_text="", score=0, moves=13,
                      unproductive={"EXAMINE PILE OF LEAVES": "Nothing special.",
                                    "NORTH": "The forest becomes impenetrable."})

    text = _proposals(ctx)

    assert "EV: 0.0" in text
    assert text.count("EV: 0.0") == 2, "both deadlocked proposals must be demoted"
    assert "ALREADY TRIED HERE" in text


def test_the_arbiter_is_told_why_not_merely_that():
    ctx = TurnContext(location="Clearing", game_text="", score=0, moves=13,
                      unproductive={"NORTH": "The forest becomes impenetrable to the north."})

    text = _proposals(ctx)

    assert "impenetrable" in text


def test_fresh_proposals_keep_their_expected_value():
    ctx = TurnContext(location="Clearing", game_text="", score=0, moves=13)

    text = _proposals(ctx)

    assert "ALREADY TRIED HERE" not in text
    assert "EV: 0.0" not in text


def test_the_prompt_block_lists_what_is_dead_and_why():
    ctx = TurnContext(location="Clearing", game_text="", score=0, moves=13,
                      unproductive={"NORTH": "The forest becomes impenetrable to the north."})

    block = ctx.research_context_for()

    assert "ALREADY TRIED HERE" in block
    assert "impenetrable" in block


def test_an_empty_history_reads_as_nothing_tried():
    ctx = TurnContext(location="Clearing", game_text="", score=0, moves=1)

    assert ctx.unproductive_summary == "None yet."

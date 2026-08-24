"""Withholding zeroed proposals, and why annotation alone was not enough.

`_format_agent_proposals` has always said a demoted proposal is annotated
rather than removed "so the arbiter can still pick it if literally everything
else is exhausted". Nothing enforced the condition. In pf3-20260824 three undo
demotions fired and TWO were chosen anyway:

    Chose ExplorerAgent (confidence 95, EV 0.0) despite the low EV because the
    current location is not advancing the score

— with a 70-EV proposal unchosen on the same ballot. Milestone 5b's urgency
signal drove it: told the score has not moved, the arbiter wants any change of
direction, and the only movement on offer is the way it just came.
"""
from types import SimpleNamespace as NS

import pytest

from tools.agent_graph.decision_graph import _format_agent_proposals
from tools.agent_graph.turn_context import TurnContext

POD = {"escape pod bulkhead": ["open bulkhead", "close bulkhead"],
       "ensign blather": ["examine blather"]}


def explorer(action="GO WEST", confidence=95, unexplored=10):
    return NS(proposed_action=action, confidence=confidence, best_direction=action,
              unexplored_directions=["x"] * unexplored, reason="r")


def interaction(action="examine blather", confidence=70):
    return NS(proposed_action=action, confidence=confidence, reason="r",
              detected_objects=[], inventory_items=[])


def issue(action="OPEN bulkhead", confidence=80, importance=900):
    return NS(proposed_action=action, confidence=confidence, importance=importance,
              issue_content="pod", reason="r")


class TestTheTurnFourBallot:
    """Reactor Lobby: EAST had just moved us here, so GO WEST reverses it."""

    def _ballot(self):
        ctx = TurnContext(location="Reactor Lobby", game_text="", score=0, moves=4,
                          available_actions=POD,
                          succeeded={"EAST": "Reactor Lobby"})
        return _format_agent_proposals([], explorer("GO WEST"), None,
                                       interaction(), ctx)

    def test_the_undo_is_withheld(self):
        assert "ExplorerAgent" not in self._ballot()

    def test_the_live_alternative_remains(self):
        text = self._ballot()
        assert "InteractionAgent" in text
        assert "examine blather" in text

    def test_the_arbiter_can_no_longer_pick_the_undo(self):
        """It literally is not on the ballot, so 'despite the low EV' has
        nothing to attach to."""
        assert "EV: 0.0" not in self._ballot()


class TestExhaustionKeepsEverything:
    def test_all_dead_means_all_shown(self):
        ctx = TurnContext(location="Clearing", game_text="", score=0, moves=9,
                          # Keyed by NORMALIZED command: fix 6 collapses
                          # "GO WEST" to "WEST", and the real _unproductive()
                          # builds its keys the same way.
                          unproductive={"WEST": "You can't go that way.",
                                        "EXAMINE BLATHER": "He ignores you."})
        text = _format_agent_proposals([], explorer("GO WEST"), None,
                                       interaction("EXAMINE BLATHER"), ctx)
        assert "ExplorerAgent" in text and "InteractionAgent" in text
        assert text.count("EV: 0.0") == 2

    def test_the_annotations_survive_in_that_case(self):
        ctx = TurnContext(location="Clearing", game_text="", score=0, moves=9,
                          unproductive={"WEST": "You can't go that way."})
        text = _format_agent_proposals([], explorer("GO WEST"), None, None, ctx)
        assert "ALREADY TRIED HERE" in text
        assert "can't go that way" in text

    def test_no_proposals_at_all_is_unchanged(self):
        ctx = TurnContext(location="Clearing", game_text="", score=0, moves=9)
        assert "No proposals available" in _format_agent_proposals(
            [], None, None, None, ctx)


class TestOnlyZeroedBlocksAreWithheld:
    def test_a_positive_explorer_is_kept(self):
        ctx = TurnContext(location="Clearing", game_text="", score=0, moves=9,
                          available_actions=POD)
        text = _format_agent_proposals([], explorer("NORTH"), None, interaction(), ctx)
        assert "ExplorerAgent" in text and "InteractionAgent" in text

    def test_a_zeroed_issue_is_withheld_too(self):
        """Not explorer-specific — any agent's zeroed proposal goes."""
        ctx = TurnContext(location="Clearing", game_text="", score=0, moves=9,
                          available_actions=POD,
                          unproductive={"OPEN BULKHEAD": "Why open it?"})
        text = _format_agent_proposals([issue()], None, None, interaction(), ctx)
        assert "IssueAgent" not in text
        assert "InteractionAgent" in text

    def test_a_zeroed_interaction_is_withheld_too(self):
        ctx = TurnContext(location="Clearing", game_text="", score=0, moves=9,
                          available_actions=POD,
                          unproductive={"EXAMINE BLATHER": "He ignores you."})
        text = _format_agent_proposals([], explorer("NORTH"), None,
                                       interaction("EXAMINE BLATHER"), ctx)
        assert "InteractionAgent" not in text
        assert "ExplorerAgent" in text

    def test_no_context_disables_demotion_entirely(self):
        """With no context there is nothing to demote against, so every
        proposal must survive — never withhold on missing information."""
        text = _format_agent_proposals([], explorer("GO WEST"), None,
                                       interaction(), None)
        assert "ExplorerAgent" in text and "InteractionAgent" in text

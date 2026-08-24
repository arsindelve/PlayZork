"""The InteractionAgent's expected value, and the run that proved it was needed.

In pf-20260824 the game reported the escape pod bulkhead in the STARTING ROOM,
the observer stored it at importance 900, and the InteractionAgent proposed
"OPEN escape pod bulkhead" at confidence 70 on turn 2. The arbiter chose GO UP,
because the ExplorerAgent had EV 47.5 and the InteractionAgent had no expected
value at all — while the decision prompt ranks by expected value.
"""
from types import SimpleNamespace as NS

import pytest

from tools.agent_graph.decision_graph import _format_agent_proposals
from tools.agent_graph.turn_context import TurnContext

BULKHEAD = {"escape pod bulkhead": ["open bulkhead", "close bulkhead"]}


def ctx(**kw):
    base = dict(location="Deck Nine", game_text="Deck Nine", score=0, moves=1,
                available_actions=BULKHEAD)
    base.update(kw)
    return TurnContext(**base)


def interaction(action="OPEN escape pod bulkhead", confidence=70):
    return NS(confidence=confidence, proposed_action=action, reason="r",
              detected_objects=["escape pod bulkhead"], inventory_items=[])


def explorer(action="GO UP", confidence=95, unexplored=10):
    return NS(confidence=confidence, proposed_action=action, reason="r",
              best_direction="UP", unexplored_directions=["X"] * unexplored)


def ev_of(text, agent):
    """Pull the EV the arbiter actually sees for `agent`."""
    for line in text.splitlines():
        if line.startswith(f"{agent}:"):
            return float(line.split("EV: ")[1].split(",")[0].rstrip("]"))
    raise AssertionError(f"{agent} has no EV in:\n{text}")


class TestTheEscapePodTurn:
    def test_interaction_now_has_an_ev_at_all(self):
        out = _format_agent_proposals([], explorer(), None, interaction(), ctx())
        assert "InteractionAgent: [Confidence: 70/100, EV:" in out

    def test_the_pod_now_outranks_the_gangway(self):
        """The exact turn-2 matchup that lost."""
        out = _format_agent_proposals([], explorer(), None, interaction(), ctx())
        pod = ev_of(out, "InteractionAgent")
        gangway = ev_of(out, "ExplorerAgent")
        assert gangway == pytest.approx(47.5), "explorer EV must be unchanged"
        assert pod > gangway, f"pod {pod} must beat gangway {gangway}"

    def test_backend_confirmation_is_what_earns_the_higher_base(self):
        confirmed = ev_of(_format_agent_proposals(
            [], None, None, interaction(), ctx()), "InteractionAgent")
        invented = ev_of(_format_agent_proposals(
            [], None, None, interaction("POLISH the bulkhead"), ctx()),
            "InteractionAgent")
        assert confirmed == pytest.approx(70.0)   # 0.70 * 100
        assert invented == pytest.approx(35.0)    # 0.70 * 50
        assert invented < 47.5, "an invented interaction must NOT beat exploration"

    def test_the_evidence_label_is_shown(self):
        out = _format_agent_proposals([], None, None, interaction(), ctx())
        assert "game-confirmed" in out
        out = _format_agent_proposals([], None, None,
                                      interaction("POLISH the bulkhead"), ctx())
        assert "model-proposed" in out


class TestMultiplierNowApplies:
    """`note, _ = repeat_note(...)` discarded the multiplier, so #18's demotion
    was a cosmetic warning line for this agent rather than a mechanism — the
    project's own "prompt text is not a mechanism" failure."""

    def test_repeat_is_zeroed_not_merely_annotated(self):
        c = ctx(unproductive={"OPEN ESCAPE POD BULKHEAD": "Nothing happens."})
        out = _format_agent_proposals([], None, None, interaction(), c)
        assert "ALREADY TRIED HERE" in out
        assert ev_of(out, "InteractionAgent") == pytest.approx(0.0)

    def test_undo_is_zeroed(self):
        c = ctx(succeeded={"OPEN BULKHEAD": "The bulkhead opens."})
        out = _format_agent_proposals([], None, None,
                                      interaction("CLOSE bulkhead"), c)
        assert "WOULD UNDO" in out
        assert ev_of(out, "InteractionAgent") == pytest.approx(0.0)

    def test_a_clean_proposal_keeps_its_ev(self):
        out = _format_agent_proposals([], None, None, interaction(), ctx())
        assert ev_of(out, "InteractionAgent") > 0


class TestOtherAgentsUnchanged:
    """The fix must not quietly re-rank the agents that already worked."""

    def test_explorer_formula_untouched(self):
        out = _format_agent_proposals([], explorer(), None, None, ctx())
        assert ev_of(out, "ExplorerAgent") == pytest.approx(47.5)

    def test_issue_formula_untouched(self):
        issue = NS(importance=900, confidence=80, proposed_action="OPEN bulkhead",
                   issue_content="pod", reason="r")
        out = _format_agent_proposals([issue], None, None, None, ctx())
        assert ev_of(out, "IssueAgent #1") == pytest.approx(72.0)  # .9*.8*100

    def test_no_context_does_not_crash(self):
        out = _format_agent_proposals([], explorer(), None, interaction(), None)
        assert "InteractionAgent" in out

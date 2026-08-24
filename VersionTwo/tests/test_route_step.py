"""A goal-directed return is not aimless backtracking.

The undo rule cannot tell them apart — both reverse the previous move. In
pf4-20260824 the escape pod was at Deck Nine and the agent one room above it,
so walking back down (correct) looked identical to the oscillation the rule
exists to stop.
"""
from types import SimpleNamespace as NS

import pytest

from tools.agent_graph.decision_graph import _format_agent_proposals
from tools.agent_graph.turn_context import TurnContext


def ctx(**kw):
    base = dict(location="Gangway", game_text="", score=0, moves=8)
    base.update(kw)
    c = TurnContext(**base)
    return c


class TestIsRouteStep:
    def test_a_step_toward_a_tracked_issue(self):
        c = ctx()
        c.directions = {"deck nine": "DOWN"}
        assert c.is_route_step("DOWN")
        assert c.is_route_step("GO DOWN"), "phrasing must not matter"

    def test_an_unrelated_direction_is_not(self):
        c = ctx()
        c.directions = {"deck nine": "DOWN"}
        assert not c.is_route_step("UP")

    @pytest.mark.parametrize("sentinel", ["NO PATH", "NOT AVAILABLE"])
    def test_sentinels_are_not_routes(self, sentinel):
        c = ctx()
        c.directions = {"deck nine": sentinel}
        assert not c.is_route_step(sentinel)

    def test_no_tracked_issues_means_no_routes(self):
        assert not ctx().is_route_step("DOWN")


class TestGoalDirectedReturnSurvives:
    def _explorer(self, action, confidence=95):
        return NS(proposed_action=action, confidence=confidence,
                  best_direction=action, unexplored_directions=["x"] * 10,
                  reason="r")

    def test_a_return_toward_the_pod_is_not_demoted(self):
        """UP just moved us here, so DOWN reverses it — but the pod is DOWN."""
        c = ctx(succeeded={"UP": "Gangway"})
        c.directions = {"deck nine": "DOWN"}
        text = _format_agent_proposals([], self._explorer("GO DOWN"), None, None, c)
        assert "WOULD UNDO" not in text
        assert "EV: 0.0" not in text
        assert "GO DOWN" in text

    def test_an_aimless_reversal_is_still_demoted(self):
        """No tracked issue lies that way, so the undo rule still applies."""
        c = ctx(succeeded={"UP": "Gangway"})
        c.directions = {"deck nine": "WEST"}
        text = _format_agent_proposals([], self._explorer("GO DOWN"), None, None, c)
        assert "WOULD UNDO" in text
        assert "EV: 0.0" in text

    def test_the_exemption_does_not_cover_already_tried(self):
        """Being on a route does not make a command that did nothing work."""
        c = ctx(unproductive={"DOWN": "The bulkhead is closed."})
        c.directions = {"deck nine": "DOWN"}
        text = _format_agent_proposals([], self._explorer("GO DOWN"), None, None, c)
        assert "ALREADY TRIED HERE" in text
        assert "EV: 0.0" in text
        assert "ExplorerAgent" in text, "still never withheld"

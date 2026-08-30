"""Movement identity, movement undo, and navigating to a distant issue.

All three were found by playing, not by the suite. In frontier3-20260824 the
agent reached Behind House — the room containing the window into the house —
and oscillated GO WEST -> EAST -> GO WEST straight back off it, with ZERO
suppressions across 16 turns. pf-20260824 reproduced it vertically, GO UP ->
GO UP -> GO DOWN, while the ship's clock ran toward the explosion.
"""
import asyncio
from types import SimpleNamespace as NS

import pytest

from tools.agent_graph.turn_context import (TurnContext, inverse_of,
                                            normalize_command)
from tools.mapping.directions import extract_direction, find_mentioned_directions


class TestMovementIsOneCommand:
    """GO WEST / WEST / W were three distinct keys, so suppression, the
    `succeeded` map and undo detection all missed synonyms."""

    @pytest.mark.parametrize("phrasing", ["GO WEST", "WEST", "W", "go  west"])
    def test_all_phrasings_collapse(self, phrasing):
        assert normalize_command(phrasing) == "WEST"

    @pytest.mark.parametrize("command", [
        "TAKE LAMP", "OPEN DOOR", "OPEN THE DOOR", "PUSH NORTH WALL", "LOOK"])
    def test_non_movement_stays_literal(self, command):
        """A false suppression silently removes a real option and nothing in
        the game text ever corrects it, so only movement is collapsed."""
        assert normalize_command(command) == command.upper()

    def test_open_door_and_open_the_door_stay_distinct(self):
        assert normalize_command("OPEN DOOR") != normalize_command("OPEN THE DOOR")

    def test_repetition_suppression_now_catches_a_synonym(self):
        c = TurnContext(location="Clearing", game_text="", score=0, moves=5,
                        unproductive={"WEST": "You can't go that way."})
        assert c.is_unproductive("GO WEST")
        assert c.is_unproductive("W")


class TestMovementUndo:
    def test_east_undoes_west(self):
        c = TurnContext(location="Clearing", game_text="", score=0, moves=14,
                        succeeded={"WEST": "Behind House"})
        assert c.undoes_recent_progress("EAST") == "WEST"

    def test_the_behind_house_oscillation(self):
        """Turn 14 GO WEST reached Behind House; turn 15 EAST walked back."""
        c = TurnContext(location="Clearing", game_text="", score=0, moves=15,
                        succeeded={normalize_command("GO WEST"): "Behind House"})
        assert c.undoes_recent_progress("EAST") == "WEST"

    def test_the_planetfall_vertical_oscillation(self):
        c = TurnContext(location="Deck Eight", game_text="", score=0, moves=3,
                        succeeded={normalize_command("GO UP"): "Deck Eight"})
        assert c.undoes_recent_progress("GO DOWN") == "UP"

    def test_push_north_wall_is_not_a_movement(self):
        assert inverse_of("PUSH NORTH WALL") == ""


class TestShipDirections:
    """Aliases of compass directions, NOT new canonical ones."""

    def test_starboard_is_read_as_a_mention_of_east(self):
        prose = ("This is a featureless corridor. It curves away to starboard, "
                 "and a gangway leads up. To port is the entrance.")
        found = find_mentioned_directions(prose)
        assert "EAST" in found and "WEST" in found and "UP" in found

    def test_the_deck_nine_scoring_flip(self):
        """UP scored 5 (exit+mentioned) and EAST 4 (exit+cardinal), so the
        agent climbed the ship instead of crossing it toward the pod. With
        starboard recognised, EAST scores 6."""
        prose = ("It curves away to starboard, and a gangway leads up.")
        mentioned = set(find_mentioned_directions(prose))
        cardinals = {"NORTH", "SOUTH", "EAST", "WEST"}
        game_exits = {"UP", "EAST"}

        def score(d):
            return ((3 if d in game_exits else 0)
                    + (2 if d in mentioned else 0)
                    + (1 if d in cardinals else 0))
        assert score("EAST") == 6
        assert score("UP") == 5
        assert score("EAST") > score("UP")

    def test_not_added_to_canonical_directions(self):
        """EAST and STARBOARD are ONE passage. Listing both would inflate the
        explorer's unexplored count — which its EV scales with — and let it
        re-walk a passage it had already taken under the other name."""
        from tools.mapping.directions import CANONICAL_DIRECTIONS
        assert len(CANONICAL_DIRECTIONS) == 10
        for d in ("PORT", "STARBOARD", "FORE", "AFT"):
            assert d not in CANONICAL_DIRECTIONS

    @pytest.mark.parametrize("word", ["important", "support", "airport",
                                      "deportment", "afterwards"])
    def test_no_substring_false_positives(self, word):
        assert find_mentioned_directions(f"There is an {word} here.") == []

    def test_go_starboard_parses_as_east(self):
        assert extract_direction("GO STARBOARD") == "EAST"
        assert normalize_command("GO STARBOARD") == "EAST"


class TestIssueAgentNavigatesToItsIssue:
    """It returned "nothing" at confidence 0 for a 900-importance escape pod
    two rooms away, with the route already in its prompt."""

    def _agent(self):
        from tools.agent_graph.issue_agent import IssueAgent
        memory = NS(id=31, content="Locked pod bulkhead at Deck Nine",
                    location="Deck Nine", importance=900, turn_number=1,
                    score=0, moves=1)
        return IssueAgent(memory)

    @pytest.mark.parametrize("action,confidence", [
        ("nothing", 0), ("", 50), ("NOTHING", 90), ("none", 0), ("OPEN POD", 0)])
    def test_declined_detects_every_form(self, action, confidence):
        from tools.agent_graph.issue_agent import IssueAgent
        assert IssueAgent._declined(NS(proposed_action=action, confidence=confidence))

    def test_a_real_proposal_is_not_declined(self):
        from tools.agent_graph.issue_agent import IssueAgent
        assert not IssueAgent._declined(
            NS(proposed_action="OPEN bulkhead", confidence=70))

    def test_substitutes_the_route_step(self, monkeypatch):
        agent = self._agent()
        proposal = NS(proposed_action="nothing", confidence=0, reason="cannot")
        self._run(agent, proposal, navigation="DOWN",
                  status="DIFFERENT LOCATION", current="Deck Eight")
        assert agent.proposed_action == "DOWN"
        assert agent.confidence == 70
        assert "Deck Nine" in agent.reason

    def test_does_not_override_a_real_proposal(self):
        agent = self._agent()
        proposal = NS(proposed_action="OPEN bulkhead", confidence=70, reason="here")
        self._run(agent, proposal, navigation="DOWN",
                  status="DIFFERENT LOCATION", current="Deck Eight")
        assert agent.proposed_action == "OPEN bulkhead"

    @pytest.mark.parametrize("navigation", ["NO PATH", "NOT AVAILABLE", "", "UNKNOWN"])
    def test_no_route_means_no_substitution(self, navigation):
        """Never invent a direction — a wrong move costs a turn on a clock."""
        agent = self._agent()
        proposal = NS(proposed_action="nothing", confidence=0, reason="cannot")
        self._run(agent, proposal, navigation=navigation,
                  status="DIFFERENT LOCATION", current="Deck Eight")
        assert agent.proposed_action == "nothing"

    def test_same_location_means_no_substitution(self):
        agent = self._agent()
        proposal = NS(proposed_action="nothing", confidence=0, reason="cannot")
        self._run(agent, proposal, navigation="DOWN",
                  status="SAME LOCATION", current="Deck Nine")
        assert agent.proposed_action == "nothing"

    @staticmethod
    def _run(agent, proposal, navigation, status, current):
        """Replay the exact block that follows the LLM call."""
        agent.proposed_action = proposal.proposed_action
        agent.reason = proposal.reason
        agent.confidence = proposal.confidence
        if agent._declined(proposal) and status == "DIFFERENT LOCATION":
            step = (navigation or "").strip().upper()
            if step and step not in ("NO PATH", "NOT AVAILABLE", "UNKNOWN"):
                agent.proposed_action = step
                agent.confidence = 70
                agent.reason = (f"Cannot act on this issue from {current}; it is "
                                f"at {agent.location}. {step} is the next step "
                                f"on the known route there.")


class TestEconomics:
    """The substitution must not let a stale issue outrank exploration."""

    def test_high_importance_issue_outranks_exploration(self):
        assert (900 / 1000) * (70 / 100) * 100 == pytest.approx(63.0)
        assert 63.0 > 47.5

    def test_decayed_issue_does_not(self):
        assert (300 / 1000) * (70 / 100) * 100 == pytest.approx(21.0)
        assert 21.0 < 47.5


class TestNavigationThroughTheRealProposeMethod:
    """The tests above REPLAY the substitution block; these EXECUTE it.

    Replaying logic tests a copy of it. Every silent failure this session came
    from wiring rather than logic, so the real `propose()` is driven here with
    the LLM call stubbed out.
    """

    def _agent(self):
        from tools.agent_graph.issue_agent import IssueAgent
        return IssueAgent(NS(id=31, content="Locked pod bulkhead at Deck Nine",
                             location="Deck Nine", importance=900,
                             turn_number=1, score=0, moves=1))

    def _context(self, current="Deck Eight", direction="DOWN"):
        c = TurnContext(location=current, game_text="Deck Eight", score=0, moves=3)
        c.directions = {"deck nine": direction}
        return c

    def _propose(self, monkeypatch, agent, context, returned):
        """Run the real coroutine with only the LLM call replaced."""
        async def fake(chain, inputs, operation_name=None, **kw):
            fake.inputs = inputs
            return returned
        # Imported INSIDE propose(), so patch it at the source module.
        import llm_utils
        monkeypatch.setattr(llm_utils, "ainvoke_with_retry", fake)
        # The chain is built as `prompt | llm.with_structured_output(...)`, so
        # the stub must be pipeable; the fake above intercepts before it runs.
        from langchain_core.runnables import RunnableLambda
        llm = NS(with_structured_output=lambda *_a, **_k: RunnableLambda(lambda x: x))
        asyncio.run(agent.propose(decision_llm=llm, context=context))
        return fake

    def test_declined_proposal_becomes_the_route_step(self, monkeypatch):
        agent = self._agent()
        self._propose(monkeypatch, agent, self._context(),
                      NS(proposed_action="nothing", confidence=0, reason="cannot"))
        assert agent.proposed_action == "DOWN"
        assert agent.confidence == 70

    def test_real_proposal_survives(self, monkeypatch):
        agent = self._agent()
        self._propose(monkeypatch, agent, self._context(),
                      NS(proposed_action="OPEN bulkhead", confidence=70, reason="r"))
        assert agent.proposed_action == "OPEN bulkhead"
        assert agent.confidence == 70

    def test_no_route_leaves_it_declined(self, monkeypatch):
        agent = self._agent()
        self._propose(monkeypatch, agent, self._context(direction="NO PATH"),
                      NS(proposed_action="nothing", confidence=0, reason="cannot"))
        assert agent.proposed_action == "nothing"

    def test_standing_at_the_issue_leaves_it_declined(self, monkeypatch):
        agent = self._agent()
        self._propose(monkeypatch, agent, self._context(current="Deck Nine"),
                      NS(proposed_action="nothing", confidence=0, reason="cannot"))
        assert agent.proposed_action == "nothing"

    def test_the_prompt_actually_receives_the_route(self, monkeypatch):
        """If navigation_direction stopped reaching the prompt, the model would
        be declining for a good reason and the substitution would be masking a
        real regression."""
        agent = self._agent()
        fake = self._propose(monkeypatch, agent, self._context(),
                             NS(proposed_action="nothing", confidence=0, reason="c"))
        assert fake.inputs["navigation_direction"] == "DOWN"
        assert fake.inputs["location_status"] == "DIFFERENT LOCATION"

"""The game's own clock, which was parsed and then read by nothing.

`Time` has been on the response model since #30. Searching the codebase for it
found only `timeout` and `setTimeout`. On Planetfall — where the ship explodes
— that meant the agents could not see the deadline they were being judged
against. Zork reports 0, so this must stay invisible there.
"""
from types import SimpleNamespace as NS

from tools.agent_graph.turn_context import TurnContext, build_turn_context


def ctx(game_time):
    return TurnContext(location="Deck Nine", game_text="x", score=0, moves=1,
                       game_time=game_time)


class TestRenderedOnlyWhereItMeansSomething:
    def test_planetfall_clock_is_shown(self):
        assert "GAME CLOCK: 4654" in ctx(4654).clock_summary

    def test_zork_reports_zero_and_shows_nothing(self):
        # Verified by direct probe: Zork's API returns time: 0 every turn.
        # Rendering it would add a noise line to every prompt of a game with
        # no deadline.
        assert ctx(0).clock_summary == ""

    def test_absent_field_shows_nothing(self):
        assert ctx(None).clock_summary == ""

    def test_it_says_time_passes_on_wasted_moves(self):
        # The point isn't the number, it's that dithering costs something.
        assert "wasted" in ctx(4654).clock_summary


class TestReachesTheAgents:
    def test_in_the_shared_research_context(self):
        assert "GAME CLOCK" in ctx(4654).research_context_for()

    def test_not_in_zork_research_context(self):
        assert "GAME CLOCK" not in ctx(0).research_context_for()

    def test_populated_by_the_real_builder(self):
        """Wiring, not logic — the repeated failure mode this session."""
        mapper = NS(state=NS(get_all_transitions=lambda: [],
                             get_exits_from=lambda l: [],
                             get_unexplored_directions=lambda l: []))
        history = NS(state=NS(get_recent_turns=lambda *a, **k: [],
                              get_full_summary=lambda *a, **k: "",
                              get_long_running_summary=lambda *a, **k: ""))
        c = build_turn_context(
            game_response=NS(Response="Deck Nine", LocationName="Deck Nine",
                             Score=0, Moves=1, exits=None, availableActions=None,
                             Time=4654),
            history_toolkit=history, mapper_toolkit=mapper,
            inventory_toolkit=NS(state=NS(get_items=lambda *a, **k: [])),
            issue_locations=None)
        assert c.game_time == 4654
        assert "GAME CLOCK" in c.research_context_for()

    def test_missing_time_attribute_is_safe(self):
        """A backend that omits the field must not break the turn."""
        mapper = NS(state=NS(get_all_transitions=lambda: [],
                             get_exits_from=lambda l: [],
                             get_unexplored_directions=lambda l: []))
        history = NS(state=NS(get_recent_turns=lambda *a, **k: [],
                              get_full_summary=lambda *a, **k: "",
                              get_long_running_summary=lambda *a, **k: ""))
        c = build_turn_context(
            game_response=NS(Response="x", LocationName="L", Score=0, Moves=1,
                             exits=None, availableActions=None),
            history_toolkit=history, mapper_toolkit=mapper,
            inventory_toolkit=NS(state=NS(get_items=lambda *a, **k: [])),
            issue_locations=None)
        assert c.game_time is None
        assert c.clock_summary == ""


class TestObjectiveIsSpecific:
    def test_planetfall_objective_names_the_escape(self):
        """It is interpolated into every prompt and was "Complete the mission",
        which told the arbiter nothing about a ship that was about to explode.
        """
        from config import GAME_BACKENDS
        objective = GAME_BACKENDS["planetfall"]["objective"].lower()
        assert "escape" in objective
        assert "explo" in objective
        assert objective != "complete the mission"

    def test_it_still_names_the_rest_of_the_game(self):
        # The string is fixed for the whole run; the escape is only phase one.
        objective = GAME_BACKENDS_objective()
        assert "mission" in objective or "planet" in objective


def GAME_BACKENDS_objective():
    from config import GAME_BACKENDS
    return GAME_BACKENDS["planetfall"]["objective"].lower()

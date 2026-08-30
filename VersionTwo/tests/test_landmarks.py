"""The frontier's second source: landmarks named in room text.

The map-only frontier measured EMPTY on a live run (see PLAN.md, 2026-08-24):
on a linear path every room reached was also departed from. These tests pin
the replacement, and especially the cases where the obvious rule is wrong.
"""
import pytest

from tools.mapping.landmarks import (LANDMARK_NOUNS, find_landmarks,
                                     unvisited_landmarks)

ZORK_OPENING = ("You are standing in an open field west of a white house, "
                "with a boarded front door. There is a small mailbox here.")
ZORK_KITCHEN = ("Kitchen. You are in the kitchen of the white house. A table "
                "seems to have been used recently. A dark chimney leads down.")


class TestExtraction:
    def test_finds_the_landmark_that_matters(self):
        assert "white house" in find_landmarks(ZORK_OPENING)

    def test_keeps_adjectives_that_identify_it(self):
        # "house" alone is uselessly generic once there are two of them.
        assert "white house" in find_landmarks(ZORK_OPENING)
        assert "house" not in find_landmarks(ZORK_OPENING)

    def test_strips_prepositions_leaking_from_the_sentence(self):
        # "west of a white house" must not yield "of a white house".
        for phrase in find_landmarks(ZORK_OPENING):
            assert not phrase.startswith(("of ", "in ", "the ", "a "))

    def test_ignores_scenery(self):
        # Terrain is the explorer's job, by direction. Listing it would drown
        # the real leads.
        assert find_landmarks("This is a forest, with trees in all directions.") == []

    def test_quantifiers_are_lazy(self):
        # A greedy quantifier once ran a phrase straight past its noun.
        for phrase in find_landmarks(ZORK_OPENING):
            assert phrase.split()[-1] in LANDMARK_NOUNS
            assert len(phrase.split()) <= 3

    def test_never_raises_on_junk(self):
        for junk in (None, "", "((((", "a " * 500, "\x00door"):
            assert isinstance(find_landmarks(junk), list)


class TestVisitedRetirement:
    """Every name-matching rule fails somewhere. These pin which one is used."""

    def test_standing_outside_does_not_count_as_entering(self):
        # THE decisive case. "West of House" contains "house", so a substring
        # test would suppress the single most valuable lead in the game.
        assert "white house" in unvisited_landmarks(ZORK_OPENING, ["West of House"])

    def test_all_three_vantage_points_still_leave_it_open(self):
        outside = ["West of House", "North of House", "South of House", "Behind House"]
        assert "white house" in unvisited_landmarks(ZORK_OPENING, outside)

    def test_a_room_named_for_it_retires_it(self):
        assert unvisited_landmarks("You are in the cellar.", ["Cellar"]) == []

    def test_same_sentence_retires_it(self):
        # Zork's Kitchen names the house while you are standing inside it.
        # Without sentence scope this lead would nag for the whole run.
        assert "white house" not in unvisited_landmarks(ZORK_KITCHEN, ["Kitchen"])

    def test_but_still_surfaces_the_new_lead_in_that_room(self):
        # The chimney is the actual route down to the Cellar.
        assert "dark chimney" in unvisited_landmarks(ZORK_KITCHEN, ["Kitchen"])

    def test_sentence_scope_does_not_over_retire(self):
        # "west of a white house" does not contain the location name
        # "West of House", so being outside retires nothing.
        text = "You are west of a white house. This is a forest."
        assert "white house" in unvisited_landmarks(text, ["West of House", "Forest"])

    def test_no_visited_locations_is_safe(self):
        assert "white house" in unvisited_landmarks(ZORK_OPENING, [])
        assert "white house" in unvisited_landmarks(ZORK_OPENING, None)


class TestRegression:
    def test_turn_four_would_now_have_a_lead(self):
        """The exact state where the verification run failed to diverge.

        At North of House the explorer picked NORTH on correct evidence and
        the arbiter had nothing to weigh it against: the map frontier was
        empty. The house must now be on offer.
        """
        north_of_house = ("North of House. You are facing the north side of a "
                          "white house. There is no door here, and all the "
                          "windows are boarded up. To the north a narrow path "
                          "winds through the trees.")
        visited = ["West of House", "North of House"]
        leads = unvisited_landmarks(north_of_house, visited)
        assert any("house" in lead for lead in leads), leads


class TestWiredIntoTurnContext:
    """The helper working proves nothing — every silent failure this session
    was in the wiring, not the logic. This calls the REAL build_turn_context.
    """

    def _context(self, game_text, location, transitions, turns=None):
        from types import SimpleNamespace as NS

        from tools.agent_graph.turn_context import build_turn_context

        edges = [NS(from_location=f, direction=d, to_location=t)
                 for f, d, t in transitions]
        mapper = NS(state=NS(
            get_all_transitions=lambda: edges,
            get_exits_from=lambda loc: [],
            get_unexplored_directions=lambda loc: [],
        ))
        history = NS(state=NS(
            get_recent_turns=lambda *a, **k: list(turns or []),
            get_full_summary=lambda *a, **k: "",
            get_long_running_summary=lambda *a, **k: "",
        ))
        inventory = NS(state=NS(get_items=lambda *a, **k: []))
        return build_turn_context(
            game_response=NS(Response=game_text, LocationName=location,
                             Score=0, Moves=4, exits=None, availableActions=None),
            history_toolkit=history, mapper_toolkit=mapper,
            inventory_toolkit=inventory, issue_locations=None,
        )

    def test_real_context_surfaces_the_house(self):
        ctx = self._context(
            "North of House. You are facing the north side of a white house.",
            "North of House",
            [("West of House", "NORTH", "North of House")],
        )
        assert any("house" in lead for lead in ctx.frontier), ctx.frontier

    def test_it_reaches_the_rendered_prompt(self):
        """The frontier landed in the prompt as the literal 'Nothing on the map
        is unexplored' line during the live run. Pin the rendered text."""
        ctx = self._context(
            "You are facing the north side of a white house.", "North of House",
            [("West of House", "NORTH", "North of House")],
        )
        assert "house" in ctx.frontier_summary.lower()
        assert "Nothing on the map is unexplored" not in ctx.frontier_summary

    def test_map_frontier_still_works(self):
        """The original source is kept, not replaced — a room reached and
        never left is still a lead."""
        ctx = self._context(
            "This is a forest.", "Forest",
            [("Clearing", "NORTH", "Forest"), ("Clearing", "EAST", "Canyon")],
        )
        assert any("Canyon" in lead for lead in ctx.frontier), ctx.frontier

    def test_empty_when_nothing_is_outstanding(self):
        # Every room reached has been left from, and the text names no
        # structure — the linear-path case where the old frontier was ALWAYS
        # empty. It should still be empty here; that was never the bug.
        ctx = self._context(
            "This is a forest, with trees in all directions.", "Forest",
            [("Clearing", "NORTH", "Forest"), ("Forest", "SOUTH", "Clearing")],
        )
        assert ctx.frontier == []
        assert "No unfollowed leads" in ctx.frontier_summary

    def test_the_linear_path_that_produced_an_empty_frontier(self):
        """The measured failure: A->B->C, every room left from, so the map
        frontier is empty — but the text still names the house."""
        ctx = self._context(
            "You are facing the north side of a white house.", "Forest",
            [("West of House", "NORTH", "North of House"),
             ("North of House", "NORTH", "Forest")],
        )
        map_leads = [f for f in ctx.frontier if "on the map" in f]
        assert map_leads == [], "map frontier is empty on a linear path — the bug"
        assert any("house" in f for f in ctx.frontier), "text frontier must cover it"


class TestScansRawProseNotTheTranscript:
    """The live failure the unit tests missed.

    `recent_turns` renders as "Turn 1: look -> West Of House\nYou are standing
    ... white house". With no sentence break between header and description,
    the location label lands in the same sentence as the landmark and fires the
    same-sentence retirement rule. Turn 4 of the frontier run reported "no
    unfollowed leads" with the house sitting in the text being scanned.
    """

    def test_the_formatted_transcript_is_safe_once_newlines_split(self):
        """This test previously pinned the OPPOSITE — that the transcript hid
        the lead — and failing was how the newline fix announced itself.

        Splitting on newlines separates "Turn 1: look -> West Of House" from
        the description below it, so the transcript no longer triggers a false
        retirement either. Reading raw prose is still preferred (it avoids the
        "Turn 3: TAKE leaflet ->" scaffolding entirely), but it is no longer
        load-bearing, and the newline split was the actual fix.
        """
        from tools.mapping.landmarks import unvisited_landmarks
        blob = ("Turn 1: look -> West Of House\n"
                "You are standing in an open field west of a white house.")
        assert "white house" in unvisited_landmarks(blob, ["West Of House"])

    def test_real_context_uses_raw_prose_and_finds_it(self):
        from types import SimpleNamespace as NS
        turns = [
            NS(turn_number=1, player_command="look",
               game_response="You are standing in an open field west of a "
                             "white house, with a boarded front door."),
            NS(turn_number=2, player_command="OPEN mailbox",
               game_response="Opening the small mailbox reveals a leaflet."),
            NS(turn_number=3, player_command="TAKE leaflet",
               game_response="Taken."),
        ]
        ctx = TestWiredIntoTurnContext()._context(
            "Taken.", "West Of House",
            [("West Of House", "NORTH", "North of House")], turns=turns)
        assert any("house" in f for f in ctx.frontier), ctx.frontier

    def test_current_response_alone_is_not_enough(self):
        """After TAKE the response is "Taken." and names nothing — exactly
        when the standing lead matters most."""
        from tools.mapping.landmarks import unvisited_landmarks
        assert unvisited_landmarks("Taken.", ["West Of House"]) == []


class TestRealGameResponseFormat:
    """The format the game actually sends, which broke this twice.

    A raw response is "West Of House\\nYou are standing in an open field west
    of a white house, with a boarded front door." -- the room name is a HEADER
    LINE with no terminating period. A sentence-only split fuses it to the
    description, putting the location name in the same "sentence" as the
    landmark and retiring the one lead that matters.

    Both failed live runs came from this. The unit tests missed it because they
    fed prose without the header line.
    """

    REAL = ("West Of House\nYou are standing in an open field west of a white "
            "house, with a boarded front door.\nThere is a small mailbox here.")

    def test_header_line_does_not_retire_the_landmark(self):
        from tools.mapping.landmarks import unvisited_landmarks
        assert "white house" in unvisited_landmarks(self.REAL, ["West Of House"])

    def test_the_turn_four_state_that_failed_twice(self):
        from tools.mapping.landmarks import unvisited_landmarks
        # "Taken." is the current response; the description is in history.
        text = "Taken.\n" + self.REAL
        assert "white house" in unvisited_landmarks(text, ["West Of House"])

    def test_header_line_still_allows_real_retirement(self):
        from tools.mapping.landmarks import unvisited_landmarks
        # Splitting on newlines must not break the inside-the-room case.
        kitchen = ("Kitchen\nYou are in the kitchen of the white house. "
                   "A dark chimney leads down.")
        leads = unvisited_landmarks(kitchen, ["Kitchen"])
        assert "white house" not in leads
        assert "dark chimney" in leads

    def test_room_named_for_landmark_with_header(self):
        from tools.mapping.landmarks import unvisited_landmarks
        assert unvisited_landmarks("Cellar\nYou are in a dark cellar.",
                                   ["Cellar"]) == []

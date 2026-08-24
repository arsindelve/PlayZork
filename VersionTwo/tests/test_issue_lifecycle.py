"""Keeping the objective alive, and pointing it at the right room.

Two defects that between them meant the memory system contributed nothing to
either Planetfall escape:

  * the IssueClosedAgent closed the escape pod — the objective — in four of
    five runs, always right after the turn-2 refusal;
  * an issue's `location` is the room it was NOTICED in, so one about
    somewhere else routed the agent to where it already stood.
"""
from types import SimpleNamespace as NS

import pytest

from tools.agent_graph.turn_context import build_turn_context
from tools.memory.issue_target import resolve_issue_target

KNOWN = ["Deck Nine", "Reactor Lobby", "Gangway", "Deck Eight",
         "West Of House", "North of House"]


class TestIssueTarget:
    def test_the_criteria_beats_the_sighting_room(self):
        """THE pf5 case."""
        assert resolve_issue_target(
            "Ensign Blather at Reactor Lobby — return to Deck Nine as ordered",
            "Reactor Lobby", KNOWN) == "Deck Nine"

    def test_the_subject_room_is_used_when_the_criteria_names_none(self):
        assert resolve_issue_target(
            "Locked pod bulkhead at Deck Nine — open bulkhead and examine pod",
            "Deck Nine", KNOWN) == "Deck Nine"

    def test_the_sighting_room_survives_when_nothing_matches(self):
        """An unmatched issue must behave exactly as it does today."""
        assert resolve_issue_target(
            "Grating at Clearing — open or unlock it", "Clearing", KNOWN) == "Clearing"

    def test_it_cannot_invent_a_destination(self):
        """The map is the vocabulary, so an unseen room is never returned."""
        assert resolve_issue_target(
            "Something at Atlantis — go to Atlantis", "Gangway", KNOWN) == "Gangway"

    def test_longest_name_wins(self):
        assert resolve_issue_target(
            "Leaflet at West Of House — carry it to North of House",
            "West Of House", KNOWN) == "North of House"

    @pytest.mark.parametrize("content", [None, "", "no rooms named here"])
    def test_degenerate_content_falls_back(self, content):
        assert resolve_issue_target(content, "Gangway", KNOWN) == "Gangway"

    def test_no_known_locations_falls_back(self):
        assert resolve_issue_target("x at y — go to Deck Nine", "Gangway", []) == "Gangway"


def _context(*, location, previous_location, score, api_inventory, held,
             prior_turns=()):
    mapper = NS(state=NS(get_all_transitions=lambda: [],
                         get_exits_from=lambda l: [],
                         get_unexplored_directions=lambda l: []))
    history = NS(state=NS(get_recent_turns=lambda *a, **k: list(prior_turns),
                          get_full_summary=lambda *a, **k: "",
                          get_long_running_summary=lambda *a, **k: ""))
    inventory = NS(state=NS(get_items=lambda *a, **k: list(held)))
    return build_turn_context(
        game_response=NS(Response="x", LocationName=location,
                         PreviousLocationName=previous_location,
                         Score=score, Moves=2, exits=None, availableActions=None,
                         Inventory=list(api_inventory)),
        history_toolkit=history, mapper_toolkit=mapper,
        inventory_toolkit=inventory, issue_locations=None)


class TestAccomplishedSomething:
    """The gate on issue closure."""

    def test_a_refusal_accomplishes_nothing(self):
        """Turn 2 of every Planetfall run: 'Why open the door to the emergency
        escape pod if there's no emergency?' — no move, no score, no items."""
        c = _context(location="Deck Nine", previous_location="Deck Nine", score=0,
                     api_inventory=["brush"], held=["brush"],
                     prior_turns=[NS(score=0, location="Deck Nine")])
        assert not c.accomplished_something

    def test_moving_counts(self):
        c = _context(location="Reactor Lobby", previous_location="Deck Nine", score=0,
                     api_inventory=["brush"], held=["brush"],
                     prior_turns=[NS(score=0, location="Deck Nine")])
        assert c.accomplished_something

    def test_scoring_counts(self):
        c = _context(location="Deck Nine", previous_location="Deck Nine", score=3,
                     api_inventory=["brush"], held=["brush"],
                     prior_turns=[NS(score=0, location="Deck Nine")])
        assert c.accomplished_something

    def test_picking_something_up_counts(self):
        """Taking an item neither moves nor scores in Zork, but it is real
        progress and an issue like 'take the leaflet' must be closable."""
        c = _context(location="West Of House", previous_location="West Of House",
                     score=0, api_inventory=["brush", "leaflet"], held=["brush"],
                     prior_turns=[NS(score=0, location="West Of House")])
        assert c.accomplished_something

    def test_it_defaults_to_permissive_on_missing_data(self):
        """Never block closure because we could not tell — the guard exists to
        stop a confident wrong closure, not to add a new failure mode."""
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
        assert c.accomplished_something


class TestKnownLocations:
    def test_collected_from_the_map(self):
        edges = [NS(from_location="Deck Nine", direction="EAST",
                    to_location="Reactor Lobby")]
        mapper = NS(state=NS(get_all_transitions=lambda: edges,
                             get_exits_from=lambda l: [],
                             get_unexplored_directions=lambda l: []))
        history = NS(state=NS(get_recent_turns=lambda *a, **k: [],
                              get_full_summary=lambda *a, **k: "",
                              get_long_running_summary=lambda *a, **k: ""))
        c = build_turn_context(
            game_response=NS(Response="x", LocationName="Gangway", Score=0, Moves=1,
                             exits=None, availableActions=None),
            history_toolkit=history, mapper_toolkit=mapper,
            inventory_toolkit=NS(state=NS(get_items=lambda *a, **k: [])),
            issue_locations=None)
        assert "Deck Nine" in c.known_locations
        assert "Reactor Lobby" in c.known_locations
        assert "Gangway" in c.known_locations, "the current room counts too"

    def test_blocked_is_not_a_place(self):
        edges = [NS(from_location="Deck Nine", direction="NORTH",
                    to_location="BLOCKED")]
        mapper = NS(state=NS(get_all_transitions=lambda: edges,
                             get_exits_from=lambda l: [],
                             get_unexplored_directions=lambda l: []))
        history = NS(state=NS(get_recent_turns=lambda *a, **k: [],
                              get_full_summary=lambda *a, **k: "",
                              get_long_running_summary=lambda *a, **k: ""))
        c = build_turn_context(
            game_response=NS(Response="x", LocationName="Deck Nine", Score=0, Moves=1,
                             exits=None, availableActions=None),
            history_toolkit=history, mapper_toolkit=mapper,
            inventory_toolkit=NS(state=NS(get_items=lambda *a, **k: [])),
            issue_locations=None)
        assert "BLOCKED" not in c.known_locations

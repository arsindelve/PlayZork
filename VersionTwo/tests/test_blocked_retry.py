"""A BLOCKED direction must be RE-TRIED, not written off (GitHub issue #31).

#11 made a stale BLOCKED edge *correctable* — a later success overwrites it —
but only if the direction is attempted again. The explorer treated any known
edge, BLOCKED included, as explored and never re-proposed it, so a wall that
had since cleared (troll left, grating unlocked, drawbridge lowered) stayed a
permanent wall unless some other agent happened to issue that command.

Option 3 (per the issue): keep BLOCKED directions as candidates but at LOW
priority, so real frontier is always preferred and an old wall is revisited
only once a room is otherwise exhausted — no turn burned re-probing walls every
cycle, but the map can finally repair itself.
"""
import pytest

from tools.agent_graph.explorer_agent import ExplorerAgent
from tools.database import DatabaseManager
from tools.mapping.directions import explorer_direction_pools
from tools.mapping.mapper_state import MapperState


# --- explorer_direction_pools: the split -----------------------------------

class TestDirectionPools:
    def test_a_real_passage_is_explored_and_in_neither_pool(self):
        unexplored, retry = explorer_direction_pools([("NORTH", "Kitchen")])
        assert "NORTH" not in unexplored
        assert "NORTH" not in retry

    def test_a_blocked_direction_is_offered_as_retry_not_dropped(self):
        unexplored, retry = explorer_direction_pools([("NORTH", "BLOCKED")])
        assert "NORTH" in retry
        assert "NORTH" not in unexplored

    def test_untried_directions_are_frontier(self):
        unexplored, retry = explorer_direction_pools([("NORTH", "BLOCKED")])
        # Everything except NORTH was never tried from here.
        assert "SOUTH" in unexplored and "EAST" in unexplored
        assert retry == ["NORTH"]

    def test_a_passage_wins_over_a_stale_blocked_row_for_the_same_direction(self):
        # get_exits_from's _collapse already prefers the passage, but the split
        # must not re-introduce the direction as a retry if both are present.
        unexplored, retry = explorer_direction_pools(
            [("DOWN", "BLOCKED"), ("DOWN", "Cellar")]
        )
        assert "DOWN" not in retry
        assert "DOWN" not in unexplored

    def test_abbreviations_normalize_before_splitting(self):
        # A legacy row stored as "N" must still count as NORTH (#9).
        unexplored, retry = explorer_direction_pools([("N", "BLOCKED")])
        assert retry == ["NORTH"]
        assert "NORTH" not in unexplored

    def test_pools_are_in_canonical_order(self):
        unexplored, retry = explorer_direction_pools(
            [("EAST", "BLOCKED"), ("NORTH", "BLOCKED")]
        )
        assert retry == ["NORTH", "EAST"]


# --- ExplorerAgent: rank, don't exclude -------------------------------------

class TestExplorerPrefersFrontier:
    def test_frontier_is_chosen_over_a_retry(self):
        # DOWN is bare frontier (no evidence); NORTH is a refused direction the
        # game even re-advertises as an exit. Frontier must still win.
        agent = ExplorerAgent(
            "X", unexplored_directions=["DOWN"], mentioned_directions=[],
            turn_number=1, game_exits=["NORTH"], retry_directions=["NORTH"],
        )
        assert agent.best_direction == "DOWN"
        assert agent.is_retry is False

    def test_a_frontier_proposal_keeps_normal_confidence_and_ev(self):
        agent = ExplorerAgent(
            "X", unexplored_directions=["NORTH", "SOUTH", "EAST"],
            mentioned_directions=[], turn_number=1, retry_directions=["WEST"],
        )
        assert agent.is_retry is False
        assert agent.exploration_ev_count() == 3          # frontier count, not 4
        assert agent._calculate_confidence(agent.best_direction) >= 55


class TestExplorerRetriesWhenExhausted:
    def test_a_lone_retry_is_proposed_when_no_frontier_remains(self):
        agent = ExplorerAgent(
            "X", unexplored_directions=[], mentioned_directions=[],
            turn_number=1, retry_directions=["NORTH"],
        )
        assert agent.best_direction == "NORTH"
        assert agent.is_retry is True

    def test_a_retry_gets_low_confidence_and_a_small_nonzero_ev(self):
        agent = ExplorerAgent(
            "X", unexplored_directions=[], mentioned_directions=[],
            turn_number=1, retry_directions=["NORTH"],
        )
        agent.confidence = agent._calculate_confidence(agent.best_direction)
        assert agent.confidence == 30
        assert agent.exploration_ev_count() == 1
        # EV = (1/10) * (30/100) * 50 = 1.5 — positive, so the arbiter can rank
        # it (a zero-EV proposal would be withheld), but tiny.
        ev = (agent.exploration_ev_count() / 10) * (agent.confidence / 100) * 50
        assert 0 < ev < 5

    def test_evidence_still_ranks_among_retries(self):
        # No frontier; two refused directions. The one the game advertises as an
        # exit is the better re-probe.
        agent = ExplorerAgent(
            "X", unexplored_directions=[], mentioned_directions=[],
            turn_number=1, game_exits=["EAST"], retry_directions=["NORTH", "EAST"],
        )
        assert agent.best_direction == "EAST"
        assert agent.is_retry is True


# --- End to end: the correction #11 enabled now actually happens -------------

class TestMapRepairViaRetry:
    @pytest.fixture
    def mapper(self, tmp_path):
        return MapperState(session_id="s", db=DatabaseManager(db_path=str(tmp_path / "t.db")))

    def test_a_blocked_edge_reaches_the_explorer_as_a_retry(self, mapper):
        mapper.record_movement("Cellar", "BLOCKED", "NORTH", 5)  # troll blocks the way
        unexplored, retry = explorer_direction_pools(mapper.get_exits_from("Cellar"))
        assert retry == ["NORTH"]

    def test_once_the_wall_clears_the_direction_becomes_explored(self, mapper):
        mapper.record_movement("Cellar", "BLOCKED", "NORTH", 5)
        mapper.record_movement("Cellar", "Troll Room", "NORTH", 12)  # troll gone
        unexplored, retry = explorer_direction_pools(mapper.get_exits_from("Cellar"))
        assert "NORTH" not in retry
        assert "NORTH" not in unexplored

"""Stage-0 pursuit: a "means" aim (find/obtain X) that can't be acted on here
searches for the means instead of giving up or looping back to where it was
noticed. Unit-tests the deterministic `_search_step` (no LLM needed).

Motivated by escaperoom-25: "find a light source" pointed at the dark Storage
Closet, so the route logic bounced the agent back into the dark instead of
searching the lit rooms for a light.
"""
from types import SimpleNamespace as NS

from tools.agent_graph.issue_agent import IssueAgent


def _agent():
    # _search_step uses only its `context` arg, so bypass __init__.
    return IssueAgent.__new__(IssueAgent)


def _ctx(exits, game_exits, actions, unproductive=()):
    up = {u.upper() for u in unproductive}
    return NS(exits=exits, game_exits=game_exits, available_actions=actions,
              is_unproductive=lambda c: (c or "").strip().upper() in up)


def test_prefers_unexplored_game_exit_as_new_ground():
    # Game reports NORTH here but the map has no NORTH edge yet -> new ground.
    ctx = _ctx(exits=[("SOUTH", "Reception")], game_exits=["SOUTH", "NORTH"],
               actions={"desk": ["examine desk"]})
    assert _agent()._search_step(ctx) == ("NORTH", 50)


def test_falls_back_to_examining_an_untried_object():
    ctx = _ctx(exits=[("SOUTH", "Reception")], game_exits=["SOUTH"],
               actions={"lamp": ["take lamp"]})
    assert _agent()._search_step(ctx) == ("EXAMINE LAMP", 45)


def test_last_resort_leaves_a_searched_dead_end_via_a_known_exit():
    ctx = _ctx(exits=[("SOUTH", "Reception")], game_exits=["SOUTH"], actions={})
    assert _agent()._search_step(ctx) == ("SOUTH", 40)


def test_returns_none_when_there_is_genuinely_nothing_to_try():
    # Dark dead-end: no unexplored exits, no visible objects, and the only known
    # exit was already shown useless. Better "nothing" than a forced loop.
    ctx = _ctx(exits=[("SOUTH", "Reception")], game_exits=["SOUTH"], actions={},
               unproductive=["SOUTH"])
    assert _agent()._search_step(ctx) is None


def test_means_aim_detection_keywords():
    # Sanity: the phrasing the Observer actually produced is detected as a means aim.
    for content in ["Darkness in Storage Closet — find a light source to proceed",
                    "Locked grating — the key is in my inventory",
                    "Sealed door — obtain a tool to open it"]:
        low = content.lower()
        assert any(k in low for k in ("find ", "obtain", "light source",
                                      "the key", "a tool", "in my inventory"))

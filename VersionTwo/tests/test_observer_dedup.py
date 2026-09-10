"""The Observer's duplicate-avoidance list must show every OPEN issue (#29).

Unlike the spawner and closer — which rank by lazily-decayed importance so fresh
issues outrank stale ones (#20) — the Observer shows tracked issues to the model
purely so it does not re-report one already open. For that, a stale-but-open
issue is exactly what you must still see: it has decayed toward importance 0 but
is still open, and re-reporting it makes a duplicate. So this list is undecayed
and open-only, capped only to bound prompt size, with a warning when the cap
bites. Decaying it (or a tight importance-ranked limit) would evict the stalest
open issues and make duplicates MORE likely — the opposite of the goal.
"""
import logging
from types import SimpleNamespace as NS

from tools.agent_graph.observer_agent import ObserverAgent, TRACKED_ISSUE_DEDUP_CAP


class FakeState:
    """Records how get_top_memories was called and returns canned memories."""

    def __init__(self, memories):
        self.memories = memories
        self.calls = []

    def get_top_memories(self, limit=10, include_closed=False, current_turn=None, decay_factor=0.9):
        self.calls.append(dict(limit=limit, include_closed=include_closed,
                               current_turn=current_turn))
        # Mirror the DB: importance-ordered, capped at limit.
        ordered = sorted(self.memories, key=lambda m: m.importance, reverse=True)
        return ordered[:limit]


def _toolkit(memories):
    return NS(state=FakeState(memories))


def _mem(content, importance):
    return NS(content=content, importance=importance)


def test_it_asks_for_open_issues_undecayed_at_the_cap():
    tk = _toolkit([_mem("troll blocks the bridge", 800)])
    ObserverAgent()._tracked_issues_for_dedup(tk)

    call = tk.state.calls[0]
    assert call["limit"] == TRACKED_ISSUE_DEDUP_CAP
    # Undecayed on purpose (#29): passing current_turn would be the #20 fix,
    # which is wrong here.
    assert call["current_turn"] is None
    # Open issues only — closed ones are handled by the deduplicator.
    assert call["include_closed"] is False


def test_a_stale_but_open_issue_is_still_shown():
    # A once-important issue decayed to near-zero importance is still open, so
    # it must remain visible or the model will re-report it.
    tk = _toolkit([
        _mem("fresh: locked grating overhead", 700),
        _mem("stale: mailbox contains a leaflet", 1),
    ])
    text = ObserverAgent()._tracked_issues_for_dedup(tk)

    assert "mailbox contains a leaflet" in text
    assert "locked grating overhead" in text


def test_no_open_issues_reads_cleanly():
    text = ObserverAgent()._tracked_issues_for_dedup(_toolkit([]))
    assert text == "No issues tracked yet."


def test_the_cap_biting_is_logged(caplog):
    many = [_mem(f"issue {i}", 500) for i in range(TRACKED_ISSUE_DEDUP_CAP + 5)]
    with caplog.at_level(logging.WARNING):
        text = ObserverAgent()._tracked_issues_for_dedup(_toolkit(many))

    assert any("hit its cap" in r.message for r in caplog.records)
    # Still returns exactly the cap's worth, one per line.
    assert len(text.splitlines()) == TRACKED_ISSUE_DEDUP_CAP


def test_a_normal_sized_list_does_not_warn(caplog):
    tk = _toolkit([_mem(f"issue {i}", 500) for i in range(5)])
    with caplog.at_level(logging.WARNING):
        ObserverAgent()._tracked_issues_for_dedup(tk)

    assert not any("hit its cap" in r.message for r in caplog.records)

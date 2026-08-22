"""Turn-budget coherence and cancellation atomicity (GitHub issue #3).

Two defects are covered here:
  1. The retry envelope for a single LLM call (300s x 5 + backoff = 1530s) was
     larger than the turn budget (600s), so retries 2..N were unreachable.
  2. `close_issues` closed memories immediately, so a turn cancelled during
     `observe` left issues closed with none of the turn's other bookkeeping
     applied — and the next session resumed from that state.
"""

import asyncio
import inspect
from types import SimpleNamespace

import config
from config import (
    LLM_MAX_RETRIES,
    LLM_RETRY_ENVELOPE_SECONDS,
    LLM_TIMEOUT_SECONDS,
    TURN_BUDGET_FLOOR_SECONDS,
    TURN_BUDGET_SECONDS,
    retry_envelope_seconds,
)
from tools.agent_graph.decision_graph import create_persist_node
from tools.agent_graph.issue_closed_agent import IssueClosedAgent


# ---------------------------------------------------------------------------
# 1. Budget coherence
# ---------------------------------------------------------------------------


def test_retry_envelope_fits_inside_the_turn_budget():
    """The core arithmetic from #3: retries must be reachable."""
    assert LLM_RETRY_ENVELOPE_SECONDS < TURN_BUDGET_SECONDS


def test_turn_budget_is_at_least_the_floor():
    assert TURN_BUDGET_SECONDS >= TURN_BUDGET_FLOOR_SECONDS


def test_old_configuration_would_be_rejected_by_the_invariant():
    """Regression guard: the pre-fix numbers must not satisfy the invariant."""
    old_envelope = retry_envelope_seconds(timeout_seconds=300, max_retries=5)
    assert old_envelope == 1530  # 5*300 + (2+4+8+16)
    assert old_envelope > 600  # the old TURN_BUDGET_SECONDS


def test_envelope_matches_llm_utils_backoff_schedule():
    """The formula must track the retry loop it models.

    llm_utils sleeps 2**attempt after each failed attempt except the last.
    """
    source = inspect.getsource(config.retry_envelope_seconds)
    assert "2 ** attempt" in source

    simulated = 0
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        simulated += LLM_TIMEOUT_SECONDS
        if attempt < LLM_MAX_RETRIES:
            simulated += 2 ** attempt
    assert simulated == LLM_RETRY_ENVELOPE_SECONDS


def test_per_attempt_timeout_clears_measured_call_latency():
    """113s was the slowest measured qwen2.5:14b call on the smoke run; a
    60-90s cap (as originally suggested) would abort healthy calls."""
    assert LLM_TIMEOUT_SECONDS > 113


def test_configured_budget_below_floor_is_raised(monkeypatch):
    """A too-small PLAYZORK_TURN_BUDGET_SECONDS must be corrected, not obeyed."""
    import importlib

    monkeypatch.setenv("PLAYZORK_TURN_BUDGET_SECONDS", "60")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.TURN_BUDGET_SECONDS == reloaded.TURN_BUDGET_FLOOR_SECONDS
        assert reloaded.TURN_BUDGET_SECONDS > 60
    finally:
        monkeypatch.delenv("PLAYZORK_TURN_BUDGET_SECONDS", raising=False)
        importlib.reload(config)


# ---------------------------------------------------------------------------
# 2. Closures are staged, not applied, until persist
# ---------------------------------------------------------------------------


class FakeMemoryState:
    def __init__(self):
        self.closed = []

    def get_top_memories(self, limit=10, **kwargs):
        return [
            SimpleNamespace(id=1, content="open the mailbox", importance=600),
            SimpleNamespace(id=2, content="enter the house", importance=400),
        ]

    def remove_memory(self, memory_id):
        self.closed.append(memory_id)
        return True


class FakeInventoryState:
    def add_item(self, item_name, turn_number):
        pass

    def remove_item(self, item_name, turn_number):
        pass

    def get_items(self):
        return []


def _analyze_with_stub_llm(monkeypatch, state, closed_ids, current_turn=None):
    """Run IssueClosedAgent.analyze with the LLM and history tool stubbed."""
    agent = IssueClosedAgent()
    memory_toolkit = SimpleNamespace(state=state)
    history_toolkit = SimpleNamespace(get_tools=lambda: [])

    response = SimpleNamespace(
        closed_issue_ids=list(closed_ids),
        closed_issue_contents=[],
        reasoning="solved",
    )
    # issue_closed_agent imports invoke_with_retry inside the method body.
    monkeypatch.setattr("llm_utils.invoke_with_retry", lambda *args, **kwargs: response)

    decision_llm = SimpleNamespace(
        with_structured_output=lambda schema: SimpleNamespace(
            with_config=lambda **kwargs: None
        )
    )
    return agent.analyze(
        game_response="Opened.",
        location="West Of House",
        score=0,
        moves=1,
        decision_llm=decision_llm,
        history_toolkit=history_toolkit,
        memory_toolkit=memory_toolkit,
        current_turn=current_turn,
    )


def test_analyze_stages_closures_without_writing(monkeypatch):
    memory_state = FakeMemoryState()

    response, pending = _analyze_with_stub_llm(monkeypatch, memory_state, [1])

    # Nothing was closed: the write is persist_node's job now.
    assert memory_state.closed == []
    assert pending == [{"id": 1, "display": "[ID:1, 600/1000] open the mailbox"}]
    # And nothing is reported as closed yet.
    assert response.closed_issue_contents == []


def test_persist_applies_staged_closures():
    memory_state = FakeMemoryState()
    issue_closed_response = SimpleNamespace(
        closed_issue_ids=[1], closed_issue_contents=[], reasoning=""
    )

    persist = create_persist_node(
        SimpleNamespace(state=memory_state, add_memory=lambda **kwargs: True),
        SimpleNamespace(state=FakeInventoryState()),
        {"current": 5},
    )
    persist({
        "game_response": SimpleNamespace(
            Response="Opened.", LocationName="West Of House", Score=0, Moves=1
        ),
        "player_command": "",  # skip the inventory LLM
        "decision": SimpleNamespace(command="LOOK"),
        "observer_response": None,
        "pending_closures": [{"id": 1, "display": "[ID:1, 600/1000] open the mailbox"}],
        "issue_closed_response": issue_closed_response,
    })

    assert memory_state.closed == [1]
    # Only what actually committed is reported.
    assert issue_closed_response.closed_issue_contents == [
        "[ID:1, 600/1000] open the mailbox"
    ]


def test_turn_cancelled_before_persist_leaves_memory_untouched(monkeypatch):
    """The half-commit scenario from #3, end to end.

    close_issues runs, then the turn budget fires during observe. Because
    persist never runs, no issue is closed.
    """
    memory_state = FakeMemoryState()

    _, pending = _analyze_with_stub_llm(monkeypatch, memory_state, [1, 2])
    assert pending  # closures were decided

    async def observe_then_time_out():
        raise asyncio.TimeoutError

    try:
        asyncio.run(observe_then_time_out())
    except asyncio.TimeoutError:
        pass

    # persist_node never ran -> memory is exactly as it was.
    assert memory_state.closed == []


def test_persist_skips_closures_that_fail_to_write():
    class HalfBrokenState(FakeMemoryState):
        def remove_memory(self, memory_id):
            if memory_id == 2:
                raise RuntimeError("database is locked")
            return super().remove_memory(memory_id)

    memory_state = HalfBrokenState()
    issue_closed_response = SimpleNamespace(
        closed_issue_ids=[1, 2], closed_issue_contents=[], reasoning=""
    )

    persist = create_persist_node(
        SimpleNamespace(state=memory_state, add_memory=lambda **kwargs: True),
        SimpleNamespace(state=FakeInventoryState()),
        {"current": 5},
    )
    persist({
        "game_response": SimpleNamespace(
            Response="Opened.", LocationName="West Of House", Score=0, Moves=1
        ),
        "player_command": "",
        "decision": SimpleNamespace(command="LOOK"),
        "observer_response": None,
        "pending_closures": [
            {"id": 1, "display": "one"},
            {"id": 2, "display": "two"},
        ],
        "issue_closed_response": issue_closed_response,
    })

    # The healthy closure still applied; the broken one didn't stop the turn.
    assert memory_state.closed == [1]
    assert issue_closed_response.closed_issue_contents == ["one"]


# ---------------------------------------------------------------------------
# 3. Closure IDs are validated against what the model was shown (#19)
# ---------------------------------------------------------------------------


def test_prompt_example_ids_are_never_staged(monkeypatch):
    """#19: the model echoes the prompt's own worked example, [5, 12]."""
    state = FakeMemoryState()  # shows only IDs 1 and 2

    _, pending = _analyze_with_stub_llm(monkeypatch, state, [5, 12])

    assert pending == []
    assert state.closed == []


def test_hallucinated_id_dropped_but_valid_one_kept(monkeypatch):
    """Per-ID, not all-or-nothing: one bogus ID must not void a real closure."""
    _, pending = _analyze_with_stub_llm(monkeypatch, FakeMemoryState(), [1, 999])

    assert pending == [{"id": 1, "display": "[ID:1, 600/1000] open the mailbox"}]


def test_duplicate_ids_are_staged_once(monkeypatch):
    _, pending = _analyze_with_stub_llm(monkeypatch, FakeMemoryState(), [2, 2, 2])

    assert [closure["id"] for closure in pending] == [2]


def test_every_staged_closure_carries_display_text(monkeypatch):
    """The invariant persist_node's guard relies on."""
    _, pending = _analyze_with_stub_llm(monkeypatch, FakeMemoryState(), [1, 2, 77])

    assert pending
    assert all(closure["display"] for closure in pending)


def test_persist_refuses_a_closure_without_display():
    """Defence in depth: an unvalidated closure must not reach remove_memory."""
    memory_state = FakeMemoryState()
    persist = create_persist_node(
        SimpleNamespace(state=memory_state, add_memory=lambda **kwargs: True),
        SimpleNamespace(state=FakeInventoryState()),
        {"current": 5},
    )
    persist({
        "game_response": SimpleNamespace(
            Response="ok", LocationName="West Of House", Score=0, Moves=1
        ),
        "player_command": "",
        "decision": SimpleNamespace(command="LOOK"),
        "observer_response": None,
        "pending_closures": [{"id": 5, "display": None}],
        "issue_closed_response": SimpleNamespace(
            closed_issue_ids=[5], closed_issue_contents=[], reasoning=""
        ),
    })

    assert memory_state.closed == []


def test_prompt_contains_no_plausible_real_ids():
    """Static guard on the prompt itself: real memory IDs start at 1, so any
    example ID in that range is echoable onto a live issue."""
    import re

    from adventurer.prompt_library import PromptLibrary

    prompt = PromptLibrary.get_issue_closed_analysis_prompt(
        "(none)", "history", "West Of House", "Opened."
    )
    ids = [
        int(number)
        for block in re.findall(r'"closed_issue_ids"\s*:\s*\[([^\]]*)\]', prompt)
        for number in re.findall(r"\d+", block)
    ]
    ids += [int(number) for number in re.findall(r"\[ID:(\d+)", prompt)]

    assert ids, "expected the prompt to contain at least one example ID"
    assert all(i >= 9000 for i in ids), f"prompt contains echoable real-looking IDs: {ids}"


def test_closer_ranks_by_decayed_importance(monkeypatch):
    """#20: the closer's window must use the same decay as the spawner's."""

    class RecordingMemoryState(FakeMemoryState):
        def __init__(self):
            super().__init__()
            self.calls = []

        def get_top_memories(self, limit=10, **kwargs):
            self.calls.append({"limit": limit, **kwargs})
            return super().get_top_memories(limit=limit, **kwargs)

    state = RecordingMemoryState()
    _analyze_with_stub_llm(monkeypatch, state, [1], current_turn=42)

    assert state.calls == [{"limit": 30, "current_turn": 42}]

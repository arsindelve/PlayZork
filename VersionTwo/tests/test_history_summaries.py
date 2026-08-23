"""The two per-turn summaries must run concurrently (GitHub issue #24).

They used to run serially at the head of every turn, blocking all agent work:
86s on turn 1 of the measured smoke run, with the recent summary alone taking
113s by turn 2.
"""

import asyncio
from types import SimpleNamespace

from tools.history import HistoryToolkit


class FakeHistoryState:
    def __init__(self):
        self.saved = None
        self.turn = SimpleNamespace(
            turn_number=7,
            player_command="OPEN MAILBOX",
            game_response="Opening the mailbox reveals a leaflet.",
            location="West Of House",
            score=0,
            moves=7,
        )

    def add_turn(self, **kwargs):
        return self.turn

    def get_turn_count(self):
        return 7

    def save_both_summaries(self, recent, long_running):
        self.saved = (recent, long_running)


class RendezvousSummarizer:
    """Each summary blocks until the other has started.

    If the two ran serially, the first would wait forever for a partner that
    has not been scheduled yet — so completing at all proves concurrency.
    """

    def __init__(self, timeout=2.0):
        self.recent_started = asyncio.Event()
        self.long_started = asyncio.Event()
        self.timeout = timeout

    async def agenerate_summary(self, state, turn):
        self.recent_started.set()
        await asyncio.wait_for(self.long_started.wait(), timeout=self.timeout)
        return "recent summary"

    async def agenerate_long_running_summary(self, state, turn):
        self.long_started.set()
        await asyncio.wait_for(self.recent_started.wait(), timeout=self.timeout)
        return "long summary"


def _make_toolkit(summarizer, state=None):
    toolkit = object.__new__(HistoryToolkit)
    toolkit.state = state if state is not None else FakeHistoryState()
    toolkit.summarizer = summarizer
    return toolkit


def _update(toolkit):
    return asyncio.run(
        toolkit.update_after_turn(
            game_response="Opening the mailbox reveals a leaflet.",
            player_command="OPEN MAILBOX",
            location="West Of House",
            score=0,
            moves=7,
        )
    )


def test_the_two_summaries_run_concurrently():
    state = FakeHistoryState()
    toolkit = _make_toolkit(RendezvousSummarizer(), state)

    # Serial execution would deadlock here and raise TimeoutError.
    _update(toolkit)

    assert state.saved == ("recent summary", "long summary")


def test_wall_clock_is_the_slower_call_not_the_sum():
    class SleepySummarizer:
        async def agenerate_summary(self, state, turn):
            await asyncio.sleep(0.20)
            return "recent"

        async def agenerate_long_running_summary(self, state, turn):
            await asyncio.sleep(0.20)
            return "long"

    state = FakeHistoryState()
    toolkit = _make_toolkit(SleepySummarizer(), state)

    loop_start = asyncio.new_event_loop()
    try:
        started = loop_start.time()
        loop_start.run_until_complete(
            toolkit.update_after_turn(
                game_response="r", player_command="c", location="l", score=0, moves=1
            )
        )
        elapsed = loop_start.time() - started
    finally:
        loop_start.close()

    # Serial would be >= 0.40s; concurrent is ~0.20s.
    assert elapsed < 0.35
    assert state.saved == ("recent", "long")


def test_one_failing_summary_does_not_raise_or_save():
    """Preserves pre-fix semantics: if either summary fails, neither is
    committed and the previous turn's summaries stay in place."""

    class HalfBrokenSummarizer:
        def __init__(self):
            self.long_completed = False

        async def agenerate_summary(self, state, turn):
            raise RuntimeError("ollama connection reset")

        async def agenerate_long_running_summary(self, state, turn):
            await asyncio.sleep(0.01)
            self.long_completed = True
            return "long summary"

    summarizer = HalfBrokenSummarizer()
    state = FakeHistoryState()
    toolkit = _make_toolkit(summarizer, state)

    _update(toolkit)  # must not raise

    assert state.saved is None
    # return_exceptions=True: the sibling ran to completion rather than being
    # left orphaned when gather re-raised.
    assert summarizer.long_completed is True


def test_both_failing_summaries_are_contained():
    class BrokenSummarizer:
        async def agenerate_summary(self, state, turn):
            raise RuntimeError("boom")

        async def agenerate_long_running_summary(self, state, turn):
            raise ValueError("also boom")

    state = FakeHistoryState()
    toolkit = _make_toolkit(BrokenSummarizer(), state)

    _update(toolkit)

    assert state.saved is None


def test_update_after_turn_is_a_coroutine():
    """game_session must await it; a forgotten await would silently skip all
    summarization and emit only a RuntimeWarning."""
    assert asyncio.iscoroutinefunction(HistoryToolkit.update_after_turn)


def test_game_session_records_the_turn_on_the_critical_path():
    """record_turn must stay synchronous and inline: this turn's agents
    research against it through get_recent_turns."""
    import inspect

    import game_session

    source = inspect.getsource(game_session.GameSession)
    assert "self.history_toolkit.record_turn(" in source
    assert "await self.history_toolkit.record_turn(" not in source


def test_game_session_dispatches_summaries_off_the_critical_path():
    """#24 option 2: summarization must not block the turn. It cost 65s+ at
    the head of every turn on the 2026-08-22 checkpoint run."""
    import inspect

    import game_session

    source = inspect.getsource(game_session.GameSession)
    assert "_dispatch_summary_refresh" in source
    # and it must not be awaited inline, or it is still on the critical path
    assert "await self.history_toolkit.refresh_summaries(" not in source


def test_summary_refresh_is_coalesced_against_overlapping_turns():
    """Turns will overlap once the engine gets faster. Two summarizers racing
    would let an older result overwrite a newer one via save_both_summaries."""
    order = []

    class SlowSummarizer:
        def __init__(self):
            self.calls = 0

        async def agenerate_summary(self, state, turn):
            self.calls += 1
            order.append(f"start{turn.turn_number}")
            await asyncio.sleep(0.05)
            order.append(f"end{turn.turn_number}")
            return f"recent{turn.turn_number}"

        async def agenerate_long_running_summary(self, state, turn):
            await asyncio.sleep(0.05)
            return f"long{turn.turn_number}"

    state = FakeHistoryState()
    toolkit = _make_toolkit(SlowSummarizer(), state)

    async def drive():
        t1 = SimpleNamespace(turn_number=1)
        t2 = SimpleNamespace(turn_number=2)
        await asyncio.gather(
            toolkit.refresh_summaries(t1),
            toolkit.refresh_summaries(t2),
        )

    asyncio.run(drive())

    # Never interleaved: one finishes before the next begins.
    assert order == ["start1", "end1", "start2", "end2"], order
    # The newer turn's summaries are what survive.
    assert state.saved == ("recent2", "long2")


def test_a_failing_summary_refresh_never_escapes():
    """It runs as a background task now; an escaping exception would surface
    only as a task-failure log, so it is contained at the source."""
    class BrokenSummarizer:
        async def agenerate_summary(self, state, turn):
            raise RuntimeError("boom")

        async def agenerate_long_running_summary(self, state, turn):
            raise RuntimeError("boom")

    state = FakeHistoryState()
    toolkit = _make_toolkit(BrokenSummarizer(), state)

    asyncio.run(toolkit.refresh_summaries(SimpleNamespace(turn_number=3)))

    assert state.saved is None

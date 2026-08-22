"""GameSession must survive a failed turn (GitHub issue #1).

Before the fix, `while True:` sat inside the try block: the first exception
from any of ~25 LLM calls per turn fell through to `finally` and the session
was over. These tests drive `play()` with a stubbed turn function.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest

import game_session as gs
from game_session import FALLBACK_COMMAND, MAX_CONSECUTIVE_TURN_FAILURES, GameSession


class FakeDisplay:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def _make_session(turn_behaviors, monkeypatch):
    """Build a GameSession that runs `turn_behaviors` one per turn.

    Each behavior is either a string (the next command that turn returns) or an
    Exception instance to raise. When the list is exhausted, KeyboardInterrupt
    is raised to terminate the otherwise-infinite loop.
    """
    display = FakeDisplay()
    monkeypatch.setattr(gs, "DisplayManager", lambda: display)

    commands_played = []
    behaviors = list(turn_behaviors)

    session = object.__new__(GameSession)
    session.logger = SimpleNamespace(
        logger=logging.getLogger("test_game_session_recovery"),
        log_error=lambda message: None,
    )
    session.turn_number = 0
    session._background_tasks = []
    session.zork_service = SimpleNamespace(
        play_turn=lambda *args, **kwargs: asyncio.sleep(0)
    )

    async def fake_bootstrap():
        return None

    session._bootstrap_inventory = fake_bootstrap

    async def fake_play_turn(input_text, display_arg):
        commands_played.append(input_text)
        session.turn_number += 1
        if not behaviors:
            raise KeyboardInterrupt
        behavior = behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        return behavior

    # play() calls the name-mangled private method.
    session._GameSession__play_turn = fake_play_turn

    return session, commands_played, display


def test_failed_turn_recovers_with_fallback_command(monkeypatch):
    session, commands, display = _make_session(
        ["NORTH", ValueError("to_location Field required"), "SOUTH"],
        monkeypatch,
    )

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(session.play())

    # Turn 2 blew up; the session kept playing with the fallback command.
    assert commands == ["look", "NORTH", FALLBACK_COMMAND, "SOUTH"]
    assert display.stopped is True


def test_failure_streak_resets_after_a_good_turn(monkeypatch):
    failures = [RuntimeError("boom")] * (MAX_CONSECUTIVE_TURN_FAILURES - 1)
    session, commands, _ = _make_session(
        failures + ["NORTH"] + failures + ["SOUTH"],
        monkeypatch,
    )

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(session.play())

    # Non-consecutive failures never end the session.
    assert commands.count(FALLBACK_COMMAND) == 2 * (MAX_CONSECUTIVE_TURN_FAILURES - 1)


def test_sustained_failure_ends_the_session_cleanly(monkeypatch):
    session, commands, display = _make_session(
        [RuntimeError("ollama unreachable")] * (MAX_CONSECUTIVE_TURN_FAILURES + 5),
        monkeypatch,
    )

    # No exception: a dead backend ends the session, it doesn't spin forever.
    asyncio.run(session.play())

    assert len(commands) == MAX_CONSECUTIVE_TURN_FAILURES
    assert display.stopped is True


def test_bootstrap_failure_does_not_prevent_play(monkeypatch):
    session, commands, _ = _make_session(["NORTH"], monkeypatch)

    async def exploding_bootstrap():
        raise RuntimeError("inventory parse failed")

    session._bootstrap_inventory = exploding_bootstrap

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(session.play())

    assert commands == ["look", "NORTH"]


def test_turn_budget_timeout_is_recovered_not_fatal(monkeypatch):
    """The turn budget fires as asyncio.TimeoutError out of `asyncio.wait_for`
    in AdventurerService. It must cost a turn, not the run (issues #1 and #3).
    """
    import asyncio as _asyncio

    session, commands, _ = _make_session(
        ["NORTH", _asyncio.TimeoutError(), "SOUTH"],
        monkeypatch,
    )

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(session.play())

    assert commands == ["look", "NORTH", FALLBACK_COMMAND, "SOUTH"]

"""Run analysis (log -> progress, cost, and where it went wrong).

Reading a 50-turn session by scrolling does not scale, and the things worth
noticing — a score that stops moving, a command issued five times, a run whose
turns are mostly re-treading — are exactly what a human skims past. This also
becomes the experiment's reporting layer, so it needs to be right.
"""
import textwrap

import pytest

from tools.reporting.run_analysis import RunAnalysis, Turn, analyse


def write_log(tmp_path, body, session="test-session"):
    path = tmp_path / f"game_{session}.log"
    path.write_text(textwrap.dedent(body).strip() + "\n")
    return str(path)


LOG = """
2026-08-24 10:00:00,000 - INFO - ###  TURN 1 START - Command: look
2026-08-24 10:00:01,000 - INFO - Location: West Of House
2026-08-24 10:00:02,000 - INFO - Score: 0, Moves: 1
2026-08-24 10:00:03,000 - INFO - Game Response (first 100): West Of House
2026-08-24 10:01:00,000 - INFO - ###  TURN 2 START - Command: NORTH
2026-08-24 10:01:01,000 - INFO - Location: North of House
2026-08-24 10:01:02,000 - INFO - Score: 0, Moves: 2
2026-08-24 10:01:03,000 - INFO - Game Response (first 100): North of House
2026-08-24 10:01:04,000 - INFO - MEMORY STORED: [700/1000] open the mailbox
2026-08-24 10:02:00,000 - INFO - ###  TURN 3 START - Command: TAKE EGG
2026-08-24 10:02:01,000 - INFO - Location: Up A Tree
2026-08-24 10:02:02,000 - INFO - Score: 5, Moves: 3
2026-08-24 10:02:03,000 - INFO - Game Response (first 100): Taken.
2026-08-24 10:03:00,000 - INFO - ###  TURN 4 START - Command: NORTH
2026-08-24 10:03:01,000 - INFO - Location: North of House
2026-08-24 10:03:02,000 - INFO - Score: 5, Moves: 4
2026-08-24 10:03:03,000 - INFO - Game Response (first 100): North of House
"""


def test_turns_and_progress_are_extracted(tmp_path):
    run = analyse(write_log(tmp_path, LOG))

    assert len(run.turns) == 4
    assert run.final_score == 5
    assert [t.command for t in run.turns] == ["look", "NORTH", "TAKE EGG", "NORTH"]


def test_scoring_turns_are_the_only_unambiguous_progress(tmp_path):
    """The score is the one signal the game gives that is not interpretation."""
    run = analyse(write_log(tmp_path, LOG))

    assert [t.number for t in run.scoring_turns] == [3]


def test_wasted_turns_counts_provable_repeats(tmp_path):
    """Same room, same command, same response — no new information, by
    definition, since the game is deterministic."""
    run = analyse(write_log(tmp_path, LOG))

    # Turn 4 repeats turn 2 exactly.
    assert run.wasted_turns == 1


def test_turn_durations_come_from_the_gaps_between_starts(tmp_path):
    run = analyse(write_log(tmp_path, LOG))

    assert run.turns[0].seconds == 60
    assert run.total_seconds == pytest.approx(180)


def test_memories_and_events_are_collected(tmp_path):
    run = analyse(write_log(tmp_path, LOG))

    assert run.memories_stored == ["open the mailbox"]
    assert run.deaths == 0
    assert run.failures == 0


def test_repeated_commands_are_ranked(tmp_path):
    run = analyse(write_log(tmp_path, LOG))

    repeats = run.repeated_commands(minimum=2)
    assert ("North of House", "NORTH", 2) in [(l, c, n) for l, c, n in repeats]


def test_distinct_rooms_are_counted_case_insensitively(tmp_path):
    log = LOG + """
2026-08-24 10:04:00,000 - INFO - ###  TURN 5 START - Command: LOOK
2026-08-24 10:04:01,000 - INFO - Location: north of house
2026-08-24 10:04:02,000 - INFO - Score: 5, Moves: 5
2026-08-24 10:04:03,000 - INFO - Game Response (first 100): North of House
"""
    run = analyse(write_log(tmp_path, log))

    assert run.distinct_locations == 3


def test_deaths_and_failures_are_surfaced(tmp_path):
    log = LOG + """
2026-08-24 10:05:00,000 - INFO - ###  TURN 6 START - Command: NORTH
2026-08-24 10:05:01,000 - INFO - Location: Forest
2026-08-24 10:05:02,000 - INFO - Score: 5, Moves: 6
2026-08-24 10:05:03,000 - INFO - Game Response (first 100): dead
2026-08-24 10:05:04,000 - INFO - [MAPPER] Death detected; not mapping
2026-08-24 10:05:05,000 - ERROR - Turn 6 failed (1/3): boom
"""
    run = analyse(write_log(tmp_path, log))

    assert run.deaths == 1
    assert run.failures == 1


def test_summary_reports_cost_per_point_when_there_is_a_score():
    run = RunAnalysis(session_id="s")
    run.turns = [Turn(number=1, score=10, seconds=60)]
    run.tokens = {1: (900, 100, 6)}

    summary = run.summary()

    assert "tokens per point   100" in summary


def test_an_empty_log_does_not_crash(tmp_path):
    run = analyse(write_log(tmp_path, "2026-08-24 10:00:00,000 - INFO - nothing here"))

    assert run.turns == []
    assert run.final_score == 0
    assert "turns              0" in run.summary()

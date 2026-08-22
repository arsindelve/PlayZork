"""Map writes must be correctable (GitHub issue #11).

`map_transitions` is UNIQUE(session_id, from_location, direction), so there is
exactly one row per passage. A plain INSERT made the FIRST observation
permanent: Living Room --[DOWN]--> BLOCKED, recorded before the rug was moved,
could never be corrected once the trap door opened, and the entire underground
became unreachable for the rest of the session.

These tests use a REAL DatabaseManager on a tmp_path DB. The MockDatabase in
test_pathfinder.py / test_directions.py has no UNIQUE constraint, so it cannot
observe upsert behaviour at all.
"""
import pytest

from tools.database import DatabaseManager
from tools.mapping.mapper_state import MapperState


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def mapper(db):
    return MapperState(session_id="test_session", db=db)


class TestSuccessCorrectsBlocked:
    """A real destination always overwrites a stale BLOCKED row."""

    def test_trap_door_route_is_learned_after_a_failed_attempt(self, mapper):
        mapper.record_movement("Living Room", "BLOCKED", "DOWN", 5)
        assert mapper.pathfinder.find_path("Living Room", "Cellar") is None

        mapper.record_movement("Living Room", "Cellar", "DOWN", 40)

        assert mapper.get_exits_from("Living Room") == [("DOWN", "Cellar")]
        assert mapper.pathfinder.find_path("Living Room", "Cellar") == ["DOWN"]

    def test_correction_reports_that_the_map_changed(self, mapper):
        mapper.record_movement("Living Room", "BLOCKED", "DOWN", 5)
        assert mapper.record_movement("Living Room", "Cellar", "DOWN", 40) is True

    def test_correction_does_not_duplicate_the_row(self, mapper, db):
        mapper.record_movement("Living Room", "BLOCKED", "DOWN", 5)
        mapper.record_movement("Living Room", "Cellar", "DOWN", 40)
        assert len(db.get_all_transitions("test_session")) == 1

    def test_a_wrong_destination_can_be_corrected(self, mapper):
        mapper.record_movement("A", "B", "NORTH", 1)
        assert mapper.record_movement("A", "C", "NORTH", 2) is True
        assert mapper.get_exits_from("A") == [("NORTH", "C")]

    def test_correction_updates_turn_discovered(self, mapper, db):
        mapper.record_movement("Living Room", "BLOCKED", "DOWN", 5)
        mapper.record_movement("Living Room", "Cellar", "DOWN", 40)
        assert db.get_all_transitions("test_session")[0][3] == 40


class TestBlockedNeverOverwritesAPassage:
    """BLOCKED is inferred, not observed -- it must not destroy a real edge."""

    def test_a_closing_trap_door_keeps_the_known_edge(self, mapper):
        mapper.record_movement("Living Room", "Cellar", "DOWN", 40)
        assert mapper.record_movement("Living Room", "BLOCKED", "DOWN", 50) is False
        assert mapper.get_exits_from("Living Room") == [("DOWN", "Cellar")]
        assert mapper.pathfinder.find_path("Living Room", "Cellar") == ["DOWN"]

    def test_a_false_blocked_from_object_manipulation_is_survivable(self, mapper):
        """MOVE RUG mis-extracts as EAST (#10); the real EAST exit must survive."""
        mapper.record_movement("Living Room", "Kitchen", "EAST", 3)
        mapper.record_movement("Living Room", "BLOCKED", "EAST", 6)  # MOVE RUG
        assert mapper.get_exits_from("Living Room") == [("EAST", "Kitchen")]

    def test_a_false_blocked_recorded_first_is_still_recoverable(self, mapper):
        mapper.record_movement("Living Room", "BLOCKED", "EAST", 6)  # MOVE RUG
        mapper.record_movement("Living Room", "Kitchen", "EAST", 7)
        assert mapper.get_exits_from("Living Room") == [("EAST", "Kitchen")]


class TestIdempotentWrites:
    """Re-walking a known passage changes nothing."""

    def test_rewalking_a_passage_is_a_no_op(self, mapper):
        mapper.record_movement("A", "B", "NORTH", 1)
        assert mapper.record_movement("A", "B", "NORTH", 99) is False

    def test_rewalking_preserves_turn_discovered(self, mapper, db):
        mapper.record_movement("A", "B", "NORTH", 1)
        mapper.record_movement("A", "B", "NORTH", 99)
        assert db.get_all_transitions("test_session")[0][3] == 1

    def test_repeated_blocked_attempts_stay_one_row(self, mapper, db):
        mapper.record_movement("A", "BLOCKED", "WEST", 1)
        assert mapper.record_movement("A", "BLOCKED", "WEST", 7) is False
        assert len(db.get_all_transitions("test_session")) == 1


class TestWritesStayScoped:
    def test_sessions_do_not_correct_each_other(self, db):
        a = MapperState(session_id="A", db=db)
        b = MapperState(session_id="B", db=db)
        a.record_movement("Living Room", "BLOCKED", "DOWN", 5)
        b.record_movement("Living Room", "Cellar", "DOWN", 5)
        assert a.get_exits_from("Living Room") == [("DOWN", "BLOCKED")]
        assert b.get_exits_from("Living Room") == [("DOWN", "Cellar")]

    def test_abbreviation_and_full_name_are_one_passage(self, mapper, db):
        """#9's write-boundary normalisation still holds under upsert."""
        mapper.record_movement("Z", "Y", "N", 1)
        assert mapper.record_movement("Z", "Y", "NORTH", 2) is False
        assert len(db.get_all_transitions("test_session")) == 1


class TestRealErrorsAreNotMasked:
    def test_a_null_destination_raises_instead_of_returning_false(self, db):
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            db.add_map_transition(
                session_id="test_session",
                from_location="A",
                to_location=None,
                direction="NORTH",
                turn_number=1,
            )

"""Telling apart rooms that share a display name (#15).

Zork has several rooms called "Forest" and an entire maze where every room
reports the same name. Identifying a room by that name alone collapses them
into one node whose exits are the union of all of them — a fictional
super-room, in the one region where the original game requires careful
mapping.

Captured live (2026-08-24): moving EAST from one "Forest" reaches a different
"Forest", and the exits array changes from [3,2,1] to [3,0,1] while the
reported name does not.
"""
from typing import List, Tuple

import pytest

from tools.mapping.mapper_state import MapperState
from tools.mapping.room_identity import RoomRegistry, exits_signature, is_compatible


# ---------------------------------------------------------------------------
# The discriminator
# ---------------------------------------------------------------------------


def test_two_forests_with_different_exits_are_different_rooms():
    r = RoomRegistry()

    first = r.resolve("Forest", [3, 2, 1])
    second = r.resolve("Forest", [3, 0, 1])

    assert first == "Forest"
    assert second == "Forest #2"
    assert r.distinct_count("Forest") == 2


def test_returning_to_the_same_forest_reuses_its_label():
    r = RoomRegistry()

    first = r.resolve("Forest", [3, 2, 1])
    r.resolve("Forest", [3, 0, 1])
    again = r.resolve("Forest", [3, 2, 1])

    assert again == first


def test_a_uniquely_named_room_keeps_its_bare_name():
    """The overwhelmingly common case must be unchanged, so existing maps,
    prompts and reports read exactly as before."""
    r = RoomRegistry()

    assert r.resolve("West Of House", [1, 0, 3]) == "West Of House"


def test_exit_order_does_not_matter():
    r = RoomRegistry()

    assert r.resolve("Forest", [3, 2, 1]) == r.resolve("Forest", [1, 2, 3])


def test_signature_is_hashable_and_deduplicated():
    assert exits_signature([3, 2, 1, 2]) == (1, 2, 3)
    assert exits_signature(None) == ()
    assert exits_signature(["nonsense"]) == ()


# ---------------------------------------------------------------------------
# Bias: merge rather than split
# ---------------------------------------------------------------------------


def test_a_room_that_gains_an_exit_is_still_the_same_room():
    """Opening the trap door adds a DOWN exit to the Living Room. Demanding
    equality would split it in two and the halves could never be connected."""
    r = RoomRegistry()

    before = r.resolve("Living Room", [2])
    after = r.resolve("Living Room", [2, 11])       # trap door now open

    assert after == before
    assert r.distinct_count("Living Room") == 1


def test_the_widened_exit_set_is_remembered():
    r = RoomRegistry()
    r.resolve("Living Room", [2])
    r.resolve("Living Room", [2, 11])

    # Coming back with only the original exits must still match.
    assert r.resolve("Living Room", [2, 11]) == "Living Room"
    assert r.distinct_count("Living Room") == 1


def test_an_unknown_discriminator_never_causes_a_split():
    """A backend that reports no exits must degrade to the pre-#15 behaviour,
    never to a fragmented map."""
    r = RoomRegistry()

    assert r.resolve("Forest", None) == "Forest"
    assert r.resolve("Forest", []) == "Forest"
    assert r.distinct_count("Forest") == 1


def test_compatibility_is_containment_not_equality():
    assert is_compatible((1, 2), (1, 2, 3))
    assert is_compatible((1, 2, 3), (1, 2))
    assert is_compatible((), (1, 2))
    assert not is_compatible((1, 2), (3, 4))


def test_a_missing_name_resolves_to_nothing():
    r = RoomRegistry()

    assert r.resolve(None, [1]) == ""
    assert r.resolve("", [1]) == ""


# ---------------------------------------------------------------------------
# Through the mapper
# ---------------------------------------------------------------------------


class MockDatabase:
    def __init__(self):
        self.transitions: List[Tuple[str, str, str, int]] = []

    def add_map_transition(self, session_id, from_location, to_location, direction, turn_number):
        for i, (f, _, d, _) in enumerate(self.transitions):
            if (f, d) == (from_location, direction):
                self.transitions[i] = (from_location, to_location, direction, turn_number)
                return True
        self.transitions.append((from_location, to_location, direction, turn_number))
        return True

    def get_all_transitions(self, session_id):
        return self.transitions

    def get_transitions_from_location(self, session_id, location):
        return [(d, t) for f, t, d, _ in self.transitions if f == location]


@pytest.fixture
def mapper():
    return MapperState(session_id="s", db=MockDatabase())


def test_the_captured_forest_to_forest_move_is_mapped_as_two_rooms(mapper, ):
    """The live capture: EAST from one Forest to another, name unchanged."""
    mapper.previous_location = None
    mapper.update_from_turn("Forest", "look", 1, game_response="A forest.", exits=[3, 2, 1])
    mapper.update_from_turn("Forest", "EAST", 2, game_response="A forest.", exits=[3, 0, 1])

    edges = mapper.db.transitions
    assert edges, "the passage between the two Forests was not recorded"
    assert edges[0][0] == "Forest"
    assert edges[0][1] == "Forest #2", "both Forests collapsed into one node"


def test_no_false_wall_is_written_between_same_named_rooms(mapper):
    """The original #15 symptom: a successful move recorded as BLOCKED."""
    mapper.previous_location = None
    mapper.update_from_turn("Forest", "look", 1, game_response="A forest.", exits=[3, 2, 1])
    mapper.update_from_turn("Forest", "EAST", 2, game_response="A forest.", exits=[3, 0, 1])

    assert not any(t[1] == "BLOCKED" for t in mapper.db.transitions)


def test_a_backend_without_exits_behaves_exactly_as_before(mapper):
    mapper.previous_location = "West Of House"
    mapper.update_from_turn("North of House", "NORTH", 1, game_response="ok")

    assert mapper.db.transitions == [("West Of House", "North of House", "NORTH", 1)]

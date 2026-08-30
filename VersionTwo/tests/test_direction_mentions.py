"""Whole-word direction mentions in room prose (GitHub issue #8).

Substring containment scored "NE" inside CORNER, "SE" inside HOUSE and "SW"
inside SWORD. A fabricated mention is not cosmetic: ExplorerAgent
._pick_best_direction returns mentioned_directions[0] ahead of the entire
cardinal->diagonal->up/down priority scheme, and _calculate_confidence adds
+20, pinning the proposal at the 95 cap.
"""
import pytest

from tools.agent_graph.explorer_agent import ExplorerAgent
from tools.mapping.directions import (
    CANONICAL_DIRECTIONS,
    DIRECTION_PROSE_ALIASES,
    find_mentioned_directions,
)

# --- Room prose: (name, text, directions the prose actually offers) ---------
# Zork I descriptions. West of House is the text recorded in
# data/zork_sessions.db; the rest are the canonical room descriptions.
ROOMS = [
    ("West of House",
     "West Of House\nYou are standing in an open field west of a white house, "
     "with a boarded front door. \nThere is a small mailbox here.",
     {"WEST"}),
    ("Behind House",
     "Behind House\nYou are behind the white house. A path leads into the forest "
     "to the east. In one corner of the house there is a small window which is "
     "slightly ajar.",
     {"EAST"}),
    ("Kitchen",
     "Kitchen\nYou are in the kitchen of the white house. A table seems to have "
     "been used recently for the preparation of food. A passage leads to the west "
     "and a dark staircase can be seen leading upward. A dark chimney leads down "
     "and to the east is a small window which is open.",
     {"WEST", "EAST", "UP", "DOWN"}),
    ("Living Room",
     "Living Room\nYou are in the living room. There is a doorway to the east, a "
     "wooden door with strange gothic lettering to the west, which appears to be "
     "nailed shut, a trophy case, and a large oriental rug in the center of the room.",
     {"EAST", "WEST"}),
    ("Attic",
     "Attic\nThis is the attic. The only exit is a stairway leading down.\n"
     "A large coil of rope is lying in the corner.\nOn a table is a nasty-looking knife.",
     {"DOWN"}),
    ("Cellar",
     "Cellar\nYou are in a dark and damp cellar with a narrow passageway leading "
     "north, and a crawlway to the south. On the west is the bottom of a steep "
     "metal ramp which is unclimbable.",
     {"NORTH", "SOUTH", "WEST"}),
    ("The Troll Room",
     "The Troll Room\nThis is a small room with passages to the east and south and "
     "a forbidding hole leading west. Bloodstains and deep scratches (perhaps made "
     "by an axe) mar the walls.",
     {"EAST", "SOUTH", "WEST"}),
    ("Gallery",
     "Gallery\nThis is an art gallery. Most of the paintings have been stolen by "
     "vandals with exceptional taste. The vandals left through either the north or "
     "west exits.\nFortunately, there is still one chance for you to be a vandal, "
     "for on the far wall is a painting of unparalleled beauty.",
     {"NORTH", "WEST"}),
    ("Grating Room",
     "Grating Room\nYou are in a small room near the maze. There are twisty "
     "passages in the immediate vicinity.",
     set()),
    ("Forest",
     "Forest\nThis is a forest, with trees in all directions. To the east, there "
     "appears to be sunlight.",
     {"EAST"}),
]

DIAGONALS = {"NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST"}


@pytest.mark.parametrize("name,text,expected", ROOMS, ids=[r[0] for r in ROOMS])
def test_no_diagonal_is_invented_from_ordinary_prose(name, text, expected):
    """CORNER/HOUSE/CASE/SWORD/NEAR must never produce a diagonal."""
    found = set(find_mentioned_directions(text))
    assert not (found & DIAGONALS), f"{name}: fabricated {sorted(found & DIAGONALS)}"


@pytest.mark.parametrize("name,text,expected", ROOMS, ids=[r[0] for r in ROOMS])
def test_real_exits_are_still_found(name, text, expected):
    """The fix must not cost recall."""
    found = set(find_mentioned_directions(text))
    assert expected <= found, f"{name}: lost {sorted(expected - found)}"


@pytest.mark.parametrize("word,direction", [
    ("corner", "NORTHEAST"), ("one", "NORTHEAST"), ("near", "NORTHEAST"),
    ("nest", "NORTHEAST"), ("chimney", "NORTHEAST"), ("darkness", "NORTHEAST"),
    ("house", "SOUTHEAST"), ("case", "SOUTHEAST"), ("seems", "SOUTHEAST"),
    ("staircase", "SOUTHEAST"), ("used", "SOUTHEAST"),
    ("sword", "SOUTHWEST"), ("answer", "SOUTHWEST"), ("switch", "SOUTHWEST"),
    ("cup", "UP"), ("occupied", "UP"),
    ("least", "EAST"), ("feast", "EAST"),
])
def test_direction_is_not_matched_inside_a_longer_word(word, direction):
    assert direction not in find_mentioned_directions(f"There is a {word} here.")


def test_compound_direction_does_not_also_report_its_halves():
    """'leads northeast' used to report NORTH, EAST and NORTHEAST — and
    _pick_best_direction then took NORTH, the first in canonical order."""
    assert find_mentioned_directions("A path leads northeast.") == ["NORTHEAST"]
    assert find_mentioned_directions("The passage runs southwest.") == ["SOUTHWEST"]


def test_hyphenated_pair_reports_both_halves():
    assert find_mentioned_directions("The path heads north-south here.") == ["NORTH", "SOUTH"]


@pytest.mark.parametrize("text", [
    "Go N.", "N leads onward", "The path heads N", "Exits: N, S, E, W",
    "(N)", "go n!", "Only exit: N",
])
def test_single_letter_aliases_match_at_edges_and_before_punctuation(text):
    """The old ' N ' padding missed every one of these."""
    assert "NORTH" in find_mentioned_directions(text)


@pytest.mark.parametrize("text", [
    "Beside you on the branch is a small bird's nest.",
    "The dam's control panel is here.",
    "There's a grating underneath.",
])
def test_possessive_apostrophe_is_not_a_direction(text):
    """Plain \\b would match the "S" in "bird's" -> SOUTH."""
    assert "SOUTH" not in find_mentioned_directions(text)


def test_candidates_filter_restricts_the_result():
    text = "A passage leads west and a staircase leads up."
    assert find_mentioned_directions(text) == ["WEST", "UP"]
    assert find_mentioned_directions(text, ["UP", "DOWN"]) == ["UP"]
    assert find_mentioned_directions(text, []) == []


def test_result_is_in_canonical_order():
    found = find_mentioned_directions("up, west, north, down, east")
    assert found == [d for d in CANONICAL_DIRECTIONS if d in found]


@pytest.mark.parametrize("text", [None, "", "Taken.", "You can't go that way.",
                                  "It is pitch black. You are likely to be eaten by a grue.",
                                  "Opening the small mailbox reveals a leaflet."])
def test_texts_with_no_direction_yield_nothing(text):
    assert find_mentioned_directions(text) == []


def test_every_prose_alias_matches_its_own_direction():
    for direction, aliases in DIRECTION_PROSE_ALIASES.items():
        for alias in aliases:
            found = find_mentioned_directions(f"You may go {alias.lower()} from here.")
            assert direction in found, f"{alias!r} did not match {direction}"


def test_prose_alias_table_covers_exactly_the_canonical_directions():
    assert sorted(DIRECTION_PROSE_ALIASES) == sorted(CANONICAL_DIRECTIONS)


# --- Downstream: the reason this matters ------------------------------------

def _explorer(text, explored):
    unexplored = [d for d in CANONICAL_DIRECTIONS if d not in explored]
    agent = ExplorerAgent(
        current_location="X",
        unexplored_directions=unexplored,
        mentioned_directions=find_mentioned_directions(text, unexplored),
        turn_number=1,
    )
    return agent, agent._calculate_confidence(agent.best_direction)


def test_attic_does_not_propose_northeast_at_max_confidence():
    """The Attic's only exit is DOWN. "corner" used to yield NORTHEAST at 95."""
    text = ("Attic\nThis is the attic. The only exit is a stairway leading down.\n"
            "A large coil of rope is lying in the corner.")
    agent, confidence = _explorer(text, explored={"DOWN"})
    assert agent.best_direction not in DIAGONALS
    assert confidence == 75, "no mention -> no +20 bonus"


def test_west_of_house_prefers_the_cardinal_over_a_fabricated_diagonal():
    """With N/S/W walked, "house" used to make SOUTHEAST beat EAST 95 to 75."""
    text = ("West Of House\nYou are standing in an open field west of a white "
            "house, with a boarded front door. \nThere is a small mailbox here.")
    agent, confidence = _explorer(text, explored={"NORTH", "SOUTH", "WEST"})
    assert agent.best_direction == "EAST"
    assert confidence == 75


def test_a_genuine_mention_still_wins_and_still_earns_the_bonus():
    text = "A narrow passageway leads north from here."
    agent, confidence = _explorer(text, explored=set())
    assert agent.best_direction == "NORTH"
    assert confidence == 95

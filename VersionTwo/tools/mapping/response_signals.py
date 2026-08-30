"""What the game's prose says about the move we just made (GitHub issue #12).

The mapper infers edges from state alone: "the room name changed, so the
command was a passage". Death breaks that inference. Zork kills you, prints an
obituary, then *teleports* you to the Forest — and the backend reports the
RESPAWN room as this turn's LocationName. Verified against the live API
(2026-08-22):

    command  : "north"          (from the Cellar, into the troll)
    response : "...I'm afraid you are dead. \n\n\t*** You have died ***\n\n
                Now, let's take a look here...\n\nForest\nThis is a forest..."
    location : "Forest"         <- NOT empty, so the #7 unnamed-room guard
                                   does not catch this
    score    : 35 -> 25

The state-only rule therefore records `Cellar --[NORTH]--> Forest`, and BFS
routes every later journey through the move that kills you.

Bias, deliberately asymmetric: a false positive costs ONE map edge, which the
next survivable move re-records. A false negative writes a fabricated edge
that (since #11) also OVERWRITES the true destination for that direction. So
this predicate is tuned to over-detect.

Deliberately dependency-free, like `directions` and `locations`.
"""
import re
from typing import Optional

# Infocom's death/ending banner: "    *** You have died ***". Asterisk runs and
# spacing vary between games and releases ("**** You have died ****"), so match
# the shape and look for a death word inside it rather than one exact string.
_BANNER_RE = re.compile(r"\*{2,}([^*\n]{1,120})\*{2,}")
_BANNER_DEATH_WORDS = ("died", "dead", "killed", "slain", "perished")

# Games that skip the banner still say it in prose. Kept narrow enough that the
# common non-death uses of "dead" do not match: "the troll is dead", "a dead
# end" and "Land of the Dead" all lack a second-person subject.
_DEATH_PHRASES = (
    "you have died",
    "you died",
    "you are dead",
    "you're dead",
    "you have been killed",
    "you have been slain",
    "you have been eaten",
    "your adventure is over",
)


def looks_like_death(game_response: Optional[str]) -> bool:
    """True when this turn's prose reports the PLAYER dying.

    Deterministic on purpose. `DeathAnalyzer` already asks an LLM the same
    question, but it runs inside the post-turn BACKGROUND task — strictly
    after `mapper_toolkit.update_after_turn` has already written the edge. The
    mapper needs the answer on the critical path, for free, before it writes.
    """
    if not game_response:
        return False
    text = game_response.casefold()

    for banner in _BANNER_RE.findall(text):
        if any(word in banner for word in _BANNER_DEATH_WORDS):
            return True

    return any(phrase in text for phrase in _DEATH_PHRASES)


# ---------------------------------------------------------------------------
# Did the game REFUSE the move? (GitHub issues #10, #15)
# ---------------------------------------------------------------------------
#
# The mapper used to infer "the move failed" from room NAMES alone: same name
# before and after => BLOCKED. Zork has several rooms called "Forest" and a
# maze of identically-named rooms, so successful moves were recorded as
# permanent walls (#15). Verified against the live backend: EAST from one
# "Forest" reaches a different "Forest" while the reported LocationName never
# changes.
#
# This flips the default from "assume wall" to "assume unknown": BLOCKED is
# written only when the text explicitly refuses the movement.
#
# Only GENERIC refusals are listed, and that is a feature. Object-specific
# refusals — "The trap door is closed.", "The troll fends you off." — describe
# TEMPORARY obstacles, exactly the puzzle states that must not be frozen into
# the map. The phrases we cannot match reliably are the phrases we should not
# record.
#
# NOTE the opposite bias to looks_like_death above, and why they stay separate
# functions rather than one classifier: death must OVER-detect (a false
# positive drops one self-healing edge; a false negative writes an edge that
# since #11 also destroys the true destination), while a refusal must
# UNDER-detect (a false positive fabricates a wall the explorer then treats as
# explored and never retries). One shared enum would force a single bias on
# both.

# "cannot" / "can't" / "can’t" / "cant" -- the live backend emits both the
# contracted and the spelled-out form, from the same session.
_CANT = r"can(?:no|['\u2019])?t"

_MOVEMENT_REFUSALS = (
    re.compile(rf"\byou\s+{_CANT}\s+go\s+that\s+way", re.IGNORECASE),
    re.compile(rf"\byou\s+{_CANT}\s+get\s+there\s+from\s+here", re.IGNORECASE),
    re.compile(r"\bthere\s+is\s+a\s+wall\s+there", re.IGNORECASE),
    # Terrain refusals. Added from observed play (#33): "The forest becomes
    # impenetrable to the north." is as permanent as "you cannot go that way",
    # but matched none of the generic patterns, so the explorer never learned
    # the wall and re-proposed NORTH until the session deadlocked.
    #
    # These stay in the ALLOW-list only because terrain is topology. They must
    # not be confused with object refusals ("The trap door is closed."), which
    # describe temporary puzzle state and are deliberately still unmatched.
    #
    # Only the vegetation family is listed, because that is what was actually
    # observed. A tempting "impassable + mountains" pattern was drafted and
    # REMOVED: Zork's Forest room DESCRIPTION reads "The forest thins out,
    # revealing impassible mountains." — scenery on a SUCCESSFUL move. Matching
    # it would fabricate a wall on arrival, which is the exact class of error
    # #10 exists to prevent. Verified phrasings only.
    re.compile(r"\b(?:forest|undergrowth|trees?|foliage|brush)\b[^.]*\b"
               r"(?:impenetrable|impassable|too\s+(?:thick|dense))", re.IGNORECASE),
)


def is_movement_refusal(game_response: Optional[str]) -> bool:
    """True when the response explicitly says the player did not move.

    Unrecognized text returns False, so an unfamiliar backend records no
    BLOCKED edges at all rather than fabricating them.
    """
    if not game_response:
        return False
    return any(p.search(game_response) for p in _MOVEMENT_REFUSALS)

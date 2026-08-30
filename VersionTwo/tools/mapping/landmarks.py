"""Landmarks named in room text that are not places on the map.

The frontier was first defined as *map nodes reached but not departed from*.
Measured against a live run, that set is EMPTY on a linear exploration path:
the agent walks A->B->C, so every room it reached it also left. The one signal
meant to redirect it toward the objective was unavailable exactly when needed.

The useful frontier is not in the map at all. Zork's opening reads "You are
standing in an open field west of a white house, with a boarded front door" —
and the white house is the entire early game. It is not a map node. It is a
noun in a description, and nothing in the pipeline turns it into a destination.

This extracts those nouns deterministically: no LLM call, no new latency.

WHY A CLOSED VOCABULARY. Extracting "interesting nouns" in general needs a
parser or a model. Both would surface furniture, scenery and abstractions, and
a wrong landmark sends the agent chasing something that does not exist. A fixed
list of *enterable structures* cannot do that. It will miss landmarks no list
anticipates — which costs only what today already costs, an empty frontier.
A false negative beats a false positive, as everywhere else in the world model.

WHY RECENCY DECIDES "VISITED", NOT NAME MATCHING. The tempting test is whether
a location name contains the landmark, but every candidate rule fails:

  - substring: "West of House" contains "house", so standing OUTSIDE the house
    would mark it visited and suppress the one lead worth having.
  - strip the positional prefix: "West of House" -> "House" -> same failure.
  - require exact equality: nothing is ever named "House", so entering through
    the window into the Kitchen never clears it, and it nags forever.

The room text settles it without any of them. While the agent is outside, the
description keeps saying "white house" and the lead keeps surfacing. Once it is
in the Kitchen, the description stops mentioning a house and the landmark ages
out of the recency window on its own. The world model does not need to know
what "inside" means.
"""
import re
from typing import Iterable, List, Optional, Set

# Structures a player can plausibly enter, approach or open. Deliberately not
# scenery: "tree", "path" and "forest" are terrain the explorer already covers
# by direction, and listing them would drown the real leads.
LANDMARK_NOUNS = frozenset({
    "house", "building", "tower", "castle", "temple", "church", "hut",
    "cabin", "shed", "barn", "mill", "cave", "tunnel", "bridge", "gate",
    "gateway", "entrance", "doorway", "door", "window", "stairs", "stairway",
    "staircase", "ladder", "well", "shaft", "chimney", "trapdoor", "hatch",
    "grating", "passage", "corridor", "boat", "dam", "vault", "crypt",
    "tomb", "maze", "altar", "chasm", "ledge", "cellar", "attic", "basement",
})

# Words that are never part of a landmark's name. Without this "west of a
# white house" yields "of a white house" as the phrase.
_NOT_ADJECTIVES = frozenset({
    "of", "in", "on", "at", "to", "by", "with", "from", "into", "onto",
    "and", "or", "but", "is", "are", "was", "were", "be", "been", "the",
    "a", "an", "this", "that", "these", "those", "there", "here", "you",
    "your", "it", "its", "his", "her", "their", "which", "who", "what",
    "cannot", "can", "not", "no", "nothing", "seems", "appears",
})

_NOUNS_ALT = "|".join(sorted(LANDMARK_NOUNS, key=len, reverse=True))

# Up to two adjectives, BOTH quantifiers lazy. A greedy quantifier here once
# turned "a small mailbox here." into a phrase running past the noun; lazy
# matching keeps the phrase minimal and the adjectives are validated below.
_LANDMARK = re.compile(
    rf"\b(?:a|an|the)\s+((?:[a-z][a-z'-]*\s+){{0,2}}?({_NOUNS_ALT}))\b",
    re.IGNORECASE,
)


def _clean(phrase: str) -> str:
    return " ".join(phrase.split()).casefold()


def find_landmarks(text: Optional[str]) -> List[str]:
    """Landmark phrases named in `text`, in order of first appearance.

    Returns e.g. ["white house", "boarded front door"] for Zork's opening.
    Never raises: a malformed description must not cost a turn.
    """
    if not text:
        return []
    found: List[str] = []
    seen: Set[str] = set()
    try:
        for match in _LANDMARK.finditer(text):
            phrase, noun = _clean(match.group(1)), _clean(match.group(2))
            words = phrase.split()
            # Drop adjectives that are really prepositions or determiners
            # leaking in from the surrounding sentence.
            while words and words[0] in _NOT_ADJECTIVES:
                words.pop(0)
            if not words:
                continue
            phrase = " ".join(words)
            if phrase in seen:
                continue
            seen.add(phrase)
            found.append(phrase)
    except Exception:  # noqa: BLE001 - extraction must never cost a turn
        return found
    return found


def unvisited_landmarks(
    recent_text: Optional[str],
    visited_locations: Iterable[str],
) -> List[str]:
    """Landmarks named in recent room text that we have not been inside.

    Two things retire a landmark, and neither is a name-containment test:

    1. A visited room is named for it — the game calls a room "Cellar", so
       "the cellar" is somewhere we have stood. Exact match only: "West of
       House" is not called "house" and retires nothing, which is the whole
       point.
    2. A visited room is named in the SAME SENTENCE. Zork's Kitchen reads "You
       are in the kitchen of the white house", so standing inside still names
       the house; without this the lead would nag for the rest of the run.
       Sentence scope is what makes this safe — "west of a white house" never
       contains the location name "West of House", so being outside retires
       nothing.

    Recency does the rest: descriptions that stop mentioning a landmark let it
    age out without the world model ever having to model "inside".
    """
    names = {_clean(n) for n in (visited_locations or []) if n}
    out: List[str] = []
    seen: Set[str] = set()
    # Split on NEWLINES as well as sentence terminators. The game puts the
    # room name on its own line above the description -- "West Of House\nYou
    # are standing in an open field west of a white house" -- with no period
    # after it, so a sentence-only split fuses the two and the location name
    # lands in the same "sentence" as the landmark, retiring the single lead
    # that matters. Seen live twice before the cause was found.
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", recent_text or ""):
        inside = any(re.search(rf"\b{re.escape(n)}\b", sentence, re.IGNORECASE)
                     for n in names if n)
        for phrase in find_landmarks(sentence):
            if phrase in seen:
                continue
            seen.add(phrase)
            noun = phrase.split()[-1]
            if inside or phrase in names or noun in names:
                continue
            out.append(phrase)
    return out

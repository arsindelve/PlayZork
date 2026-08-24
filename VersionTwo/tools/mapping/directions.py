"""Canonical direction vocabulary shared by the mapper and the explorer.

The game and the LLM both use abbreviations ("N") and full names ("NORTH")
interchangeably, and several prompts actively push the model toward the
shortest form. The mapper stored whatever arrived, while the explored-check
compared against full names only — so a move recorded as "N" left NORTH
looking unexplored forever and the ExplorerAgent re-proposed it every turn
(GitHub issue #9). `map_transitions` is UNIQUE(session_id, from_location,
direction), so "N" and "NORTH" are two distinct rows: the constraint cannot
dedupe them.

Deliberately dependency-free so `decision_graph` can import it without
pulling in DatabaseManager.
"""
import re
from typing import Iterable, List, Optional

CANONICAL_DIRECTIONS = [
    "NORTH", "SOUTH", "EAST", "WEST",
    "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST",
    "UP", "DOWN",
]

DIRECTION_ABBREVIATIONS = {
    "N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST",
    "NE": "NORTHEAST", "NW": "NORTHWEST", "SE": "SOUTHEAST", "SW": "SOUTHWEST",
    "U": "UP", "D": "DOWN",
    # Planetfall's ship uses nautical directions. They are ALIASES of compass
    # directions, not new ones: the backend itself translates them, verified by
    # direct probe against the live API on 2026-08-24 —
    #   "starboard" -> lastMovementDirection "E"   "port" -> "W"
    #   "fore"      -> "N"                         "aft"  -> "S"
    # so EAST and STARBOARD are one passage, not two. That is exactly why they
    # are NOT added to CANONICAL_DIRECTIONS: the explorer counts unexplored
    # directions out of that list and its expected value scales with the
    # count, so aliasing them as separate directions would both inflate its EV
    # and let it re-walk a passage it had already taken under the other name.
    "PORT": "WEST", "STARBOARD": "EAST", "FORE": "NORTH", "AFT": "SOUTH",
}


def normalize_direction(direction: Optional[str]) -> str:
    """Map any direction token to its canonical full name.

    Unknown tokens pass through uppercased and stripped, so this is safe to
    apply to arbitrary stored strings (including "BLOCKED" destinations and
    non-cardinal edges).
    """
    token = (direction or "").strip().upper()
    return DIRECTION_ABBREVIATIONS.get(token, token)


# ---------------------------------------------------------------------------
# Prose scanning (GitHub issue #8)
# ---------------------------------------------------------------------------
#
# Aliases the ExplorerAgent looks for in ROOM PROSE — as opposed to
# DIRECTION_ABBREVIATIONS above, which normalizes STORED map data. Matched on
# whole-word boundaries: bare substring containment found "NE" inside CORNER,
# "SE" inside HOUSE and "SW" inside SWORD, so 44% of "mentioned" hits on a real
# room corpus were fabricated. A fabricated mention is not cosmetic — it
# short-circuits the explorer's whole priority scheme (mentioned_directions[0]
# wins outright) and adds +20 confidence, which is exactly what pins the
# proposal at its 95 cap.
#
# Substring matching also meant "NORTH" matched inside "NORTHEAST": a room that
# explicitly said *northeast* sent the agent *north*.
DIRECTION_PROSE_ALIASES = {
    # FORE/AFT/PORT/STARBOARD: Planetfall describes every corridor this way, so
    # without them the explorer's "+2 mentioned" bonus was unavailable for every
    # LATERAL move on the ship while UP still earned it from "a gangway leads
    # up". Deck Nine scored UP 5 (exit+mentioned) against EAST 4
    # (exit+cardinal) and the agent climbed the ship instead of crossing it,
    # away from the escape pod. Observed in run pf-20260824.
    "NORTH":     ["NORTH", "NORTHERN", "NORTHWARD", "NORTHWARDS", "N", "FORE"],
    "SOUTH":     ["SOUTH", "SOUTHERN", "SOUTHWARD", "SOUTHWARDS", "S", "AFT"],
    "EAST":      ["EAST", "EASTERN", "EASTWARD", "EASTWARDS", "E", "STARBOARD"],
    "WEST":      ["WEST", "WESTERN", "WESTWARD", "WESTWARDS", "W", "PORT"],
    "NORTHEAST": ["NORTHEAST", "NORTH-EAST", "NORTHEASTERN", "NE"],
    "NORTHWEST": ["NORTHWEST", "NORTH-WEST", "NORTHWESTERN", "NW"],
    "SOUTHEAST": ["SOUTHEAST", "SOUTH-EAST", "SOUTHEASTERN", "SE"],
    "SOUTHWEST": ["SOUTHWEST", "SOUTH-WEST", "SOUTHWESTERN", "SW"],
    # ABOVE/BELOW are deliberately absent: across the room corpus they only
    # ever describe object placement or an out-of-reach branch, never an exit.
    "UP":        ["UP", "UPWARD", "UPWARDS", "U"],
    "DOWN":      ["DOWN", "DOWNWARD", "DOWNWARDS", "D"],
}

# Apostrophes are excluded from the boundary on purpose: a plain \b lets the
# single-letter alias "S" match the possessive in "bird'S nest".
_PROSE_PATTERNS = {
    direction: re.compile(
        r"(?<![\w'])(?:%s)(?![\w'])"
        % "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
    )
    for direction, aliases in DIRECTION_PROSE_ALIASES.items()
}


def find_mentioned_directions(text: Optional[str],
                              candidates: Optional[Iterable[str]] = None) -> List[str]:
    """Directions named as whole words in `text`, in CANONICAL_DIRECTIONS order.

    `candidates` restricts the search (the explorer passes its unexplored set).
    """
    if not text:
        return []
    haystack = text.upper()
    wanted = CANONICAL_DIRECTIONS if candidates is None else set(candidates)
    return [
        direction for direction in CANONICAL_DIRECTIONS
        if direction in wanted and _PROSE_PATTERNS[direction].search(haystack)
    ]


# ---------------------------------------------------------------------------
# Command parsing (GitHub issue #10)
# ---------------------------------------------------------------------------
#
# Verbs that may PRECEDE a direction word. "MOVE" is deliberately absent: in
# interactive fiction MOVE is object manipulation ("MOVE RUG" is a required
# Zork action), and the live backend agrees -- it leaves lastMovementDirection
# untouched for "move rug".
MOVEMENT_VERBS = frozenset({"GO", "WALK", "HEAD", "RUN"})

_COMMAND_WORD_RE = re.compile(r"[A-Z]+")


def is_direction(token: Optional[str]) -> bool:
    """True when `token` is a direction word or abbreviation, on its own."""
    t = (token or "").strip().upper()
    return t in DIRECTION_ABBREVIATIONS or t in CANONICAL_DIRECTIONS


def tokenize_command(command: Optional[str]) -> List[str]:
    """Split a command into uppercase alphabetic words, dropping punctuation."""
    return _COMMAND_WORD_RE.findall((command or "").upper())


def extract_direction(command: Optional[str]) -> Optional[str]:
    """Return the canonical direction a COMMAND asks for, or None.

    Matching is TOKEN-based, never substring-based (#10). The old extractor
    asked `if direction in command_upper` under a `startswith("MOVE ")` guard,
    so the "E" in the verb MOVE itself matched: every `MOVE <noun>` reported
    EAST-ish movement, and because object manipulation leaves the room
    unchanged the mapper wrote a false `--[E]--> BLOCKED` edge.

    Exactly two shapes are accepted:

      1. the whole command is one direction token   -- "N", "north.", "SOUTHWEST"
      2. a movement verb followed by one direction  -- "GO NORTH", "walk east"

    Anything else returns None. The bias is deliberate and asymmetric: a wrong
    direction writes a lie into the map (and #11 shows lies are hard to
    retract), while a missed direction costs at most an unrecorded edge.

    Sibling of `find_mentioned_directions`, which scans room PROSE; this one
    parses the player's COMMAND and is far stricter.
    """
    tokens = tokenize_command(command)

    if len(tokens) == 1 and is_direction(tokens[0]):
        return normalize_direction(tokens[0])

    if len(tokens) == 2 and tokens[0] in MOVEMENT_VERBS and is_direction(tokens[1]):
        return normalize_direction(tokens[1])

    return None


# ---------------------------------------------------------------------------
# Non-cardinal movement (GitHub issue #14)
# ---------------------------------------------------------------------------
#
# Zork moves the player with plain commands as often as with compass
# directions: CLIMB TREE, ENTER HOUSE, IN, OUT, CROSS BRIDGE, TOUCH MIRROR,
# PRAY. The mapper only understood cardinals, so those passages were never
# recorded and their destinations became orphan nodes — rooms the map knows
# exist but can never route *to*. The fix records the raw command as the edge
# label; BFS does not care that a label is not a compass point, and a label
# like "CLIMB TREE" is directly executable by the agent that receives it.

# Verbs that manipulate or inspect the world rather than move the player. A
# location change on one of these is a side effect — a death and respawn, a
# timed relocation — not a passage, and recording it would hand BFS a route
# nobody can walk. Meta commands are here too: they never move anyone.
#
# A bogus raw label is uncorrectable even after #11: the upsert only repairs a
# row when the same (from, direction) key is written again, and nobody ever
# re-issues "ATTACK TROLL WITH SWORD" from that room. That is why this is a
# deny-list rather than the bare "changed + no cardinal" rule.
NON_MOVEMENT_VERBS = frozenset({
    "TAKE", "GET", "DROP", "PUT", "OPEN", "CLOSE", "EXAMINE", "READ", "LOOK",
    "SEARCH", "MOVE", "PUSH", "PULL", "TURN", "LIGHT", "EXTINGUISH", "BURN",
    "ATTACK", "KILL", "FIGHT", "EAT", "DRINK", "FILL", "POUR", "TIE", "UNTIE",
    "LOCK", "UNLOCK", "WEAR", "WAVE", "THROW", "DIG", "KNOCK", "RING", "SMELL",
    "LISTEN", "COUNT", "ASK", "TELL", "ANSWER", "GIVE", "SHOW",
    "INVENTORY", "I", "WAIT", "Z", "AGAIN", "G",
    "SAVE", "RESTORE", "SCORE", "DIAGNOSE", "VERBOSE", "BRIEF", "SUPERBRIEF",
    "VERSION", "QUIT", "RESTART", "UNDO",
})

# Particles that turn a manipulation verb into a movement one: GET IN BOAT,
# GET OUT OF BOAT, CLIMB UP. Without this exemption the deny-list would throw
# away real passages.
MOVEMENT_PARTICLES = frozenset({
    "IN", "INTO", "OUT", "ON", "ONTO", "OFF", "UP", "DOWN",
    "OVER", "THROUGH", "ACROSS", "INSIDE", "OUTSIDE", "UNDER",
})

# An edge label is a command the agent will be asked to re-execute. Anything
# longer than this is a sentence, not a passage.
MAX_MOVEMENT_LABEL_WORDS = 4


def normalize_movement_command(command: Optional[str]) -> str:
    """Canonical edge label for a movement expressed as a raw command.

    Uppercases, collapses whitespace, and drops a leading GO/WALK so that
    "GO IN" and "IN" describe one passage rather than two rows under
    UNIQUE(session_id, from_location, direction).
    """
    tokens = (command or "").strip().upper().split()
    if len(tokens) > 1 and tokens[0] in ("GO", "WALK"):
        tokens = tokens[1:]
    return " ".join(tokens)


def is_probable_movement_command(command: Optional[str]) -> bool:
    """True when a command that changed the player's location should be mapped.

    The caller has already established that the room changed and that no
    compass direction could be extracted. That is strong evidence the command
    moved the player — this function only rejects the cases where something
    else plausibly did: object manipulation, combat and meta commands, which
    are the commands in flight when a death or a timed relocation fires.

    A deny-list rather than an allow-list on purpose: only the broad rule
    catches the game-specific teleports no verb list would contain — TOUCH
    MIRROR (a real Zork passage), PRAY, ODYSSEUS, Planetfall's hatch verbs.
    """
    tokens = normalize_movement_command(command).split()
    if not tokens or len(tokens) > MAX_MOVEMENT_LABEL_WORDS:
        return False
    if tokens[0] in NON_MOVEMENT_VERBS:
        # "GET IN BOAT" is movement; "GET LAMP" is not.
        return any(token in MOVEMENT_PARTICLES for token in tokens[1:])
    return True

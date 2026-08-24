"""Everything the agents need to know this turn, fetched in code (issue #25).

Every agent used to open its turn with a "research phase": a full LLM
round-trip whose instruction *named the exact tools to call*, executed them
once, and never fed the results back for another round. The dedicated research
node then repeated substantially the same fetches. On the measured turn that
was 176s of a 445s turn — a 14B model being asked for permission to run SQLite
queries.

There is no judgement in any of it. The instructions were already imperative
("REQUIRED: 1) Call get_direction_to_location(...) 2) ..."), so the code
always knew exactly what it wanted. Routing that through a model did not add
information; it added latency and three separate failure modes:

  * #4  — `tool_choice="any"` is ignored by ChatOllama, so the model could
          simply fetch nothing.
  * #5  — returned tool calls were dropped by narrower execution maps.
  * #6  — the model was told to call a tool that did not exist.

Fetching deterministically eliminates all three at once, and the data is
strictly better: it cannot be partially fetched, silently dropped, or
hallucinated.

Built once per turn and sliced per agent. Every read here is a local SQLite
query or an in-memory lookup — milliseconds in total.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tools.mapping.directions import (extract_direction,
                                      is_probable_movement_command)
from tools.mapping.landmarks import unvisited_landmarks
from tools.mapping.locations import is_known_location

# How many recent turns to put in front of an agent. Bounded deliberately:
# per-call latency scales with prompt size, and the 2026-08-22 checkpoint
# showed history-shaped prompt content is what drives turn-time growth.
RECENT_TURNS_FOR_AGENTS = 10


# Actions that undo each other. The backend's accepted-command list contains
# both halves of each pair — it is a grammar, not advice — and an agent that
# reads "close grating" as a suggestion will undo its own progress. Observed
# live: the agent closed the grating it was trying to open, at confidence 90.
_INVERSES = {
    "OPEN": "CLOSE", "CLOSE": "OPEN",
    "TAKE": "DROP", "GET": "DROP", "DROP": "TAKE",
    "LOCK": "UNLOCK", "UNLOCK": "LOCK",
    "LIGHT": "EXTINGUISH", "EXTINGUISH": "LIGHT",
    "WEAR": "REMOVE", "REMOVE": "WEAR",
    "ENTER": "EXIT", "EXIT": "ENTER",
    "RAISE": "LOWER", "LOWER": "RAISE",
    "TURN ON": "TURN OFF", "TURN OFF": "TURN ON",
}


# Movement inverses, kept SEPARATE from the verb table above because they are
# whole commands rather than verbs taking an object. Without these,
# `undoes_recent_progress` could not tell that EAST reverses WEST or DOWN
# reverses UP — so nothing demoted a proposal that walked straight back where
# it came from. In frontier3-20260824 the agent reached Behind House, the room
# with the window into the house, and oscillated off it; pf-20260824 did the
# same vertically, GO UP -> GO UP -> GO DOWN, while the ship's clock ran.
_DIRECTION_INVERSES = {
    "NORTH": "SOUTH", "SOUTH": "NORTH",
    "EAST": "WEST", "WEST": "EAST",
    "NORTHEAST": "SOUTHWEST", "SOUTHWEST": "NORTHEAST",
    "NORTHWEST": "SOUTHEAST", "SOUTHEAST": "NORTHWEST",
    "UP": "DOWN", "DOWN": "UP",
}


def inverse_of(command: Optional[str]) -> str:
    """The command that would undo `command`, or "" if it has no inverse."""
    normalized = normalize_command(command)
    if not normalized:
        return ""

    # Whole-command match only, so "NORTH" inverts but "PUSH NORTH WALL" does
    # not become "PUSH SOUTH WALL". normalize_command has already collapsed
    # "GO WEST" and "W" to "WEST", so every phrasing lands here.
    if normalized in _DIRECTION_INVERSES:
        return _DIRECTION_INVERSES[normalized]

    tokens = normalized.split()
    for verb_len in (2, 1):
        # `>=` not `>`: a verb-only command like "OPEN" has an inverse too, and
        # requiring an object meant a bare direction could never resolve one
        # even once directions were listed.
        if len(tokens) >= verb_len:
            verb = " ".join(tokens[:verb_len])
            if verb in _INVERSES:
                return " ".join([_INVERSES[verb]] + tokens[verb_len:])
    return ""


def normalize_command(command: Optional[str]) -> str:
    """Canonical form for comparing two commands.

    Case and spacing, plus ONE semantic rule: a command that is purely a
    movement collapses to its canonical direction, so "GO WEST", "WEST" and
    "W" are one command rather than three. They were three, and the cost was
    real — repetition suppression, the `succeeded` map and undo detection all
    key off this string, so an agent could alternate phrasings forever without
    any of them noticing. Observed in frontier3-20260824: the agent reached
    Behind House, the room containing the window into the house, and bounced
    GO WEST -> EAST -> GO WEST off it with ZERO suppressions in 16 turns.

    Everything else stays literal: "OPEN DOOR" and "OPEN THE DOOR" remain
    distinct, because a false suppression silently removes a real option and
    nothing in the game text ever corrects it. `extract_direction` is safe to
    lean on here precisely because it is the STRICT parser — it returns None
    for "TAKE LAMP", "OPEN WINDOW" and "PUSH NORTH WALL" — as opposed to
    `find_mentioned_directions`, which scans prose permissively. Keep the two
    apart; merging them is a standing temptation and would be a bug.
    """
    text = " ".join((command or "").strip().upper().split())
    if not text:
        return ""
    if is_probable_movement_command(text):
        direction = extract_direction(text)
        if direction:
            return direction
    return text


@dataclass
class TurnContext:
    """Deterministic snapshot of the world at the start of a turn."""

    location: str
    game_text: str
    score: int
    moves: int

    inventory: List[str] = field(default_factory=list)
    recent_turns: str = ""
    full_summary: str = ""
    long_summary: str = ""
    exits: List[Tuple[str, str]] = field(default_factory=list)
    strategic_analysis: str = ""

    # target location (casefolded) -> next step, "NO PATH", or "ALREADY THERE"
    directions: Dict[str, str] = field(default_factory=dict)

    # Directions the GAME says exist here, decoded from its exits array (#30).
    # Not a perfect oracle — North of House reports an exit that is refused —
    # so it is used to RANK candidates, never to restrict them.
    game_exits: List[str] = field(default_factory=list)

    # object -> commands the GAME says it will accept for it, straight from
    # the backend (#30). Authoritative, so the InteractionAgent no longer has
    # to guess what is interactable by pattern-matching English (#16).
    available_actions: Dict[str, List[str]] = field(default_factory=dict)

    # Turns since the score last moved, and the frontier of places seen but
    # not entered. Both feed the arbiter, which had no notion of the objective
    # (Milestone 5b): it ranked proposals with no idea whether any of them
    # advanced the game.
    turns_since_score_change: int = 0

    # The GAME's own clock, straight from the backend (`time`). Parsed on the
    # response model since #30 and read by nothing until now — which on a timed
    # objective means the agents could not see the deadline they were being
    # judged against. Planetfall's ship explodes; Zork reports 0, so this
    # renders only where it means something.
    game_time: Optional[int] = None
    frontier: List[str] = field(default_factory=list)

    # Commands that SUCCEEDED in this room (changed something). Used to spot a
    # proposal that would undo one of them.
    succeeded: Dict[str, str] = field(default_factory=dict)

    # Commands already tried IN THIS ROOM that changed nothing, mapped to the
    # response they produced (GitHub issue #18). Zork is deterministic: the
    # same command, in the same room, with nothing changed since, produces the
    # same result. Re-proposing it is guaranteed waste.
    unproductive: Dict[str, str] = field(default_factory=dict)

    @property
    def inventory_summary(self) -> str:
        """Inventory rendered for a prompt. Never blank — an empty string in a
        prompt reads as a missing value rather than as 'carrying nothing'."""
        return ", ".join(self.inventory) if self.inventory else "empty"

    @property
    def exits_summary(self) -> str:
        if not self.exits:
            return "No known exits from here yet."
        return ", ".join(f"{direction} -> {dest}" for direction, dest in self.exits)

    def direction_to(self, target: Optional[str]) -> str:
        """Next step toward `target`, precomputed for every spawned issue."""
        if not is_known_location(target):
            return "NOT AVAILABLE"
        return self.directions.get(target.strip().casefold(), "NO PATH")

    @property
    def available_actions_summary(self) -> str:
        """The game's own list of valid commands here, rendered for a prompt.

        Everything listed is guaranteed to parse — the backend supplied it —
        so an agent choosing from this list cannot invent an object that is
        not present, which is the entire failure mode of #16.
        """
        if not self.available_actions:
            return "The game did not report any interactable objects here."
        lines = []
        for obj, commands in self.available_actions.items():
            if self.unproductive:
                commands = [c for c in commands
                            if normalize_command(c) not in self.unproductive]
            if commands:
                lines.append(f"  {obj}: {', '.join(commands)}")
        return "\n".join(lines) or "Everything here has already been tried."

    @property
    def score_trajectory(self) -> str:
        """How the score is moving, phrased for a prompt.

        A bare number tells the arbiter nothing; "unchanged for 22 turns" tells
        it the current line of play is not working.
        """
        if self.turns_since_score_change <= 0:
            return "the score moved this turn — that line of play is working"
        if self.turns_since_score_change < 5:
            return f"unchanged for {self.turns_since_score_change} turns"
        return (f"UNCHANGED FOR {self.turns_since_score_change} TURNS — "
                f"the current approach is not scoring, prefer a change of direction")

    @property
    def clock_summary(self) -> str:
        """The game clock, rendered only when the backend actually keeps one.

        Zork reports 0 for every turn, so rendering it there would add a line
        of noise to every prompt in the game that has no deadline. Planetfall
        advances it per move and blows the ship up.
        """
        if not self.game_time:
            return ""
        return (f"GAME CLOCK: {self.game_time} — the game's own clock. It "
                f"advances with every move, including wasted ones.")

    @property
    def frontier_summary(self) -> str:
        """Leads named but not followed up, rendered for a prompt."""
        if not self.frontier:
            return "No unfollowed leads — nothing named nearby is unvisited."
        return "\n".join(f"  - {room}" for room in self.frontier)

    def is_backend_confirmed(self, command: Optional[str]) -> bool:
        """True when the GAME itself listed this command for an object here.

        Mirrors the ExplorerAgent's `+3 for a game-confirmed exit`: the
        backend's accepted-command list (#30/#16) is the strongest evidence
        available that a proposal will do something, because the server
        supplied it. Matching is verb + object rather than string equality —
        the agent says "OPEN escape pod bulkhead" where the backend lists
        "open bulkhead" under the key "escape pod bulkhead", and both name the
        same act.
        """
        proposed = normalize_command(command)
        if not proposed or not self.available_actions:
            return False
        verb = proposed.split()[0]
        for obj, commands in self.available_actions.items():
            obj_norm = normalize_command(obj)
            mentions_object = obj_norm and (
                obj_norm in proposed
                or any(w in proposed.split() for w in obj_norm.split()))
            if not mentions_object:
                continue
            if any(normalize_command(c).split()[:1] == [verb] for c in commands):
                return True
        return False

    def undoes_recent_progress(self, command: Optional[str]) -> str:
        """The command this proposal would undo, or "" if none.

        The agent opened the grating, then proposed closing it — because the
        game's accepted-command list mentions "close grating". Reading a
        grammar as a recommendation is a failure mode worth blocking in code,
        not only in the prompt.
        """
        proposed = normalize_command(command)
        if not proposed:
            return ""
        for done in self.succeeded:
            if inverse_of(done) == proposed:
                return done
        return ""

    def is_route_step(self, command: Optional[str]) -> bool:
        """True when this command is the next step toward a tracked issue.

        The undo rule cannot tell aimless backtracking from a goal-directed
        return: both reverse the previous move. In pf4-20260824 the escape pod
        was at Deck Nine and the agent was one room above it, so walking back
        down — the correct move — looked identical to the oscillation the rule
        exists to stop.

        `directions` holds the precomputed next step toward each tracked
        issue's location, so a proposal matching one of them is going somewhere
        on purpose.
        """
        proposed = normalize_command(command)
        if not proposed or not self.directions:
            return False
        return any(normalize_command(step) == proposed
                   for step in self.directions.values()
                   if step and step not in ("NO PATH", "NOT AVAILABLE"))

    def is_unproductive(self, command: Optional[str]) -> bool:
        """True when this exact command already did nothing in this room."""
        return normalize_command(command) in self.unproductive

    @property
    def unproductive_summary(self) -> str:
        """Rendered for a prompt. Shows the response too, so the model can see
        *why* the command is pointless rather than just being forbidden — the
        #21 lesson: a rule that only says "don't" invites the model to invent
        an alternative."""
        if not self.unproductive:
            return "None yet."
        return "\n".join(
            f"  - {command} -> \"{response.strip()[:90]}\""
            for command, response in self.unproductive.items()
        )

    def research_context_for(self, target_location: Optional[str] = None) -> str:
        """The text block that replaces an agent's research phase.

        Same information the tool calls returned, assembled in code — and
        complete by construction, where the LLM route could return any subset.
        """
        blocks = [
            f"CURRENT LOCATION: {self.location}",
            f"INVENTORY: {self.inventory_summary}",
            f"KNOWN EXITS: {self.exits_summary}",
            f"ALREADY TRIED HERE, NO EFFECT (do not repeat):\n{self.unproductive_summary}",
            f"THE GAME ACCEPTS THESE COMMANDS HERE:\n{self.available_actions_summary}",
        ]
        if self.clock_summary:
            blocks.insert(1, self.clock_summary)
        if target_location:
            blocks.append(f"DIRECTION TO '{target_location}': {self.direction_to(target_location)}")
        if self.strategic_analysis:
            blocks.append(f"STRATEGIC ANALYSIS:\n{self.strategic_analysis}")
        if self.full_summary:
            blocks.append(f"RECENT SUMMARY:\n{self.full_summary}")
        if self.long_summary:
            blocks.append(f"STORY SO FAR:\n{self.long_summary}")
        if self.recent_turns:
            blocks.append(f"RECENT TURNS:\n{self.recent_turns}")
        return "\n\n".join(blocks)


def build_turn_context(
    *,
    game_response,
    history_toolkit,
    mapper_toolkit,
    inventory_toolkit,
    issue_locations: Optional[List[str]] = None,
) -> TurnContext:
    """Assemble the turn's context from local state. Never raises.

    Each read is guarded independently: a failure in one source degrades that
    one field rather than costing the turn its whole context (#1).
    """
    import logging
    logger = logging.getLogger(__name__)

    def safe(label, fn, default):
        """Guard each source independently — including the attribute lookup,
        which is why every caller passes a lambda rather than a bound method:
        a toolkit missing `.state` must degrade one field, not the turn."""
        try:
            return fn()
        except Exception as e:
            logger.warning(f"[TurnContext] {label} unavailable: {e}")
            return default

    location = game_response.LocationName or ""
    context = TurnContext(
        location=location or "Unknown",
        game_text=game_response.Response or "",
        score=game_response.Score,
        moves=game_response.Moves,
        game_time=getattr(game_response, "Time", None),
    )

    # Decode the backend's exits array into direction names (#30).
    _EXIT_CODES = {0: "NORTH", 1: "SOUTH", 2: "EAST", 3: "WEST",
                   10: "UP", 11: "DOWN"}
    api_exits = getattr(game_response, "Exits", None)
    if api_exits:
        context.game_exits = [
            _EXIT_CODES[int(code)] for code in api_exits
            if isinstance(code, int) and int(code) in _EXIT_CODES
        ]

    # The game reports exactly which commands it will accept for each object
    # in this room (#30). Nothing to infer.
    api_actions = getattr(game_response, "ActionsAvailableFromLocation", None)
    if isinstance(api_actions, dict):
        context.available_actions = {
            str(obj): [str(c) for c in (cmds or [])]
            for obj, cmds in api_actions.items()
        }

    # The backend reports inventory itself (#30); fall back to our tracking.
    api_inventory = getattr(game_response, "Inventory", None)
    if api_inventory is not None:
        context.inventory = list(api_inventory)
    else:
        context.inventory = safe("inventory", lambda: inventory_toolkit.state.get_items(), [])

    context.full_summary = safe("recent summary", lambda: history_toolkit.state.get_full_summary(), "")
    context.long_summary = safe("long summary", lambda: history_toolkit.state.get_long_running_summary(), "")

    def _recent():
        turns = history_toolkit.state.get_recent_turns(RECENT_TURNS_FOR_AGENTS)
        return "\n".join(
            f"Turn {t.turn_number}: {t.player_command} -> {t.game_response}"
            for t in turns
        )
    context.recent_turns = safe("recent turns", _recent, "")

    # Which commands have already been shown to do nothing HERE (#18).
    #
    # A turn counts as unproductive when it neither scored nor moved us. The
    # room must match the room we are in now: "OPEN DOOR" failing in the
    # Kitchen says nothing about the Cellar. Scoped to the recent window so a
    # command becomes retryable again once it ages out — the world does change
    # (a door gets unlocked, a lamp gets lit), and permanent suppression would
    # repeat #11's mistake of letting an inference become unfalsifiable.
    def _unproductive():
        turns = history_toolkit.state.get_recent_turns(RECENT_TURNS_FOR_AGENTS)
        seen = {}
        previous = None
        for turn in turns:
            # The first turn in the window has nothing to compare against, so
            # we cannot tell whether it moved or scored. Skip it: the command
            # that WALKED us into this room would otherwise be recorded as
            # having done nothing here. Bias is always toward not suppressing
            # — a false suppression silently removes a legitimate action and
            # nothing in the game text would ever correct it.
            if previous is None:
                previous = turn
                continue
            moved = turn.location != previous.location
            scored = turn.score > previous.score
            here = turn.location and location and \
                turn.location.strip().casefold() == location.strip().casefold()
            if here and not moved and not scored:
                seen[normalize_command(turn.player_command)] = turn.game_response
            previous = turn
        seen.pop("", None)
        return seen

    context.unproductive = safe("unproductive commands", _unproductive, {})

    def _succeeded():
        """Commands that visibly changed something in this room."""
        turns = history_toolkit.state.get_recent_turns(RECENT_TURNS_FOR_AGENTS)
        done = {}
        for turn in turns:
            here = turn.location and location and \
                turn.location.strip().casefold() == location.strip().casefold()
            command = normalize_command(turn.player_command)
            if here and command and command not in context.unproductive:
                done[command] = turn.game_response
        return done

    context.succeeded = safe("successful commands", _succeeded, {})

    # How long since the score moved (Milestone 5b). The arbiter needs to know
    # the current approach is not working, not just what the score is.
    def _stale():
        turns = history_toolkit.state.get_recent_turns(50)
        if not turns:
            return 0
        best = max(t.score for t in turns)
        stale = 0
        for turn in reversed(turns):
            if turn.score >= best and turn.score > 0:
                break
            stale += 1
        return stale
    context.turns_since_score_change = safe("score trajectory", _stale, 0)

    # Leads the agent has been told about but not followed up (Milestone 5b).
    # TWO sources, because the map alone measured EMPTY on a live run: on a
    # linear path A->B->C every room reached was also left, so "reached but
    # not departed from" is the empty set exactly when a redirect is needed.
    #
    # The second source is the room text, and it is the one that matters. "A
    # white house" is the whole early game and is not a map node — it is a
    # noun in a description that nothing else converts into a destination.
    def _frontier():
        transitions = mapper_toolkit.state.get_all_transitions()
        reached = {t.to_location for t in transitions if t.to_location != "BLOCKED"}
        departed = {t.from_location for t in transitions}
        here = (location or "").strip().casefold()
        unexplored = sorted(r for r in reached - departed
                            if r and r.strip().casefold() != here)

        visited = {t.from_location for t in transitions} | reached
        if location:
            visited.add(location)
        # RAW room prose, not the formatted transcript. `recent_turns` renders
        # as "Turn 1: look -> West Of House\nYou are standing ... white house",
        # and with no sentence break between the header and the description the
        # location label lands in the same sentence as the landmark — firing
        # the same-sentence retirement rule and silently killing the lead. Seen
        # live: turn 4 reported "no unfollowed leads" while the house sat in
        # the very text being scanned.
        #
        # The current response alone is not enough either: after "TAKE leaflet"
        # it is just "Taken." and names nothing, which is exactly when the
        # standing lead matters most.
        def _prose():
            turns = history_toolkit.state.get_recent_turns(RECENT_TURNS_FOR_AGENTS)
            return [t.game_response for t in turns if t.game_response]
        recent_prose = safe("recent prose", _prose, [])
        named = unvisited_landmarks(
            "\n".join(filter(None, [context.game_text, *recent_prose])), visited)

        return [f"{room} (on the map, never explored from)" for room in unexplored] + \
               [f"{phrase} (named nearby, never entered)" for phrase in named]
    context.frontier = safe("frontier", _frontier, [])

    if is_known_location(location):
        context.exits = safe("exits", lambda: mapper_toolkit.state.get_exits_from(location), [])

    context.strategic_analysis = safe(
        "strategic analysis",
        lambda: __import__("tools.analysis", fromlist=["get_strategic_analysis"]).get_strategic_analysis.invoke({}),
        "",
    )

    # Precompute routing for every issue an agent will advocate for, so no
    # agent has to ask an LLM for permission to run a BFS.
    if issue_locations and is_known_location(location):
        pathfinder = safe("pathfinder", lambda: mapper_toolkit.state.pathfinder, None)
        if pathfinder is not None:
            for target in issue_locations:
                if not is_known_location(target):
                    continue
                key = target.strip().casefold()
                if key in context.directions:
                    continue
                context.directions[key] = safe(
                    f"direction to {target}",
                    lambda t=target: pathfinder.get_next_step(location, t) or "NO PATH",
                    "NO PATH",
                )

    return context

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

from tools.mapping.locations import is_known_location

# How many recent turns to put in front of an agent. Bounded deliberately:
# per-call latency scales with prompt size, and the 2026-08-22 checkpoint
# showed history-shaped prompt content is what drives turn-time growth.
RECENT_TURNS_FOR_AGENTS = 10


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

    def research_context_for(self, target_location: Optional[str] = None) -> str:
        """The text block that replaces an agent's research phase.

        Same information the tool calls returned, assembled in code — and
        complete by construction, where the LLM route could return any subset.
        """
        blocks = [
            f"CURRENT LOCATION: {self.location}",
            f"INVENTORY: {self.inventory_summary}",
            f"KNOWN EXITS: {self.exits_summary}",
        ]
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
    )

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

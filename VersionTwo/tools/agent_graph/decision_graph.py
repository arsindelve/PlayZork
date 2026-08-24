"""
Simple LangGraph for managing the decision-making flow.

Flow: BuildContext → (SpawnAgents → Decide | CloseIssues | Observe) → Persist → END

This introduces graph-based control flow while keeping the existing
research and decision logic intact.
"""
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from zork.zork_api_response import ZorkApiResponse
from adventurer.adventurer_response import AdventurerResponse
from tools.history import HistoryToolkit
from tools.memory import MemoryToolkit
from tools.mapping import MapperToolkit
from tools.mapping.directions import (
    CANONICAL_DIRECTIONS,
    find_mentioned_directions,
    normalize_direction,
)
from tools.memory.closure_guard import closure_is_contradicted
from tools.memory.issue_target import resolve_issue_target
from tools.mapping.locations import UNKNOWN_LOCATION, is_known_location
from langchain_core.runnables import Runnable
from .issue_agent import IssueAgent
from .explorer_agent import ExplorerAgent
from .loop_detection_agent import LoopDetectionAgent
from .interaction_agent import InteractionAgent
from .issue_closed_agent import IssueClosedAgent
from .issue_closed_response import IssueClosedResponse
from .observer_agent import ObserverAgent
from .observer_response import ObserverResponse
from .tool_execution import invoke_tool_safely
from .turn_context import build_turn_context, normalize_command


class DecisionState(TypedDict):
    """State passed through the decision graph"""

    # Input
    game_response: ZorkApiResponse
    player_command: str
    # In graph state rather than a mutable side-channel dict (#26): the graph
    # should carry the turn's data, not a reference smuggled past it.
    turn_number: int

    # Spawn phase output
    issue_agents: List[IssueAgent]
    explorer_agent: Optional[ExplorerAgent]  # Single agent, can be None
    loop_detection_agent: Optional[LoopDetectionAgent]  # Single agent, always spawned
    interaction_agent: Optional[InteractionAgent]  # Single agent, always spawned

    # Deterministic turn context + the memory snapshot every branch shares,
    # both produced by build_context (#25, #23)
    turn_context: object
    memories: List

    # Research phase output (legacy; kept so reports and any external reader
    # that expects the key keep working)
    research_context: str
    research_tool_calls: List[dict]  # Tool calls made by research agent

    # Decision phase output
    decision: Optional[AdventurerResponse]
    decision_prompt: str  # Formatted prompt for reporting
    decision_tool_calls: List[dict]  # Tool calls made by decision agent

    # Issue closing phase output
    issue_closed_response: Optional[IssueClosedResponse]
    # Closures decided by close_issues but APPLIED by persist, so a turn
    # cancelled by the turn budget can't half-apply memory state (see #3).
    pending_closures: List[dict]

    # Observation phase output
    observer_response: Optional[ObserverResponse]

    # Persistence tracking
    memory_persisted: bool


def _neutralize_failed_agent(agent, error: BaseException) -> None:
    """Reset an agent that raised during research so it cannot advocate.

    Each agent type has its own "I have no proposal" representation, and both
    the proposal formatter and the report writer key off it:
      * InteractionAgent / LoopDetectionAgent use confidence == 0 (their
        confidence is typed int and compared with `> 0`, so it must never
        become None).
      * IssueAgent / ExplorerAgent use proposed_action None + confidence None.
    The failure reason is left on the agent so it surfaces in the HTML report
    instead of vanishing.
    """
    if isinstance(agent, (InteractionAgent, LoopDetectionAgent)):
        agent.proposed_action = "nothing"
        agent.confidence = 0
        if isinstance(agent, InteractionAgent):
            agent.detected_objects = []
            agent.inventory_items = []
        else:
            agent.loop_detected = False
    else:
        agent.proposed_action = None
        agent.confidence = None
    agent.reason = f"Agent failed during research: {error}"


def create_build_context_node(
    memory_toolkit: MemoryToolkit,
    mapper_toolkit: MapperToolkit,
    inventory_toolkit,
    history_toolkit: HistoryToolkit,
):
    """Create the node that assembles the turn's deterministic facts.

    Hoisted out of spawn_agents so the branches that run beside it —
    close_issues and observe — can share the same snapshot (#23/#26). It is
    pure code: local SQLite reads and the turn response, milliseconds total.
    """
    def build_context_node(state: DecisionState) -> dict:
        import logging
        logger = logging.getLogger(__name__)

        # Top tracked issues, ranked by lazily-decayed importance. Read once
        # here rather than in spawn, so every branch sees one consistent
        # snapshot of memory for this turn.
        memories = memory_toolkit.state.get_top_memories(
            limit=5,
            current_turn=state.get("turn_number"),
        )
        logger.info(f"Retrieved {len(memories)} memories from database")

        # Sort by location name for cleaner console display
        memories_sorted = sorted(memories, key=lambda m: m.location if m.location else "")

        context = build_turn_context(
            game_response=state["game_response"],
            history_toolkit=history_toolkit,
            mapper_toolkit=mapper_toolkit,
            inventory_toolkit=inventory_toolkit,
            issue_locations=[m.location for m in memories_sorted if m.location],
        )

        # Precompute routes to where each issue actually POINTS, which is not
        # always where it was noticed: "Ensign Blather at Reactor Lobby —
        # return to Deck Nine" was routed to Reactor Lobby, the room the agent
        # was already standing in. Done as a second pass because it needs
        # `known_locations`, which the first pass produces.
        targets = [
            resolve_issue_target(m.content, m.location, context.known_locations)
            for m in memories_sorted
        ]
        extra = [t for t in targets if t and t.strip().casefold() not in context.directions]
        if extra:
            context = build_turn_context(
                game_response=state["game_response"],
                history_toolkit=history_toolkit,
                mapper_toolkit=mapper_toolkit,
                inventory_toolkit=inventory_toolkit,
                issue_locations=[m.location for m in memories_sorted if m.location] + extra,
            )
        return {"turn_context": context, "memories": memories_sorted}

    return build_context_node


def _blocked_signature(memory, context) -> tuple:
    """What an IssueAgent's "I cannot act" verdict actually depends on.

    Observed live, every blocking reason was of the form "the grating is
    locked and I have no key" or "there is no known path to the Clearing from
    here" — they turn on INVENTORY and LOCATION, not on the room description.
    While both are unchanged the answer cannot change either, and re-asking
    costs ~3000 tokens (the most expensive call in the system) to hear the
    same thing again. 55% of IssueAgent calls on the measured run returned
    "nothing".

    A change to either input invalidates the verdict and the agent runs again.
    """
    return (
        memory.id,
        (context.location or "").strip().casefold(),
        tuple(sorted(item.strip().casefold() for item in context.inventory)),
    )


def create_spawn_agents_node(
    memory_toolkit: MemoryToolkit,
    mapper_toolkit: MapperToolkit,
    inventory_toolkit,
    decision_llm,
    history_toolkit: HistoryToolkit,
):
    """
    Create the spawn agents node that creates IssueAgents and ExplorerAgent.

    Args:
        memory_toolkit: MemoryToolkit for accessing stored strategic issues
        mapper_toolkit: MapperToolkit for accessing map state
        inventory_toolkit: InventoryToolkit for accessing inventory
        decision_llm: LLM for generating proposals
        history_toolkit: HistoryToolkit for accessing tools

    Returns:
        Node function for the graph
    """
    # issue signature -> why it could not be acted on. Lives as long as the
    # graph, so a verdict persists across turns.
    blocked_issues: dict = {}

    async def spawn_agents_node(state: DecisionState) -> dict:
        """
        Spawn phase: Create one IssueAgent for each tracked strategic issue.
        Each agent performs its own research and generates a proposal IN PARALLEL.
        """
        import asyncio
        import logging
        logger = logging.getLogger(__name__)

        logger.info("\n" + "=" * 80)
        logger.info("SPAWN AGENTS - Creating specialized agents for this turn")
        logger.info("=" * 80)

        memories_sorted = state["memories"]

        context = state["turn_context"]

        # One IssueAgent per issue, but skip those already known to be
        # unactionable under this location + inventory. A skipped agent keeps
        # its recorded reason, so the HTML report still explains its silence.
        issue_agents = []
        skipped = 0
        for mem in memories_sorted:
            agent = IssueAgent(memory=mem)
            cached = blocked_issues.get(_blocked_signature(mem, context))
            if cached is not None:
                agent.proposed_action = None
                agent.confidence = None
                agent.reason = cached
                skipped += 1
                logger.info(f"SKIPPED IssueAgent ID:{mem.id} — still blocked: {cached[:70]}")
            issue_agents.append(agent)

        logger.info(f"SPAWNED {len(issue_agents) - skipped} IssueAgents "
                    f"({skipped} skipped as already blocked)")

        # Extract current game state
        game_response = state["game_response"]
        # "Unknown" is prose for the prompts only. Anything that indexes the
        # map, routes, or gets stored must go through is_known_location (#7).
        current_location = game_response.LocationName or UNKNOWN_LOCATION
        location_is_known = is_known_location(game_response.LocationName)
        current_game_text = game_response.Response
        current_score = game_response.Score
        current_moves = game_response.Moves

        # ========== NEW: Spawn ONE ExplorerAgent (if unexplored directions exist) ==========
        # Get known exits from current location. With no room name there is no
        # map node to explore *from*: querying exits for the fake room
        # "Unknown" returned nothing, so the explorer confidently reported all
        # ten directions unexplored and proposed a move it could not map (#7).
        known_exits = (
            mapper_toolkit.state.get_exits_from(current_location)
            if location_is_known
            else []
        )
        # Canonicalize so a passage recorded as "N" counts as NORTH explored
        # (#9). Without this the explorer re-proposed the same direction every
        # turn, forever.
        known_directions = {normalize_direction(direction) for direction, _ in known_exits}

        unexplored_directions = [
            d for d in CANONICAL_DIRECTIONS
            if d not in known_directions
        ]

        # Which unexplored directions does the room prose actually name?
        # Whole-word matching only: substring containment scored "NE" inside
        # CORNER and "SE" inside HOUSE, and a fabricated mention both outranks
        # every real exit and adds +20 confidence (#8). It also matched
        # "NORTH" inside "NORTHEAST", sending the agent north when the room
        # said northeast.
        mentioned_directions = find_mentioned_directions(
            current_game_text,
            unexplored_directions,
        )

        # Create ONE ExplorerAgent if there are unexplored directions
        explorer_agent = None
        if unexplored_directions and not location_is_known:
            logger.info(
                "NO ExplorerAgent spawned - current location is unknown, so there "
                "is no map node to explore from"
            )
        elif unexplored_directions:
            explorer_agent = ExplorerAgent(
                current_location=current_location,
                unexplored_directions=unexplored_directions,
                mentioned_directions=mentioned_directions,
                game_exits=context.game_exits,
                turn_number=0  # Will be set properly when turn_number added to state
            )
            logger.info(f"SPAWNED 1 ExplorerAgent - {len(unexplored_directions)} unexplored directions: {unexplored_directions}")
            logger.info(f"  Mentioned in description: {mentioned_directions if mentioned_directions else 'None'}")
            logger.info(f"  Best direction chosen: {explorer_agent.best_direction}")
        else:
            logger.info("NO ExplorerAgent spawned - all directions explored from this location")

        # ========== DISABLED: LoopDetectionAgent ==========
        # loop_detection_agent = LoopDetectionAgent()
        # logger.info("SPAWNED 1 LoopDetectionAgent - monitors for stuck/oscillating patterns")
        loop_detection_agent = None  # DISABLED - not useful in practice
        logger.info("LoopDetectionAgent DISABLED")

        # ========== NEW: Spawn ONE InteractionAgent (ALWAYS) ==========
        interaction_agent = InteractionAgent()
        logger.info("SPAWNED 1 InteractionAgent - identifies local object interactions")

        # ========== PARALLEL RESEARCH: IssueAgents + ExplorerAgent + InteractionAgent ==========
        num_special_agents = (1 if explorer_agent else 0) + 1  # +1 for Interaction (Loop disabled)
        logger.info(f"Starting PARALLEL research for {len(issue_agents)} IssueAgents + {num_special_agents} special agents...")

        # Build a coroutine for each agent's research+propose pass. Agents are
        # async-native (chain.ainvoke), so no thread offload is needed.
        def agent_coroutine(agent):
            # One LLM call per agent now, not two.
            return agent.propose(decision_llm=decision_llm, context=context)

        # Filter out None agents (e.g., loop_detection_agent is disabled)
        runnable_issues = [a for a in issue_agents if a.reason is None]
        all_agents = [a for a in runnable_issues
                      + [explorer_agent, loop_detection_agent, interaction_agent]
                      if a is not None]

        # Run all agents in parallel — pure async, no threads.
        # return_exceptions=True isolates failures: one agent blowing up must
        # not cancel its siblings or end the turn (see #1). A failed agent is
        # neutralized so it cannot contribute a proposal, but is kept in state
        # so the HTML report still shows what it attempted and why it failed.
        failed_agents = 0
        if all_agents:
            results = await asyncio.gather(
                *(agent_coroutine(a) for a in all_agents),
                return_exceptions=True,
            )
            for agent, result in zip(all_agents, results):
                if not isinstance(result, BaseException):
                    continue
                if isinstance(result, asyncio.CancelledError):
                    # Turn budget expired / task cancelled — must propagate.
                    raise result
                failed_agents += 1
                agent_label = type(agent).__name__
                logger.error(
                    f"{agent_label} failed during research/proposal: {result}",
                    exc_info=result,
                )
                _neutralize_failed_agent(agent, result)

        # Remember a fresh "cannot act" verdict so the next turn does not pay
        # ~3000 tokens to hear it again.
        for agent, mem in zip(issue_agents, memories_sorted):
            action = (agent.proposed_action or "").strip().lower()
            if agent in all_agents and action in ("nothing", "none"):
                blocked_issues[_blocked_signature(mem, context)] = agent.reason or "no action available"

        logger.info(
            f"All {len(all_agents)} agents completed research in PARALLEL "
            f"({failed_agents} failed and were excluded from proposals)"
        )
        logger.info("=" * 80)
        logger.info("SPAWN AGENTS COMPLETE")
        logger.info("=" * 80)

        return {
            "issue_agents": issue_agents,
            "explorer_agent": explorer_agent,          # single agent, can be None
            "loop_detection_agent": loop_detection_agent,
            "interaction_agent": interaction_agent,
        }


    return spawn_agents_node


def create_decision_node(decision_chain: Runnable):
    """
    Create the decision node that generates structured output from agent
    proposals and previously gathered research context.

    Args:
        decision_chain: The LangChain decision chain with structured output

    Returns:
        Node function for the graph
    """
    async def decision_node(state: DecisionState) -> DecisionState:
        """
        Decision phase: Generate AdventurerResponse from agent proposals plus
        research_context already gathered by research_node and per-agent research.
        """
        import logging
        logger = logging.getLogger(__name__)

        zork_response = state["game_response"]
        # Assembled in code by the spawn node (#25); the research node that
        # used to produce this via an LLM round-trip is gone.
        turn_context = state.get("turn_context")
        research_context = (
            turn_context.research_context_for() if turn_context is not None
            else state.get("research_context", "")
        )
        issue_agents = state["issue_agents"]
        explorer_agent = state["explorer_agent"]
        loop_detection_agent = state["loop_detection_agent"]
        interaction_agent = state["interaction_agent"]

        logger.info("\n" + "=" * 80)
        logger.info("[DecisionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("[DecisionAgent] AGENT: DecisionAgent")
        logger.info("[DecisionAgent] PURPOSE: Choose best action from all agent proposals")
        logger.info(f"[DecisionAgent] LOCATION: {zork_response.LocationName}")
        logger.info(f"[DecisionAgent] SCORE: {zork_response.Score}, MOVES: {zork_response.Moves}")
        logger.info("[DecisionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        logger.info("=" * 80)
        logger.info("DECISION - Choosing best action from agent proposals")
        logger.info("=" * 80)
        logger.info(f"Location: {zork_response.LocationName}")
        logger.info(f"Score: {zork_response.Score}, Moves: {zork_response.Moves}")
        logger.info(f"Game Response (first 100): {zork_response.Response[:100]}...")

        # Format agent proposals for Decision Agent
        agent_proposals_text = _format_agent_proposals(
            issue_agents, explorer_agent, loop_detection_agent, interaction_agent,
            context=turn_context,
        )
        logger.info(f"Agent Proposals:\n{agent_proposals_text}")
        logger.info("=" * 80)

        # No additional tool-calling pass here: research_node + per-agent research
        # already gathered sufficient context. Keep tool_calls_history empty for the
        # report writer's compatibility.
        tool_calls_history: list = []
        full_research_context = research_context

        decision_input = {
            "score_trajectory": (turn_context.score_trajectory if turn_context
                                 else "unknown"),
            "frontier": (turn_context.frontier_summary if turn_context
                         else "unknown"),
            "score": zork_response.Score,
            "locationName": zork_response.LocationName,
            "moves": zork_response.Moves,
            "game_response": zork_response.Response,
            "research_context": full_research_context,
            "agent_proposals": agent_proposals_text
        }

        # Format the full prompt for reporting (from prompt_library.py)
        from adventurer.prompt_library import PromptLibrary
        system_prompt = PromptLibrary.get_decision_agent_evaluation_prompt()
        human_prompt = PromptLibrary.get_decision_agent_human_prompt()

        # Format human prompt with actual values
        formatted_human = human_prompt.format(
            score_trajectory=(turn_context.score_trajectory if turn_context else "unknown"),
            frontier=(turn_context.frontier_summary if turn_context else "unknown"),
            locationName=zork_response.LocationName,
            score=zork_response.Score,
            moves=zork_response.Moves,
            game_response=zork_response.Response,
            research_context=full_research_context,
            agent_proposals=agent_proposals_text
        )

        # Combine system + human for full prompt
        full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[HUMAN]\n{formatted_human}"

        from llm_utils import ainvoke_with_retry
        decision = await ainvoke_with_retry(
            decision_chain.with_config(run_name="Decision Agent"),
            decision_input,
            operation_name="Decision Agent"
        )

        logger.info(f"DECISION MADE: {decision.command}")
        logger.info(f"REASON: {decision.reason}")

        return {
            "decision": decision,
            "decision_prompt": full_prompt,
            "decision_tool_calls": tool_calls_history,
        }

    return decision_node


def _format_agent_proposals(issue_agents, explorer_agent, loop_detection_agent,
                            interaction_agent, context=None):
    """Format agent proposals for Decision Agent evaluation.

    A proposal that repeats a command already shown to do nothing in this room
    is marked and its expected value zeroed (#18). This is done in CODE rather
    than by asking the model nicely: the agent deadlocked for five turns
    alternating two commands whose failures were both sitting in its own
    recent history, and the #21 inventory bug showed that a 14B model given a
    prohibition will happily invent its own way around it.

    The proposal is annotated rather than removed, so the arbiter can still
    pick it if literally everything else is exhausted — and can see why it was
    demoted.
    """
    # Proposals are collected as BLOCKS carrying their expected value, so a
    # demoted one can be WITHHELD rather than merely annotated. See the filter
    # at the end of this function.
    blocks = []

    def block(ev=None, withholdable=False):
        """Start a new proposal block and return its line list.

        `withholdable` marks a demotion safe to take off the ballot entirely.
        Only WOULD-UNDO qualifies; see repeat_note.
        """
        blocks.append([ev, [], withholdable])
        return blocks[-1][1]

    lines = block()

    def repeat_note(action):
        """Marker, EV multiplier, and whether the demotion may be WITHHELD.

        The two demotion reasons have different epistemics, and conflating them
        nearly lost the game. See the filter at the end of this function.

        - ALREADY TRIED: true only while the world is unchanged. An explosion
          opened the escape pod bulkhead in pf4-20260824, and `WEST` — the move
          that escaped the ship — was still marked unproductive from turn 10
          when it had been closed. Demote it, but NEVER take it off the ballot:
          the arbiter chose it anyway, reasoning "the escape pod bulkhead is
          now open", which is a world change this layer cannot represent.
        - WOULD UNDO: reverses progress just made. Wasteful independently of
          world state, so it is safe to withhold.

        A move that is the next step of a route toward a tracked issue is not
        demoted as an undo at all: a goal-directed return is not aimless
        backtracking, and the undo rule cannot tell them apart on its own.
        """
        if context is None:
            return None, 1.0, False
        if context.is_unproductive(action):
            prior = context.unproductive.get(normalize_command(action), "")
            return (f"  ⚠️ ALREADY TRIED HERE, no effect: \"{prior.strip()[:80]}\"",
                    0.0, False)
        undone = context.undoes_recent_progress(action)
        if undone and not context.is_route_step(action):
            # The backend's accepted-command list is a grammar and contains
            # both halves of every pair, so an agent reading it as advice
            # proposes the inverse of what it just achieved. Observed live:
            # "close grating" at confidence 90, right after opening it.
            return (f"  ⚠️ WOULD UNDO '{undone}' — reverses this turn's own progress",
                    0.0, True)
        return None, 1.0, False

    # LoopDetectionAgent (FIRST - highest priority if loop detected)
    if loop_detection_agent and loop_detection_agent.confidence > 0:
        lines = block()  # no EV of its own; never withheld
        lines.append(f"LoopDetectionAgent: [⚠️ LOOP DETECTED, Confidence: {loop_detection_agent.confidence}/100]")
        lines.append(f"  Loop Type: {loop_detection_agent.loop_type}")
        lines.append(f"  Proposed Action: {loop_detection_agent.proposed_action}")
        lines.append(f"  Reason: {loop_detection_agent.reason}")
        lines.append("")

    # IssueAgents
    for i, agent in enumerate(issue_agents, 1):
        if agent.proposed_action and agent.confidence is not None:
            note, mult, withholdable = repeat_note(agent.proposed_action)
            ev = (agent.importance/1000) * (agent.confidence/100) * 100 * mult
            lines = block(ev, withholdable)
            lines.append(f"IssueAgent #{i}: [Importance: {agent.importance}/1000, Confidence: {agent.confidence}/100, EV: {ev:.1f}]")
            lines.append(f"  Issue: {agent.issue_content}")
            lines.append(f"  Proposed Action: {agent.proposed_action}")
            if note:
                lines.append(note)
            lines.append(f"  Reason: {agent.reason}")
            lines.append("")

    # InteractionAgent (AFTER IssueAgents, BEFORE ExplorerAgent)
    if interaction_agent and interaction_agent.confidence > 0:
        # This agent had NO expected value at all, while both others did, and
        # the arbiter is instructed to rank by expected value — so the only
        # agent that proposes object interactions was structurally unrankable.
        # Observed in pf-20260824: it proposed OPEN escape pod bulkhead (the
        # way off a ship that was about to explode) at confidence 70, and lost
        # to GO UP at EV 47.5 on turn 2.
        #
        # Base is evidence-weighted, mirroring the ExplorerAgent's +3 for a
        # game-confirmed exit: a command the BACKEND listed for an object here
        # is guaranteed to parse and to name something present (#30/#16),
        # which is strictly stronger evidence than an advertised exit — those
        # are refused sometimes. An interaction the model invented gets the
        # weaker base, comparable to exploration's ceiling.
        confirmed = bool(context and context.is_backend_confirmed(
            interaction_agent.proposed_action))
        note, mult, withholdable = repeat_note(interaction_agent.proposed_action)
        base = 100 if confirmed else 50
        ev = (interaction_agent.confidence / 100) * base * mult
        evidence = "game-confirmed" if confirmed else "model-proposed"
        lines = block(ev, withholdable)
        lines.append(f"InteractionAgent: [Confidence: {interaction_agent.confidence}/100, "
                     f"EV: {ev:.1f}, {evidence}]")
        if note:
            lines.append(note)
        if interaction_agent.detected_objects:
            lines.append(f"  Detected Objects: {', '.join(interaction_agent.detected_objects)}")
        lines.append(f"  Proposed Action: {interaction_agent.proposed_action}")
        lines.append(f"  Reason: {interaction_agent.reason}")
        if interaction_agent.inventory_items:
            lines.append(f"  Using Items: {', '.join(interaction_agent.inventory_items)}")
        lines.append("")

    # ExplorerAgent (LAST)
    if explorer_agent and explorer_agent.proposed_action and explorer_agent.confidence is not None:
        note, mult, withholdable = repeat_note(explorer_agent.proposed_action)
        ev = (len(explorer_agent.unexplored_directions)/10) * (explorer_agent.confidence/100) * 50 * mult
        lines = block(ev, withholdable)
        lines.append(f"ExplorerAgent: [Confidence: {explorer_agent.confidence}/100, EV: {ev:.1f}]")
        if note:
            lines.append(note)
        lines.append(f"  Best Direction: {explorer_agent.best_direction}")
        lines.append(f"  Proposed Action: {explorer_agent.proposed_action}")
        lines.append(f"  Reason: {explorer_agent.reason}")
        lines.append(f"  Unexplored Directions: {len(explorer_agent.unexplored_directions)} total")
        lines.append("")

    # A zeroed proposal is WITHHELD while any positive-EV proposal exists.
    #
    # The docstring above says a demoted proposal is annotated rather than
    # removed "so the arbiter can still pick it if literally everything else is
    # exhausted" — but nothing enforced that condition, and the arbiter simply
    # reasoned past the zero. In pf3-20260824 three undo demotions fired and
    # TWO were chosen anyway: "Chose ExplorerAgent (confidence 95, EV 0.0)
    # despite the low EV because the current location is not advancing the
    # score" — while a 70-EV proposal sat unchosen on the same ballot.
    #
    # Milestone 5b's urgency signal is what drives it: told the score has not
    # moved for N turns, the arbiter wants ANY change of direction, and the
    # only movement on offer is the way it just came. Two mechanisms were
    # fighting and the softer one kept winning.
    #
    # This implements the original intent rather than overruling it: when
    # everything else really is exhausted every EV is zero, nothing is
    # withheld, and the annotated proposals remain visible.
    if any(b[0] is not None and b[0] > 0 for b in blocks):
        withheld = [b for b in blocks
                    if b[0] is not None and b[0] <= 0 and b[1] and b[2]]
        if withheld:
            import logging as _logging
            log = _logging.getLogger(__name__)
            for b in withheld:
                log.info(f"[Decide] withheld zero-EV undo proposal: {b[1][0]}")
            blocks = [b for b in blocks if b not in withheld]

    rendered = [line for b in blocks for line in b[1]]
    return "\n".join(rendered) if rendered else "No proposals available. Choose LOOK to observe the current situation."


def create_close_issues_node(decision_llm, history_toolkit: HistoryToolkit,
                            memory_toolkit: MemoryToolkit):
    """
    Create the issue closing node that identifies and removes resolved issues.

    This node runs AFTER the decision is made and BEFORE the observer identifies new issues.
    It analyzes recent history to close issues that have been solved.

    Args:
        decision_llm: The LLM to use for analysis
        history_toolkit: HistoryToolkit for accessing recent game history
        memory_toolkit: MemoryToolkit for removing resolved issues

    Returns:
        Node function for the graph
    """
    async def close_issues_node(state: DecisionState) -> dict:
        """
        Issue closing phase: Identify and remove resolved issues from memory.
        """
        import logging
        logger = logging.getLogger(__name__)

        zork_response = state["game_response"]

        logger.info("\n" + "=" * 80)
        logger.info("CLOSE ISSUES - Identifying resolved issues")
        logger.info("=" * 80)
        logger.info(f"Analyzing recent history at {zork_response.LocationName}")

        # Create IssueClosedAgent to analyze recent history
        issue_closer = IssueClosedAgent()

        # Bookkeeping runs AFTER the decision is already made: a failure here
        # must not throw away a usable command (see #1). Downstream consumers
        # all treat a None response as "nothing closed this turn".
        try:
            issue_closed_response, pending_closures = await issue_closer.analyze(
                game_response=zork_response.Response,
                location=zork_response.LocationName or "Unknown",
                score=zork_response.Score,
                moves=zork_response.Moves,
                decision_llm=decision_llm,
                history_toolkit=history_toolkit,
                memory_toolkit=memory_toolkit,
                current_turn=state.get("turn_number"),
            )
        except Exception as e:
            logger.error(f"CLOSE ISSUES failed, skipping this turn: {e}", exc_info=True)
            issue_closed_response = None
            pending_closures = []

        logger.info("=" * 80)
        logger.info("CLOSE ISSUES COMPLETE")
        logger.info("=" * 80)
        # Staged, not applied: persist_node commits these last (#3).
        return {
            "issue_closed_response": issue_closed_response,
            "pending_closures": pending_closures,
        }

    return close_issues_node


def create_observe_node(decision_llm, history_toolkit: HistoryToolkit, memory_toolkit: MemoryToolkit):
    """
    Create the observation node that identifies new strategic issues.

    This node runs AFTER the decision is made and analyzes the game response
    to identify new puzzles, obstacles, or items.

    Args:
        decision_llm: The LLM to use for observation
        history_toolkit: HistoryToolkit for accessing game history
        memory_toolkit: MemoryToolkit for accessing tracked issues

    Returns:
        Node function for the graph
    """
    async def observe_node(state: DecisionState) -> dict:
        """
        Observation phase: Identify new strategic issues from game response.
        """
        import logging
        logger = logging.getLogger(__name__)

        zork_response = state["game_response"]

        logger.info("\n" + "=" * 80)
        logger.info("OBSERVE - Identifying new strategic issues")
        logger.info("=" * 80)
        logger.info(f"Analyzing game response at {zork_response.LocationName}")

        # Create ObserverAgent to analyze the game response
        observer = ObserverAgent()

        # Post-decision bookkeeping: a failure here must not discard the
        # command already chosen (see #1). persist_node and GameSession both
        # handle a None observer_response as "no new issue this turn".
        try:
            observer_response = await observer.observe(
                game_response=zork_response.Response,
                location=zork_response.LocationName or "Unknown",
                score=zork_response.Score,
                moves=zork_response.Moves,
                decision_llm=decision_llm,
                memory_toolkit=memory_toolkit,
                context=state.get("turn_context"),
            )
        except Exception as e:
            logger.error(f"OBSERVE failed, skipping this turn: {e}", exc_info=True)
            observer_response = None

        logger.info("=" * 80)
        logger.info("OBSERVE COMPLETE")
        logger.info("=" * 80)
        return {"observer_response": observer_response}

    return observe_node


def create_persist_node(memory_toolkit: MemoryToolkit, inventory_toolkit):
    """
    Create the persistence node that stores strategic issues and updates inventory.

    Args:
        memory_toolkit: MemoryToolkit for storing strategic issues
        inventory_toolkit: InventoryToolkit for tracking inventory

    Returns:
        Node function for the graph
    """
    def persist_node(state: DecisionState) -> dict:
        """
        Persistence phase: Store strategic issues identified by Observer Agent.
        """
        import logging
        logger = logging.getLogger(__name__)

        observer_response = state.get("observer_response")
        zork_response = state["game_response"]

        logger.info("\n" + "=" * 80)
        logger.info("PERSIST - Saving new issues to memory and decaying old ones")
        logger.info("=" * 80)

        memory_persisted = False

        if observer_response is None:
            # Observation failed or was skipped this turn (see #1). Nothing to
            # store, but the inventory update below must still run.
            logger.info("NO OBSERVER RESPONSE - nothing to store this turn")
        else:
            logger.info(f"Observer.remember: '{observer_response.remember}'")
            logger.info(f"Observer.rememberImportance: {observer_response.rememberImportance}")
            logger.info(f"Observer.item: '{observer_response.item}'")

        has_memory = bool(
            observer_response is not None
            and observer_response.remember
            and observer_response.remember.strip()
        )

        if has_memory:
            logger.info(f"ATTEMPTING TO STORE MEMORY: [{observer_response.rememberImportance}/1000] {observer_response.remember}")
            try:
                was_added = memory_toolkit.add_memory(
                    content=observer_response.remember,
                    importance=observer_response.rememberImportance or 500,
                    turn_number=state.get("turn_number"),
                    # Empty, not "Unknown": a memory anchored to a fake room
                    # sends every later IssueAgent pathfinding to nowhere (#7).
                    location=zork_response.LocationName or "",
                    score=zork_response.Score,
                    moves=zork_response.Moves
                )
            except Exception as e:
                logger.error(f"MEMORY STORAGE RAISED, continuing turn: {e}", exc_info=True)
                was_added = False
            memory_persisted = was_added

            # Log summary
            logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            if was_added:
                logger.info(f"MEMORY STORED: [{observer_response.rememberImportance}/1000] {observer_response.remember}")
            else:
                logger.info(f"MEMORY STORAGE FAILED (duplicate?): [{observer_response.rememberImportance}/1000] {observer_response.remember}")
            logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        elif observer_response is not None:
            logger.info("NO MEMORY TO STORE (remember field empty or whitespace)")

        # Decay is now applied lazily on read (see MemoryState.get_top_memories);
        # no per-turn UPDATE is needed here.

        # Update inventory based on this turn
        logger.info("\n" + "-" * 80)
        logger.info("UPDATING INVENTORY")
        logger.info("-" * 80)

        # The backend reports its own inventory (#30). It is ground truth —
        # a failed TAKE leaves it unchanged, a successful one updates it — so
        # when it is present we reconcile against it and skip the LLM analyzer
        # entirely. That removes one cheap-model call from every turn and ends
        # the whole class of drift #21 was fighting: no name matching, no
        # add/remove inference, no phantom items.
        api_inventory = getattr(zork_response, "Inventory", None)
        if api_inventory is not None:
            try:
                inventory_toolkit.state.sync_with_game(
                    api_inventory, state.get("turn_number")
                )
                logger.info(
                    f"Inventory synced from the game itself "
                    f"({len(api_inventory)} items): {api_inventory}"
                )
            except Exception as e:
                logger.error(f"INVENTORY SYNC FAILED, inventory may be stale: {e}", exc_info=True)
            _apply_pending_closures(state, memory_toolkit, logger)
            logger.info("=" * 80)
            logger.info("PERSIST COMPLETE")
            logger.info("=" * 80)
            return {"memory_persisted": memory_persisted}

        player_command = state.get("player_command")
        if player_command:
            from tools.inventory import InventoryAnalyzer
            from config import get_cheap_llm

            # Inventory analysis is post-decision bookkeeping: a failed LLM
            # call or malformed structured output must not cost us the turn
            # (see #1). Stale inventory is recoverable; a dead session is not.
            try:
                # Use cheap LLM for inventory analysis
                analyzer = InventoryAnalyzer(get_cheap_llm(temperature=0))

                # Analyze turn for inventory changes
                # Give the analyzer the held items so it can name a removal
                # using the string we actually store: the player types
                # "DROP LAMP" while the DB holds "brass lantern" (#21).
                changes = analyzer.analyze_turn(
                    player_command=player_command,
                    game_response=zork_response.Response,
                    current_inventory=inventory_toolkit.state.get_items(),
                )

                logger.info(f"Items added: {changes.items_added}")
                logger.info(f"Items removed: {changes.items_removed}")
                logger.info(f"Reasoning: {changes.reasoning}")

                # Apply changes to inventory state
                for item in changes.items_added:
                    inventory_toolkit.state.add_item(item, state.get("turn_number"))

                for item in changes.items_removed:
                    inventory_toolkit.state.remove_item(item, state.get("turn_number"))

                current_inventory = inventory_toolkit.state.get_items()
                logger.info(f"Current inventory ({len(current_inventory)} items): {current_inventory}")
            except Exception as e:
                logger.error(f"INVENTORY UPDATE FAILED, inventory may be stale: {e}", exc_info=True)
        else:
            logger.info("No command to analyze (decision was None)")

        logger.info("-" * 80)

        _apply_pending_closures(state, memory_toolkit, logger)

        logger.info("=" * 80)
        logger.info("PERSIST COMPLETE")
        logger.info("=" * 80)
        return {"memory_persisted": memory_persisted}

    return persist_node


def _apply_pending_closures(state, memory_toolkit, logger) -> None:
        # Apply the closures staged by close_issues_node — LAST, after every
        # cancellable LLM call in this turn (#3). Closing an issue is the only
        # destructive memory write in the graph; doing it here means a turn
        # killed by the budget leaves memory untouched instead of half-applied.
        # Anything not closed stays open and is simply re-detected next turn.
        pending_closures = state.get("pending_closures") or []
        issue_closed_response = state.get("issue_closed_response")

        # REFUSE a closure the game's own transcript contradicts.
        #
        # The IssueClosedAgent closed the ESCAPE POD — the objective — in five
        # of six Planetfall runs, always shortly after the turn-2 refusal "Why
        # open the door to the emergency escape pod if there's no emergency?",
        # reading a TEMPORAL refusal ("not yet") as resolution ("done"). pf4
        # escaped with every one of its four issues closed, so the memory
        # system contributed nothing to either success.
        #
        # Not fixable by wording: the closure prompt already parses the
        # acceptance criteria, says "ONLY close if the acceptance criteria is
        # SATISFIED", and carries explicit DO-NOT-CLOSE examples.
        #
        # An earlier version of this guard deferred closures on any turn that
        # changed nothing. That correctly caught the turn-2 refusal and still
        # lost the pod on turn 3, because the closer re-stages the same closure
        # every turn and the next turn was a move. A TURN-level guard cannot
        # fix an ISSUE-level misjudgement. So the question asked here is about
        # the issue: has its action already been tried, in the room it applies
        # to, and accomplished nothing?
        context = state.get("turn_context")
        if pending_closures and context is not None:
            memories_by_id = {m.id: m for m in (state.get("memories") or [])}
            survivors = []
            for closure in pending_closures:
                memory = memories_by_id.get(closure.get("id"))
                if memory is None:
                    survivors.append(closure)
                    continue
                target = resolve_issue_target(
                    memory.content, memory.location, context.known_locations)
                disproof = closure_is_contradicted(
                    memory.content, target, context.recent_turn_records)
                if disproof:
                    logger.info(
                        f"REFUSED closure of ID {closure.get('id')}: the "
                        f"transcript contradicts it — its action was tried at "
                        f"{target} and did nothing: \"{disproof[:80]}\""
                    )
                    continue
                survivors.append(closure)
            pending_closures = survivors

        # SECONDARY, and deliberately kept after the evidence check above: a
        # turn that changed nothing cannot have resolved anything either. On
        # its own this is insufficient — it merely postpones a wrong closure to
        # the next turn that moves — but it costs one turn's delay at most and
        # catches issues whose criteria is too terse for the transcript check
        # to match on.
        if pending_closures and context is not None and not context.accomplished_something:
            logger.info(
                f"DEFERRED {len(pending_closures)} closure(s): this turn changed "
                f"nothing (no move, no score, no inventory change)"
            )
            pending_closures = []

        if pending_closures:
            logger.info("\n" + "-" * 80)
            logger.info(f"APPLYING {len(pending_closures)} STAGED ISSUE CLOSURE(S)")
            logger.info("-" * 80)

            closed_contents = []
            for closure in pending_closures:
                issue_id = closure.get("id")
                display = closure.get("display")
                # IssueClosedAgent stages a display string for every ID it
                # validated against what the model was shown (#19). No display
                # means this closure bypassed that path — refuse the write.
                if not display:
                    logger.warning(
                        f"SKIPPED unvalidated closure for ID {issue_id} (no display text)"
                    )
                    continue
                try:
                    success = memory_toolkit.state.remove_memory(issue_id)
                except Exception as e:
                    logger.error(f"CLOSE FAILED for ID {issue_id}: {e}", exc_info=True)
                    continue

                if success:
                    logger.info(f"[OK] CLOSED ID {issue_id}: '{display}'")
                    if display:
                        closed_contents.append(display)
                else:
                    logger.warning(f"[FAIL] Database close failed for ID {issue_id}: '{display}'")

            # Report only what actually committed.
            if issue_closed_response is not None:
                issue_closed_response.closed_issue_contents = closed_contents



def create_decision_graph(
    decision_chain: Runnable,
    decision_llm,
    history_toolkit: HistoryToolkit,
    memory_toolkit: MemoryToolkit,
    mapper_toolkit: MapperToolkit,
    inventory_toolkit,
):
    """
    Build the decision-making graph.

    Flow:
        BuildContext → (SpawnAgents → Decide | CloseIssues | Observe) → Persist → END

    Args:
        decision_chain: Decision chain with structured output
        decision_llm: LLM for IssueAgent, ExplorerAgent proposals, Observer, and IssueClosedAgent
        history_toolkit: History toolkit for tool execution
        memory_toolkit: Memory toolkit for persistence and issue agent spawning
        mapper_toolkit: Mapper toolkit for ExplorerAgent spawning

    Returns:
        Compiled LangGraph
    """
    graph = StateGraph(DecisionState)

    # Add nodes
    graph.add_node("build_context", create_build_context_node(
        memory_toolkit,
        mapper_toolkit,
        inventory_toolkit,
        history_toolkit,
    ))
    graph.add_node("spawn_agents", create_spawn_agents_node(
        memory_toolkit,
        mapper_toolkit,
        inventory_toolkit,
        decision_llm,
        history_toolkit,
    ))
    graph.add_node("decide", create_decision_node(decision_chain))
    graph.add_node("close_issues", create_close_issues_node(
        decision_llm, history_toolkit, memory_toolkit))
    graph.add_node("observe", create_observe_node(decision_llm, history_toolkit, memory_toolkit))
    graph.add_node("persist", create_persist_node(memory_toolkit, inventory_toolkit))

    # Define flow
    # Flow (#23, #26):
    #
    #   build_context ─┬─ spawn_agents → decide ─┐
    #                  ├─ close_issues ──────────┤
    #                  └─ observe ───────────────┴─ persist → END
    #
    # close_issues and observe were chained AFTER decide, so ~20% of every
    # turn was spent on bookkeeping once the command was already chosen. They
    # have no data dependency on the decision at all — both read the game
    # response, which is available at the top of the turn — so they now run
    # BESIDE the spawn→decide chain and finish inside its shadow.
    #
    # This is also the first real use of the graph: LangGraph runs the three
    # branches in one parallel super-step and joins them at persist. It was
    # previously a six-node straight line that would have behaved identically
    # as six sequential awaits.
    #
    # Ordering safety: the parallel branches are READ-ONLY with respect to
    # memory. close_issues stages its closures rather than applying them (#3),
    # and persist — which every branch joins — remains the single writer.
    graph.set_entry_point("build_context")
    graph.add_edge("build_context", "spawn_agents")
    graph.add_edge("build_context", "close_issues")
    graph.add_edge("build_context", "observe")
    graph.add_edge("spawn_agents", "decide")
    # A LIST start_key is a real join: LangGraph waits for ALL three branches.
    # Adding the three edges separately does NOT do this — the branches have
    # different depths (spawn->decide is two hops, close and observe are one),
    # so persist was scheduled in the super-step where close and observe
    # finished AND AGAIN when decide finished. Verified live: PERSIST ran
    # twice per turn.
    graph.add_edge(["decide", "close_issues", "observe"], "persist")
    graph.add_edge("persist", END)

    return graph.compile()

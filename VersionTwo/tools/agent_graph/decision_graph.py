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
from .turn_context import build_turn_context


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
        return {"turn_context": context, "memories": memories_sorted}

    return build_context_node


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
    async def spawn_agents_node(state: DecisionState) -> DecisionState:
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

        # Create one IssueAgent for each issue (max 5)
        issue_agents = [IssueAgent(memory=mem) for mem in memories_sorted]

        logger.info(f"SPAWNED {len(issue_agents)} IssueAgents (top 5 by importance)")

        context = state["turn_context"]

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
        all_agents = [a for a in issue_agents + [explorer_agent, loop_detection_agent, interaction_agent] if a is not None]

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
        agent_proposals_text = _format_agent_proposals(issue_agents, explorer_agent, loop_detection_agent, interaction_agent)
        logger.info(f"Agent Proposals:\n{agent_proposals_text}")
        logger.info("=" * 80)

        # No additional tool-calling pass here: research_node + per-agent research
        # already gathered sufficient context. Keep tool_calls_history empty for the
        # report writer's compatibility.
        tool_calls_history: list = []
        full_research_context = research_context

        decision_input = {
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


def _format_agent_proposals(issue_agents, explorer_agent, loop_detection_agent, interaction_agent):
    """Format agent proposals for Decision Agent evaluation"""
    lines = []

    # LoopDetectionAgent (FIRST - highest priority if loop detected)
    if loop_detection_agent and loop_detection_agent.confidence > 0:
        lines.append(f"LoopDetectionAgent: [⚠️ LOOP DETECTED, Confidence: {loop_detection_agent.confidence}/100]")
        lines.append(f"  Loop Type: {loop_detection_agent.loop_type}")
        lines.append(f"  Proposed Action: {loop_detection_agent.proposed_action}")
        lines.append(f"  Reason: {loop_detection_agent.reason}")
        lines.append("")

    # IssueAgents
    for i, agent in enumerate(issue_agents, 1):
        if agent.proposed_action and agent.confidence is not None:
            ev = (agent.importance/1000) * (agent.confidence/100) * 100
            lines.append(f"IssueAgent #{i}: [Importance: {agent.importance}/1000, Confidence: {agent.confidence}/100, EV: {ev:.1f}]")
            lines.append(f"  Issue: {agent.issue_content}")
            lines.append(f"  Proposed Action: {agent.proposed_action}")
            lines.append(f"  Reason: {agent.reason}")
            lines.append("")

    # InteractionAgent (AFTER IssueAgents, BEFORE ExplorerAgent)
    if interaction_agent and interaction_agent.confidence > 0:
        lines.append(f"InteractionAgent: [Confidence: {interaction_agent.confidence}/100]")
        if interaction_agent.detected_objects:
            lines.append(f"  Detected Objects: {', '.join(interaction_agent.detected_objects)}")
        lines.append(f"  Proposed Action: {interaction_agent.proposed_action}")
        lines.append(f"  Reason: {interaction_agent.reason}")
        if interaction_agent.inventory_items:
            lines.append(f"  Using Items: {', '.join(interaction_agent.inventory_items)}")
        lines.append("")

    # ExplorerAgent (LAST)
    if explorer_agent and explorer_agent.proposed_action and explorer_agent.confidence is not None:
        ev = (len(explorer_agent.unexplored_directions)/10) * (explorer_agent.confidence/100) * 50
        lines.append(f"ExplorerAgent: [Confidence: {explorer_agent.confidence}/100, EV: {ev:.1f}]")
        lines.append(f"  Best Direction: {explorer_agent.best_direction}")
        lines.append(f"  Proposed Action: {explorer_agent.proposed_action}")
        lines.append(f"  Reason: {explorer_agent.reason}")
        lines.append(f"  Unexplored Directions: {len(explorer_agent.unexplored_directions)} total")
        lines.append("")

    return "\n".join(lines) if lines else "No proposals available. Choose LOOK to observe the current situation."


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

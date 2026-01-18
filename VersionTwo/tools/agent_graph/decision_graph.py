"""
Simple LangGraph for managing the decision-making flow.

Flow: SpawnAgents → Research → Decide → CloseIssues → Observe → Persist → END

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
from langchain_core.runnables import Runnable
from .issue_agent import IssueAgent
from .explorer_agent import ExplorerAgent
from .loop_detection_agent import LoopDetectionAgent
from .interaction_agent import InteractionAgent
from .issue_closed_agent import IssueClosedAgent
from .issue_closed_response import IssueClosedResponse
from .observer_agent import ObserverAgent
from .observer_response import ObserverResponse


class DecisionState(TypedDict):
    """State passed through the decision graph"""

    # Input
    game_response: ZorkApiResponse

    # Spawn phase output
    issue_agents: List[IssueAgent]
    explorer_agent: Optional[ExplorerAgent]  # Single agent, can be None
    loop_detection_agent: Optional[LoopDetectionAgent]  # Single agent, always spawned
    interaction_agent: Optional[InteractionAgent]  # Single agent, always spawned

    # Research phase output
    research_context: str

    # Decision phase output
    decision: Optional[AdventurerResponse]
    decision_prompt: str  # Formatted prompt for reporting

    # Issue closing phase output
    issue_closed_response: Optional[IssueClosedResponse]

    # Observation phase output
    observer_response: Optional[ObserverResponse]

    # Persistence tracking
    memory_persisted: bool


def create_spawn_agents_node(
    memory_toolkit: MemoryToolkit,
    mapper_toolkit: MapperToolkit,
    inventory_toolkit,
    research_agent: Runnable,
    decision_llm,
    history_toolkit: HistoryToolkit
):
    """
    Create the spawn agents node that creates IssueAgents and ExplorerAgent.

    Args:
        memory_toolkit: MemoryToolkit for accessing stored strategic issues
        mapper_toolkit: MapperToolkit for accessing map state
        inventory_toolkit: InventoryToolkit for accessing inventory
        research_agent: Research agent with tools for IssueAgents and ExplorerAgent to use
        decision_llm: LLM for generating proposals
        history_toolkit: HistoryToolkit for accessing tools

    Returns:
        Node function for the graph
    """
    def spawn_agents_node(state: DecisionState) -> DecisionState:
        """
        Spawn phase: Create one IssueAgent for each tracked strategic issue.
        Each agent performs its own research and generates a proposal IN PARALLEL.
        """
        import logging
        import asyncio
        logger = logging.getLogger(__name__)

        try:
            logger.info("\n" + "=" * 80)
            logger.info("SPAWN AGENTS - Creating specialized agents for this turn")
            logger.info("=" * 80)

            # Get top 5 tracked issues from database (ordered by importance)
            memories = memory_toolkit.state.get_top_memories(limit=5)
            logger.info(f"Retrieved {len(memories)} memories from database")

            # Sort by location name for cleaner console display
            memories_sorted = sorted(memories, key=lambda m: m.location if m.location else "")

            # Create one IssueAgent for each issue (max 5)
            issue_agents = [IssueAgent(memory=mem) for mem in memories_sorted]

            logger.info(f"SPAWNED {len(issue_agents)} IssueAgents (top 5 by importance)")

            # Extract current game state
            game_response = state["game_response"]
            current_location = game_response.LocationName or "Unknown"
            current_game_text = game_response.Response
            current_score = game_response.Score
            current_moves = game_response.Moves

            # ========== NEW: Spawn ONE ExplorerAgent (if unexplored directions exist) ==========
            # Get known exits from current location
            known_exits = mapper_toolkit.state.get_exits_from(current_location)
            known_directions = {direction.upper() for direction, _ in known_exits}

            # Determine unexplored directions
            CARDINAL_DIRECTIONS = [
                "NORTH", "SOUTH", "EAST", "WEST",
                "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST",
                "UP", "DOWN"
            ]

            unexplored_directions = [
                d for d in CARDINAL_DIRECTIONS
                if d not in known_directions
            ]

            # Parse game text to see which directions are mentioned
            game_text_upper = current_game_text.upper()
            mentioned_directions = []

            # Direction alias mapping
            DIRECTION_ALIASES = {
                "NORTH": ["NORTH", "NORTHERN", " N "],
                "SOUTH": ["SOUTH", "SOUTHERN", " S "],
                "EAST": ["EAST", "EASTERN", " E "],
                "WEST": ["WEST", "WESTERN", " W "],
                "NORTHEAST": ["NORTHEAST", "NE"],
                "NORTHWEST": ["NORTHWEST", "NW"],
                "SOUTHEAST": ["SOUTHEAST", "SE"],
                "SOUTHWEST": ["SOUTHWEST", "SW"],
                "UP": ["UP", "ABOVE", "UPWARD"],
                "DOWN": ["DOWN", "BELOW", "DOWNWARD"],
            }

            for direction in unexplored_directions:
                for alias in DIRECTION_ALIASES.get(direction, [direction]):
                    if alias in game_text_upper:
                        mentioned_directions.append(direction)
                        break  # Only count each direction once

            # Create ONE ExplorerAgent if there are unexplored directions
            explorer_agent = None
            if unexplored_directions:
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

            # Get tools (include inventory and analysis for agents to query)
            history_tools = history_toolkit.get_tools()
            mapper_tools = mapper_toolkit.get_tools()
            inventory_tools = inventory_toolkit.get_tools()

            # Get analysis tools (big picture strategic analysis)
            from tools.analysis import get_analysis_tools
            analysis_tools = get_analysis_tools()

            # Combine all tools for IssueAgents (they use combined tools)
            all_tools = history_tools + mapper_tools + inventory_tools + analysis_tools

            # Execute all agent research in parallel using threads
            def research_agent_sync(agent):
                """Synchronous research function to be run in thread"""
                try:
                    if isinstance(agent, ExplorerAgent):
                        logger.info(f"Starting research for ExplorerAgent: {agent.best_direction}")
                        agent.research_and_propose(
                            research_agent=research_agent,
                            decision_llm=decision_llm,
                            history_tools=history_tools,
                            mapper_tools=mapper_tools,
                            current_game_response=current_game_text,
                            current_score=current_score,
                            current_moves=current_moves
                        )
                        logger.info(f"Research complete for ExplorerAgent -> {agent.proposed_action}")
                    elif isinstance(agent, LoopDetectionAgent):
                        logger.info(f"Starting research for LoopDetectionAgent")
                        agent.research_and_propose(
                            research_agent=research_agent,
                            decision_llm=decision_llm,
                            history_tools=history_tools,
                            mapper_tools=mapper_tools,
                            current_location=current_location,
                            current_game_response=current_game_text,
                            current_score=current_score,
                            current_moves=current_moves
                        )
                        logger.info(f"Research complete for LoopDetectionAgent -> {agent.proposed_action} (conf: {agent.confidence})")
                    elif isinstance(agent, InteractionAgent):
                        logger.info(f"Starting research for InteractionAgent")
                        agent.research_and_propose(
                            research_agent=research_agent,
                            decision_llm=decision_llm,
                            history_tools=history_tools,
                            mapper_tools=mapper_tools,
                            current_location=current_location,
                            current_game_response=current_game_text,
                            current_score=current_score,
                            current_moves=current_moves,
                            inventory_tools=inventory_tools  # InteractionAgent needs inventory tools
                        )
                        logger.info(f"Research complete for InteractionAgent -> {agent.proposed_action} (conf: {agent.confidence})")
                    else:  # IssueAgent
                        logger.info(f"Starting research for IssueAgent: '{agent.issue_content}'")
                        agent.research_and_propose(
                            research_agent=research_agent,
                            decision_llm=decision_llm,
                            history_tools=all_tools,  # IssueAgents get all tools including inventory
                            current_location=current_location,
                            current_game_response=current_game_text,
                            current_score=current_score,
                            current_moves=current_moves
                        )
                        logger.info(f"Research complete for: '{agent.issue_content}' -> {agent.proposed_action}")
                except Exception as e:
                    logger.error(f"ERROR in agent research:")
                    logger.error(f"  Exception type: {type(e).__name__}")
                    logger.error(f"  Exception message: {str(e)}")
                    import traceback
                    logger.error(f"  Traceback:\n{traceback.format_exc()}")
                    # Set defaults if research fails
                    agent.proposed_action = "nothing"
                    agent.confidence = 0
                    agent.reason = f"Research failed: {str(e)}"

            # Combine all agents (IssueAgents + ExplorerAgent + LoopDetectionAgent + InteractionAgent)
            all_agents = issue_agents + ([explorer_agent] if explorer_agent else []) + [loop_detection_agent, interaction_agent]

            # Run all agents in parallel using threads
            if all_agents:
                import threading
                threads = []
                for agent in all_agents:
                    thread = threading.Thread(target=research_agent_sync, args=(agent,))
                    thread.start()
                    threads.append(thread)

                # Wait for all threads to complete
                for thread in threads:
                    thread.join()

            logger.info(f"All {len(all_agents)} agents completed research in PARALLEL")
            logger.info("=" * 80)
            logger.info("SPAWN AGENTS COMPLETE")
            logger.info("=" * 80)

            state["issue_agents"] = issue_agents
            state["explorer_agent"] = explorer_agent  # Single agent, can be None
            state["loop_detection_agent"] = loop_detection_agent  # Always present
            state["interaction_agent"] = interaction_agent  # Always present
            return state

        except Exception as e:
            logger.error(f"CRITICAL ERROR in spawn_agents_node: {e}", exc_info=True)
            # Return empty agents on error
            state["issue_agents"] = []
            state["explorer_agent"] = None
            state["loop_detection_agent"] = None
            state["interaction_agent"] = None
            return state

    return spawn_agents_node


def create_research_node(research_agent: Runnable, history_toolkit: HistoryToolkit):
    """
    Create the research node that calls history tools.

    Args:
        research_agent: The LangChain research agent chain
        history_toolkit: HistoryToolkit for executing tools

    Returns:
        Node function for the graph
    """
    def research_node(state: DecisionState) -> DecisionState:
        """
        Research phase: Call history tools to gather context.
        """
        import logging
        logger = logging.getLogger(__name__)

        zork_response = state["game_response"]

        research_input = {
            "input": "Use the available tools to gather relevant history context.",
            "score": zork_response.Score,
            "locationName": zork_response.LocationName,
            "moves": zork_response.Moves,
            "game_response": zork_response.Response
        }

        logger.info("\n" + "=" * 80)
        logger.info("[ResearchAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("[ResearchAgent] AGENT: ResearchAgent")
        logger.info("[ResearchAgent] PURPOSE: Gather historical context for decision making")
        logger.info(f"[ResearchAgent] LOCATION: {zork_response.LocationName}")
        logger.info("[ResearchAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("RESEARCH - Gathering historical context")
        logger.info("=" * 80)
        logger.info(f"Current location: {zork_response.LocationName}")
        logger.info(f"Current game response (first 100): {zork_response.Response[:100]}...")

        # Call research agent with timeout and retry
        from llm_utils import invoke_with_retry
        response = invoke_with_retry(
            research_agent,
            research_input,
            operation_name="Research Agent"
        )

        # Execute tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            # Include analysis tools alongside history tools
            from tools.analysis import get_analysis_tools
            all_research_tools = history_toolkit.get_tools() + get_analysis_tools()
            tools_map = {tool.name: tool for tool in all_research_tools}

            logger.info(f"[ResearchAgent] Made {len(response.tool_calls)} tool calls:")

            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})

                logger.info(f"[ResearchAgent]   -> {tool_name}({tool_args})")

                if tool_name in tools_map:
                    tool_result = tools_map[tool_name].invoke(tool_args)
                    logger.info(f"[ResearchAgent]      Result: {str(tool_result)[:150]}...")
                    tool_results.append(f"{tool_name} result: {tool_result}")

            # Combine tool results into summary
            research_context = "\n\n".join(tool_results) if tool_results else "No tools executed."
        else:
            # If no tool calls, use the content directly
            logger.info("[ResearchAgent] No tool calls made")
            research_context = response.content if hasattr(response, 'content') else str(response)

        logger.info(f"Research context length: {len(research_context)} chars")
        logger.info(f"Research context (first 300): {research_context[:300]}...")
        logger.info("=" * 80)
        logger.info("RESEARCH COMPLETE")
        logger.info("=" * 80)

        state["research_context"] = research_context
        return state

    return research_node


def create_decision_node(decision_chain: Runnable):
    """
    Create the decision node that generates structured output.

    Args:
        decision_chain: The LangChain decision chain with structured output

    Returns:
        Node function for the graph
    """
    def decision_node(state: DecisionState) -> DecisionState:
        """
        Decision phase: Generate AdventurerResponse based on agent proposals.
        """
        import logging
        logger = logging.getLogger(__name__)

        zork_response = state["game_response"]
        research_context = state["research_context"]
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
        logger.info("DECISION - Choosing best action from agent proposals")
        logger.info("=" * 80)
        logger.info(f"Location: {zork_response.LocationName}")
        logger.info(f"Score: {zork_response.Score}, Moves: {zork_response.Moves}")
        logger.info(f"Game Response (first 100): {zork_response.Response[:100]}...")
        logger.info(f"Research Context (first 500):\n{research_context[:500]}...")

        # Format agent proposals for Decision Agent
        agent_proposals_text = _format_agent_proposals(issue_agents, explorer_agent, loop_detection_agent, interaction_agent)
        logger.info(f"Agent Proposals:\n{agent_proposals_text}")
        logger.info("=" * 80)

        decision_input = {
            "score": zork_response.Score,
            "locationName": zork_response.LocationName,
            "moves": zork_response.Moves,
            "game_response": zork_response.Response,
            "research_context": research_context,
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
            research_context=research_context,
            agent_proposals=agent_proposals_text
        )

        # Combine system + human for full prompt
        full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[HUMAN]\n{formatted_human}"

        # Invoke decision chain with timeout and retry
        from llm_utils import invoke_with_retry
        decision = invoke_with_retry(
            decision_chain.with_config(run_name="Decision Agent"),
            decision_input,
            operation_name="Decision Agent"
        )

        logger.info(f"DECISION MADE: {decision.command}")
        logger.info(f"REASON: {decision.reason}")

        state["decision"] = decision
        state["decision_prompt"] = full_prompt
        return state

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


def create_close_issues_node(decision_llm, history_toolkit: HistoryToolkit, memory_toolkit: MemoryToolkit):
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
    def close_issues_node(state: DecisionState) -> DecisionState:
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

        # Analyze recent history and close resolved issues
        issue_closed_response = issue_closer.analyze(
            game_response=zork_response.Response,
            location=zork_response.LocationName or "Unknown",
            score=zork_response.Score,
            moves=zork_response.Moves,
            decision_llm=decision_llm,
            history_toolkit=history_toolkit,
            memory_toolkit=memory_toolkit
        )

        state["issue_closed_response"] = issue_closed_response

        logger.info("=" * 80)
        logger.info("CLOSE ISSUES COMPLETE")
        logger.info("=" * 80)
        return state

    return close_issues_node


def create_observe_node(decision_llm, research_agent: Runnable, history_toolkit: HistoryToolkit, memory_toolkit: MemoryToolkit):
    """
    Create the observation node that identifies new strategic issues.

    This node runs AFTER the decision is made and analyzes the game response
    to identify new puzzles, obstacles, or items.

    Args:
        decision_llm: The LLM to use for observation
        research_agent: Research agent with access to history tools
        history_toolkit: HistoryToolkit for accessing game history
        memory_toolkit: MemoryToolkit for accessing tracked issues

    Returns:
        Node function for the graph
    """
    def observe_node(state: DecisionState) -> DecisionState:
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

        # Get history tools
        history_tools = history_toolkit.get_tools()

        # Analyze game response and identify new issues (with full context)
        observer_response = observer.observe(
            game_response=zork_response.Response,
            location=zork_response.LocationName or "Unknown",
            score=zork_response.Score,
            moves=zork_response.Moves,
            decision_llm=decision_llm,
            research_agent=research_agent,
            history_tools=history_tools,
            memory_toolkit=memory_toolkit
        )

        state["observer_response"] = observer_response

        logger.info("=" * 80)
        logger.info("OBSERVE COMPLETE")
        logger.info("=" * 80)
        return state

    return observe_node


def create_persist_node(memory_toolkit: MemoryToolkit, inventory_toolkit, turn_number_ref: dict):
    """
    Create the persistence node that stores strategic issues and updates inventory.

    Args:
        memory_toolkit: MemoryToolkit for storing strategic issues
        inventory_toolkit: InventoryToolkit for tracking inventory
        turn_number_ref: Mutable dict with current turn number

    Returns:
        Node function for the graph
    """
    def persist_node(state: DecisionState) -> DecisionState:
        """
        Persistence phase: Store strategic issues identified by Observer Agent.
        """
        import logging
        logger = logging.getLogger(__name__)

        observer_response = state["observer_response"]
        zork_response = state["game_response"]

        logger.info("\n" + "=" * 80)
        logger.info("PERSIST - Saving new issues to memory and decaying old ones")
        logger.info("=" * 80)
        logger.info(f"Observer.remember: '{observer_response.remember}'")
        logger.info(f"Observer.rememberImportance: {observer_response.rememberImportance}")
        logger.info(f"Observer.item: '{observer_response.item}'")

        if observer_response.remember and observer_response.remember.strip():
            logger.info(f"ATTEMPTING TO STORE MEMORY: [{observer_response.rememberImportance}/1000] {observer_response.remember}")
            was_added = memory_toolkit.add_memory(
                content=observer_response.remember,
                importance=observer_response.rememberImportance or 500,
                turn_number=turn_number_ref["current"],
                location=zork_response.LocationName or "Unknown",
                score=zork_response.Score,
                moves=zork_response.Moves
            )
            state["memory_persisted"] = was_added

            # Log summary
            logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            if was_added:
                logger.info(f"MEMORY STORED: [{observer_response.rememberImportance}/1000] {observer_response.remember}")
            else:
                logger.info(f"MEMORY STORAGE FAILED (duplicate?): [{observer_response.rememberImportance}/1000] {observer_response.remember}")
            logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        else:
            logger.info("NO MEMORY TO STORE (remember field empty or whitespace)")
            state["memory_persisted"] = False

        # Decay all existing memory importance scores by 10%
        # This ensures new issues are prioritized and stale issues lose importance
        logger.info("Decaying all memory importance scores by 10%...")
        decay_count = memory_toolkit.state.decay_all_importances(decay_factor=0.9)
        logger.info(f"Decayed {decay_count} memories")

        # Update inventory based on this turn
        logger.info("\n" + "-" * 80)
        logger.info("UPDATING INVENTORY")
        logger.info("-" * 80)

        decision = state["decision"]
        if decision and decision.command:
            from tools.inventory import InventoryAnalyzer
            from config import get_cheap_llm

            # Use cheap LLM for inventory analysis
            analyzer = InventoryAnalyzer(get_cheap_llm(temperature=0))

            # Analyze turn for inventory changes
            changes = analyzer.analyze_turn(
                player_command=decision.command,
                game_response=zork_response.Response
            )

            logger.info(f"Items added: {changes.items_added}")
            logger.info(f"Items removed: {changes.items_removed}")
            logger.info(f"Reasoning: {changes.reasoning}")

            # Apply changes to inventory state
            for item in changes.items_added:
                inventory_toolkit.state.add_item(item, turn_number_ref["current"])

            for item in changes.items_removed:
                inventory_toolkit.state.remove_item(item, turn_number_ref["current"])

            current_inventory = inventory_toolkit.state.get_items()
            logger.info(f"Current inventory ({len(current_inventory)} items): {current_inventory}")
        else:
            logger.info("No command to analyze (decision was None)")

        logger.info("-" * 80)

        logger.info("=" * 80)
        logger.info("PERSIST COMPLETE")
        logger.info("=" * 80)
        return state

    return persist_node


def create_decision_graph(
    research_agent: Runnable,
    decision_chain: Runnable,
    decision_llm,
    history_toolkit: HistoryToolkit,
    memory_toolkit: MemoryToolkit,
    mapper_toolkit: MapperToolkit,
    inventory_toolkit,
    turn_number_ref: dict
):
    """
    Build the decision-making graph.

    Flow:
        SpawnAgents → Research → Decide → CloseIssues → Observe → Persist → END

    Args:
        research_agent: Research agent that calls history tools
        decision_chain: Decision chain with structured output
        decision_llm: LLM for IssueAgent, ExplorerAgent proposals, Observer, and IssueClosedAgent
        history_toolkit: History toolkit for tool execution
        memory_toolkit: Memory toolkit for persistence and issue agent spawning
        mapper_toolkit: Mapper toolkit for ExplorerAgent spawning
        turn_number_ref: Mutable reference to current turn number

    Returns:
        Compiled LangGraph
    """
    graph = StateGraph(DecisionState)

    # Add nodes
    graph.add_node("spawn_agents", create_spawn_agents_node(
        memory_toolkit,
        mapper_toolkit,
        inventory_toolkit,
        research_agent,
        decision_llm,
        history_toolkit
    ))
    graph.add_node("research", create_research_node(research_agent, history_toolkit))
    graph.add_node("decide", create_decision_node(decision_chain))
    graph.add_node("close_issues", create_close_issues_node(decision_llm, history_toolkit, memory_toolkit))
    graph.add_node("observe", create_observe_node(decision_llm, research_agent, history_toolkit, memory_toolkit))
    graph.add_node("persist", create_persist_node(memory_toolkit, inventory_toolkit, turn_number_ref))

    # Define flow
    graph.set_entry_point("spawn_agents")
    graph.add_edge("spawn_agents", "research")
    graph.add_edge("research", "decide")
    graph.add_edge("decide", "close_issues")
    graph.add_edge("close_issues", "observe")
    graph.add_edge("observe", "persist")
    graph.add_edge("persist", END)

    return graph.compile()

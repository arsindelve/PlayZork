"""
Simple LangGraph for managing the decision-making flow.

Flow: SpawnAgents → Research → Decide → Persist → END

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


class DecisionState(TypedDict):
    """State passed through the decision graph"""

    # Input
    game_response: ZorkApiResponse

    # Spawn phase output
    issue_agents: List[IssueAgent]
    explorer_agent: Optional[ExplorerAgent]  # Single agent, can be None

    # Research phase output
    research_context: str

    # Decision phase output
    decision: Optional[AdventurerResponse]

    # Persistence tracking
    memory_persisted: bool


def create_spawn_agents_node(
    memory_toolkit: MemoryToolkit,
    mapper_toolkit: MapperToolkit,
    research_agent: Runnable,
    decision_llm,
    history_toolkit: HistoryToolkit
):
    """
    Create the spawn agents node that creates IssueAgents and ExplorerAgent.

    Args:
        memory_toolkit: MemoryToolkit for accessing stored strategic issues
        mapper_toolkit: MapperToolkit for accessing map state (NEW)
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
            logger.info("========== SPAWN_AGENTS_NODE STARTING ==========")

            # Get all tracked issues from database (ordered by importance)
            memories = memory_toolkit.state.get_top_memories(limit=100)
            logger.info(f"Retrieved {len(memories)} memories from database")

            # Create one IssueAgent for each issue
            issue_agents = [IssueAgent(memory=mem) for mem in memories]

            logger.info(f"SPAWNED {len(issue_agents)} IssueAgents")

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

            # ========== PARALLEL RESEARCH: IssueAgents + ExplorerAgent ==========
            logger.info(f"Starting PARALLEL research for {len(issue_agents)} IssueAgents + {1 if explorer_agent else 0} ExplorerAgent...")

            # Get tools
            history_tools = history_toolkit.get_tools()
            mapper_tools = mapper_toolkit.get_tools()

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
                    else:  # IssueAgent
                        logger.info(f"Starting research for IssueAgent: '{agent.issue_content}'")
                        agent.research_and_propose(
                            research_agent=research_agent,
                            decision_llm=decision_llm,
                            history_tools=history_tools,
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

            # Combine all agents
            all_agents = issue_agents + ([explorer_agent] if explorer_agent else [])

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
            logger.info("========== SPAWN_AGENTS_NODE COMPLETE ==========")

            state["issue_agents"] = issue_agents
            state["explorer_agent"] = explorer_agent  # Single agent, can be None
            return state

        except Exception as e:
            logger.error(f"CRITICAL ERROR in spawn_agents_node: {e}", exc_info=True)
            # Return empty agents on error
            state["issue_agents"] = []
            state["explorer_agent"] = None
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

        logger.info("========== RESEARCH_NODE START ==========")
        logger.info(f"Current location: {zork_response.LocationName}")
        logger.info(f"Current game response (first 100): {zork_response.Response[:100]}...")

        # Call research agent
        response = research_agent.invoke(research_input)

        # Execute tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            tools_map = {tool.name: tool for tool in history_toolkit.get_tools()}

            logger.info(f"Research agent made {len(response.tool_calls)} tool calls")

            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})

                logger.info(f"  Calling tool: {tool_name} with args: {tool_args}")

                if tool_name in tools_map:
                    tool_result = tools_map[tool_name].invoke(tool_args)
                    logger.info(f"  Tool result (first 200 chars): {str(tool_result)[:200]}...")
                    tool_results.append(f"{tool_name} result: {tool_result}")

            # Combine tool results into summary
            research_context = "\n\n".join(tool_results) if tool_results else "No tools executed."
        else:
            # If no tool calls, use the content directly
            research_context = response.content if hasattr(response, 'content') else str(response)
            logger.info("No tool calls made by research agent")

        logger.info(f"Research context length: {len(research_context)} chars")
        logger.info(f"Research context (first 300): {research_context[:300]}...")
        logger.info("========== RESEARCH_NODE END ==========")

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
        Decision phase: Generate AdventurerResponse based on research.
        """
        import logging
        logger = logging.getLogger(__name__)

        zork_response = state["game_response"]
        research_context = state["research_context"]

        logger.info("========== DECISION_NODE INPUT ==========")
        logger.info(f"Location: {zork_response.LocationName}")
        logger.info(f"Score: {zork_response.Score}, Moves: {zork_response.Moves}")
        logger.info(f"Game Response (first 100): {zork_response.Response[:100]}...")
        logger.info(f"Research Context (first 500):\n{research_context[:500]}...")
        logger.info("=========================================")

        decision_input = {
            "score": zork_response.Score,
            "locationName": zork_response.LocationName,
            "moves": zork_response.Moves,
            "game_response": zork_response.Response,
            "research_context": research_context
        }

        # Invoke decision chain
        decision = decision_chain.invoke(decision_input)

        logger.info(f"DECISION MADE: {decision.command}")
        logger.info(f"REASON: {decision.reason}")

        state["decision"] = decision
        return state

    return decision_node


def create_persist_node(memory_toolkit: MemoryToolkit, turn_number_ref: dict):
    """
    Create the persistence node that stores strategic issues.

    Args:
        memory_toolkit: MemoryToolkit for storing strategic issues
        turn_number_ref: Mutable dict with current turn number

    Returns:
        Node function for the graph
    """
    def persist_node(state: DecisionState) -> DecisionState:
        """
        Persistence phase: Store strategic issues if flagged.
        """
        decision = state["decision"]
        zork_response = state["game_response"]

        if decision.remember and decision.remember.strip():
            was_added = memory_toolkit.add_memory(
                content=decision.remember,
                importance=decision.rememberImportance or 500,
                turn_number=turn_number_ref["current"],
                location=zork_response.LocationName or "Unknown",
                score=zork_response.Score,
                moves=zork_response.Moves
            )
            state["memory_persisted"] = was_added
        else:
            state["memory_persisted"] = False

        return state

    return persist_node


def create_decision_graph(
    research_agent: Runnable,
    decision_chain: Runnable,
    decision_llm,
    history_toolkit: HistoryToolkit,
    memory_toolkit: MemoryToolkit,
    mapper_toolkit: MapperToolkit,
    turn_number_ref: dict
):
    """
    Build the decision-making graph.

    Flow:
        SpawnAgents → Research → Decide → Persist → END

    Args:
        research_agent: Research agent that calls history tools
        decision_chain: Decision chain with structured output
        decision_llm: LLM for IssueAgent and ExplorerAgent proposals
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
        research_agent,
        decision_llm,
        history_toolkit
    ))
    graph.add_node("research", create_research_node(research_agent, history_toolkit))
    graph.add_node("decide", create_decision_node(decision_chain))
    graph.add_node("persist", create_persist_node(memory_toolkit, turn_number_ref))

    # Define flow
    graph.set_entry_point("spawn_agents")
    graph.add_edge("spawn_agents", "research")
    graph.add_edge("research", "decide")
    graph.add_edge("decide", "persist")
    graph.add_edge("persist", END)

    return graph.compile()

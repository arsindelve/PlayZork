"""
Simple LangGraph for managing the decision-making flow.

Flow: Research → Decide → Persist → END

This introduces graph-based control flow while keeping the existing
research and decision logic intact.
"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from zork.zork_api_response import ZorkApiResponse
from adventurer.adventurer_response import AdventurerResponse
from tools.history import HistoryToolkit
from tools.memory import MemoryToolkit
from langchain_core.runnables import Runnable


class DecisionState(TypedDict):
    """State passed through the decision graph"""

    # Input
    game_response: ZorkApiResponse

    # Research phase output
    research_context: str

    # Decision phase output
    decision: Optional[AdventurerResponse]

    # Persistence tracking
    memory_persisted: bool


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
        zork_response = state["game_response"]

        research_input = {
            "input": "Use the available tools to gather relevant history context.",
            "score": zork_response.Score,
            "locationName": zork_response.LocationName,
            "moves": zork_response.Moves,
            "game_response": zork_response.Response
        }

        # Call research agent
        response = research_agent.invoke(research_input)

        # Execute tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            tools_map = {tool.name: tool for tool in history_toolkit.get_tools()}

            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})

                if tool_name in tools_map:
                    tool_result = tools_map[tool_name].invoke(tool_args)
                    tool_results.append(f"{tool_name} result: {tool_result}")

            # Combine tool results into summary
            research_context = "\n\n".join(tool_results) if tool_results else "No tools executed."
        else:
            # If no tool calls, use the content directly
            research_context = response.content if hasattr(response, 'content') else str(response)

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
        zork_response = state["game_response"]
        research_context = state["research_context"]

        decision_input = {
            "score": zork_response.Score,
            "locationName": zork_response.LocationName,
            "moves": zork_response.Moves,
            "game_response": zork_response.Response,
            "research_context": research_context
        }

        # Invoke decision chain
        decision = decision_chain.invoke(decision_input)

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
    history_toolkit: HistoryToolkit,
    memory_toolkit: MemoryToolkit,
    turn_number_ref: dict
):
    """
    Build the decision-making graph.

    Flow:
        Research → Decide → Persist → END

    Args:
        research_agent: Research agent that calls history tools
        decision_chain: Decision chain with structured output
        history_toolkit: History toolkit for tool execution
        memory_toolkit: Memory toolkit for persistence
        turn_number_ref: Mutable reference to current turn number

    Returns:
        Compiled LangGraph
    """
    graph = StateGraph(DecisionState)

    # Add nodes
    graph.add_node("research", create_research_node(research_agent, history_toolkit))
    graph.add_node("decide", create_decision_node(decision_chain))
    graph.add_node("persist", create_persist_node(memory_toolkit, turn_number_ref))

    # Define flow
    graph.set_entry_point("research")
    graph.add_edge("research", "decide")
    graph.add_edge("decide", "persist")
    graph.add_edge("persist", END)

    return graph.compile()

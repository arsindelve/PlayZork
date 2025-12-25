from typing import Optional, List
from langchain_core.tools import tool

from .history_state import HistoryState, GameTurn


# Module-level reference to shared state
_history_state: Optional[HistoryState] = None


def initialize_history_tools(history_state: HistoryState) -> None:
    """
    Initialize the history tools with a reference to the history state.
    Must be called before tools can be used.

    Args:
        history_state: The HistoryState instance to use for tool queries
    """
    global _history_state
    _history_state = history_state


@tool
def get_recent_turns(n: int = 5) -> str:
    """Get the last N turns of game history to understand recent actions and context.

    Use this tool when you need to:
    - Check what actions were recently attempted
    - See recent location changes
    - Review recent puzzle attempts
    - Avoid repeating failed commands
    - Understand immediate context

    Args:
        n: Number of recent turns to retrieve (1-20). Use 3-5 for immediate context,
           10+ when investigating patterns or when stuck in loops. Default is 5.

    Returns:
        Formatted string with recent turns showing player commands and game responses
    """
    if _history_state is None:
        return "Error: History tools not initialized."

    # Limit to reasonable range
    n = max(1, min(n, 20))

    turns = _history_state.get_recent_turns(n)

    if not turns:
        return "No turns recorded yet."

    # Format turns as readable text
    result = f"Last {len(turns)} turn(s):\n\n"
    for turn in turns:
        result += f"Turn #{turn.turn_number}"
        if turn.location:
            result += f" (at {turn.location})"
        if turn.score > 0:
            result += f" [Score: {turn.score}]"
        result += f"\n"
        result += f"  Player: {turn.player_command}\n"
        result += f"  Game: {turn.game_response}\n\n"

    return result.strip()


@tool
def get_full_summary() -> str:
    """Get the complete narrative summary of all game history from the beginning.

    Use this tool when you need to:
    - Understand the overall progress in the game
    - Review what has been accomplished so far
    - Get context about puzzles, items, and locations discovered
    - Plan long-term strategy
    - Remember important details from earlier in the game

    Returns:
        Narrative summary of everything that has happened in the game
    """
    if _history_state is None:
        return "Error: History tools not initialized."

    summary = _history_state.get_full_summary()

    # Add metadata
    turn_count = _history_state.get_turn_count()
    if turn_count > 0:
        return f"Summary of {turn_count} turn(s):\n\n{summary}"
    else:
        return summary


def get_history_tools() -> List:
    """
    Get the list of all history tools for use with LangChain agents

    Returns:
        List of tool functions decorated with @tool
    """
    return [get_recent_turns, get_full_summary]

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
    """Get the RAW last N turns showing EXACT commands and responses - see precisely what failed or succeeded.

    This shows the actual command/response pairs, NOT a summary. Critical for:
    - Detecting if you just repeated the same failed command
    - Seeing the EXACT error message (not a paraphrase)
    - Identifying recent location changes turn-by-turn
    - Spotting immediate loops (tried X, failed, tried X again)

    WARNING: Only shows recent turns, NOT the full game context or big picture.

    Args:
        n: Number of recent turns (1-20). Use 3-5 for immediate context,
           10+ when investigating loops or patterns. Default is 5.

    Returns:
        Raw command/response history with locations and scores
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
    """Get narrative summary of ALL game progress from turn 1 to now - the BIG PICTURE across your entire journey.

    This is NOT raw turns - it's an AI-generated overview showing patterns and progress across
    ALL history. Essential for:
    - Understanding your overall strategy and what you've accomplished
    - Seeing patterns across dozens of turns that recent history misses
    - Remembering puzzles/items/locations discovered many turns ago
    - Avoiding repeating strategies that failed earlier (but aren't in recent turns)
    - Getting the full context before making important decisions

    Without this, you're blind to everything beyond the last few turns.

    Returns:
        Comprehensive narrative of your entire game from the beginning
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

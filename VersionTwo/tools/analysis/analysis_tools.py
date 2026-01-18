"""Tools for accessing strategic analysis in LangChain agents"""
from typing import Optional
from langchain_core.tools import tool

from tools.database import DatabaseManager

# Module-level state for tools
_db: Optional[DatabaseManager] = None
_session_id: Optional[str] = None


def initialize_analysis_tools(session_id: str, db: DatabaseManager) -> None:
    """
    Initialize the analysis tools with database access.

    Args:
        session_id: Game session ID
        db: DatabaseManager instance
    """
    global _db, _session_id
    _db = db
    _session_id = session_id


@tool
def get_strategic_analysis() -> str:
    """
    Get the latest big-picture strategic analysis of the game.

    This provides high-level strategic insight about:
    - Where we are in the overall story arc
    - Our theory of what the game requires
    - Major milestones achieved and remaining
    - Strategic resources and blind spots
    - Emergent patterns in the game design

    Use this to understand the broader context before making decisions.

    Returns:
        Strategic analysis from the previous turn
    """
    if _db is None or _session_id is None:
        return "Strategic analysis not initialized."

    result = _db.get_latest_strategic_analysis(_session_id)
    if result:
        turn_number, analysis = result
        return f"[Strategic Analysis from Turn {turn_number}]\n\n{analysis}"
    return "No strategic analysis available yet. This is generated after each turn."


def get_analysis_tools() -> list:
    """
    Get the list of analysis tools for use with LangChain agents.

    Returns:
        List of tool functions
    """
    return [get_strategic_analysis]

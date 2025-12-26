from langchain_core.tools import tool
from typing import Optional
from .memory_state import MemoryState
from .memory_retriever import MemoryRetriever

# Module-level state (initialized by toolkit)
_memory_state: Optional[MemoryState] = None
_memory_retriever: Optional[MemoryRetriever] = None


def initialize_memory_tools(memory_state: MemoryState, retriever: MemoryRetriever):
    """
    Initialize the module-level memory state and retriever.
    Called by MemoryToolkit on creation.
    """
    global _memory_state, _memory_retriever
    _memory_state = memory_state
    _memory_retriever = retriever


@tool
def get_top_memories(limit: int = 10) -> str:
    """
    Get strategic issues YOU FLAGGED: unsolved puzzles, major obstacles, obvious things to try.

    These are ACTIONABLE CHALLENGES you recorded, sorted by how much solving them helps win
    the game (importance 1-1000). This is NOT a list of observations or items - it's a list
    of PROBLEMS TO SOLVE and OBSTACLES TO OVERCOME.

    Each memory is a strategic issue like:
    - "locked grating blocks path east" (puzzle to solve)
    - "need to get inside white house" (obvious thing to try)
    - "troll demands payment to pass bridge" (major obstacle)

    Higher importance = more critical to winning. Focus on high-importance issues first.

    Args:
        limit: Number of top strategic issues (default: 10, max: 20)

    Returns:
        Strategic issues sorted by importance with scores
    """
    if _memory_state is None:
        return "Error: Memory system not initialized."

    limit = max(1, min(limit, 20))  # Clamp to 1-20

    top_memories = _memory_state.get_top_memories(limit)

    if not top_memories:
        return "No strategic issues recorded yet. As you discover unsolved puzzles, obstacles, or obvious things to try, they'll be tracked here."

    return _memory_retriever.get_top_insights(top_memories, limit)


@tool
def query_memories(question: str) -> str:
    """
    Search strategic issues (puzzles, obstacles, things to try) with AI-powered semantic search.

    This uses an LLM to intelligently find relevant UNSOLVED PROBLEMS and synthesize an answer.
    Ask about specific puzzles, obstacles, or challenges you've encountered.

    Example queries:
        "What puzzles are still unsolved?" - Track what needs solving
        "What obstacles are blocking my progress?" - Find what to overcome
        "What obvious things should I try?" - Get leads
        "Are there any locked doors?" - Find blocking obstacles

    This searches your STRATEGIC ISSUES, not general observations.

    Args:
        question: Natural language question

    Returns:
        AI-generated answer based on relevant strategic issues
    """
    if _memory_state is None:
        return "Error: Memory system not initialized."

    if not question or not question.strip():
        return "Error: Please provide a specific question."

    # Get top memories (limit 100 for search corpus)
    all_memories = _memory_state.get_top_memories(limit=100)

    if not all_memories:
        return "No strategic issues to search. No puzzles, obstacles, or challenges have been flagged yet."

    return _memory_retriever.query_memories(question, all_memories, max_results=5)


@tool
def get_location_memories(location: str) -> str:
    """
    Get strategic issues (puzzles, obstacles) you flagged at a specific location.

    When revisiting a place, this shows PROBLEMS YOU NEED TO SOLVE there:
    - Unsolved puzzles at this location
    - Obstacles blocking progress here
    - Things you should try at this place

    Critical when returning to a location with new items or knowledge - you can see
    if you now have what's needed to solve puzzles you couldn't solve before.

    Does NOT show general observations or items - only ACTIONABLE CHALLENGES.

    Args:
        location: Exact location name (e.g., "West Of House", "Living Room")

    Returns:
        Strategic issues at that location with importance scores
    """
    if _memory_state is None:
        return "Error: Memory system not initialized."

    if not location or not location.strip():
        return "Error: Please provide a location name."

    location_memories = _memory_state.get_memories_by_location(location)

    if not location_memories:
        return f"No strategic issues recorded for '{location}'. No unsolved puzzles or obstacles were flagged at this location."

    return _memory_retriever.summarize_location_memories(location, location_memories)


def get_memory_tools() -> list:
    """
    Get all memory tools for use by the agent.

    Returns:
        List of LangChain tools
    """
    return [
        get_top_memories,
        query_memories,
        get_location_memories
    ]

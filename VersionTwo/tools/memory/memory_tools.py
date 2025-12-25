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
    Get the discoveries YOU EXPLICITLY FLAGGED as worth remembering - your own persistent insights.

    These are NOT automatic - these are things you consciously chose to remember via the
    'remember' field because they were critical: puzzle solutions, important items, obstacles,
    key locations, breakthrough discoveries. Sorted by importance (you rated 1-1000).

    If you flagged something as worth remembering, it was IMPORTANT. Ignoring this means
    forgetting your own hard-won insights from earlier exploration.

    Args:
        limit: Number of top memories (default: 10, max: 20)

    Returns:
        Your most important flagged discoveries with importance scores
    """
    if _memory_state is None:
        return "Error: Memory system not initialized."

    limit = max(1, min(limit, 20))  # Clamp to 1-20

    top_memories = _memory_state.get_top_memories(limit)

    if not top_memories:
        return "No memories recorded yet. As you play, important discoveries will be remembered here."

    return _memory_retriever.get_top_insights(top_memories, limit)


@tool
def query_memories(question: str) -> str:
    """
    Search your flagged memories with AI-powered semantic search to answer a SPECIFIC question.

    This uses an LLM to intelligently find relevant memories and synthesize an answer - NOT
    just keyword matching. Ask about items, locations, puzzles, obstacles, anything you
    previously flagged as important.

    Examples of powerful queries:
        "Where did I see a rusty key?" - Find item locations
        "What puzzles are still unsolved?" - Track progress
        "Which doors are locked?" - Remember obstacles
        "What do I know about the attic?" - Recall location details

    This is your ONLY way to search memories by content, not just look at the top 10.

    Args:
        question: Natural language question

    Returns:
        AI-generated answer based on relevant memories found
    """
    if _memory_state is None:
        return "Error: Memory system not initialized."

    if not question or not question.strip():
        return "Error: Please provide a specific question."

    all_memories = _memory_state.get_all_memories()

    if not all_memories:
        return "No memories to search. Nothing important has been flagged yet."

    return _memory_retriever.query_memories(question, all_memories, max_results=5)


@tool
def get_location_memories(location: str) -> str:
    """
    Recall EVERYTHING you flagged as important at a specific location - essential when revisiting places.

    When you return to a location, this shows all discoveries you made there: items found,
    puzzles encountered, observations noted. Critical for:
    - Remembering what items were available here
    - Recalling puzzle clues specific to this room
    - Avoiding re-exploring what you already checked
    - Seeing if you have new tools/knowledge to solve old obstacles

    Without this, returning to a location means starting from scratch instead of building
    on what you learned before.

    Args:
        location: Exact location name (e.g., "West Of House", "Living Room")

    Returns:
        All memories from that location with context
    """
    if _memory_state is None:
        return "Error: Memory system not initialized."

    if not location or not location.strip():
        return "Error: Please provide a location name."

    location_memories = _memory_state.get_memories_by_location(location)

    if not location_memories:
        return f"No memories recorded for '{location}'. Either you haven't been there or nothing important was noted."

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

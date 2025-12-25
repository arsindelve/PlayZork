from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from typing import List
from .memory_state import MemoryState, Memory
from .memory_retriever import MemoryRetriever
from .memory_tools import initialize_memory_tools, get_memory_tools


class MemoryToolkit:
    """
    Facade for the memory system.
    Manages memory state and provides tools for the agent.
    """

    def __init__(self, retriever_llm: ChatOpenAI):
        """
        Initialize the memory toolkit.

        Args:
            retriever_llm: A cheap LLM for semantic search (e.g., GPT-3.5-turbo)
        """
        self.state = MemoryState()
        self.retriever = MemoryRetriever(retriever_llm)

        # Initialize module-level tools
        initialize_memory_tools(self.state, self.retriever)

    def add_memory(
        self,
        content: str,
        importance: int,
        turn_number: int,
        location: str,
        score: int,
        moves: int
    ) -> bool:
        """
        Add a new memory (called by game loop after each turn).

        Args:
            content: What to remember (from AdventurerResponse.remember)
            importance: Priority 1-1000 (from AdventurerResponse.rememberImportance)
            turn_number: Current turn number
            location: Current location
            score: Current game score
            moves: Current move count

        Returns:
            True if memory was added, False if skipped (empty or duplicate)
        """
        memory = self.state.add_memory(
            content=content,
            importance=importance,
            turn_number=turn_number,
            location=location,
            score=score,
            moves=moves
        )

        return memory is not None

    def get_tools(self) -> List[Tool]:
        """
        Get all memory tools for the agent to use.

        Returns:
            List of LangChain tools: get_top_memories, query_memories, get_location_memories
        """
        return get_memory_tools()

    def get_memory_count(self) -> int:
        """Get total number of stored memories"""
        return self.state.get_memory_count()

    def get_summary_stats(self) -> dict:
        """Get summary statistics about memories"""
        memories = self.state.get_all_memories()

        if not memories:
            return {
                "total_memories": 0,
                "avg_importance": 0,
                "locations_covered": 0
            }

        return {
            "total_memories": len(memories),
            "avg_importance": sum(m.importance for m in memories) / len(memories),
            "locations_covered": len(set(m.location for m in memories)),
            "top_location": max(set(m.location for m in memories), key=lambda loc: sum(1 for m in memories if m.location == loc))
        }


# Export main classes
__all__ = ['MemoryToolkit', 'Memory', 'MemoryState']

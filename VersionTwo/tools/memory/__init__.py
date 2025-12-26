from .memory_state import MemoryState, Memory
from .memory_deduplicator import MemoryDeduplicator
from tools.database import DatabaseManager
from langchain_core.language_models import BaseChatModel


class MemoryToolkit:
    """
    Manages strategic issue storage (puzzles, obstacles, things to try).

    Memory is WRITE-ONLY: the agent flags strategic issues via the 'remember' field,
    but cannot query them as tools. All strategic issues persist to SQLite.

    Includes LLM-based de-duplication to prevent storing semantically similar issues.
    """

    def __init__(self, session_id: str, db: DatabaseManager, dedup_llm: BaseChatModel):
        """
        Initialize the memory toolkit with database backend and de-duplication.

        Args:
            session_id: Unique identifier for this game session
            db: DatabaseManager instance for persistence
            dedup_llm: LLM instance for semantic de-duplication (should be cheap model)
        """
        self.deduplicator = MemoryDeduplicator(llm=dedup_llm)
        self.state = MemoryState(session_id=session_id, db=db, deduplicator=self.deduplicator)

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

    def get_memory_count(self) -> int:
        """Get total number of stored memories"""
        return self.state.get_memory_count()

    def get_summary_stats(self) -> dict:
        """Get summary statistics about memories"""
        # Get top memories (up to 100) for stats
        memories = self.state.get_top_memories(limit=100)

        if not memories:
            return {
                "total_memories": 0,
                "avg_importance": 0,
                "locations_covered": 0
            }

        return {
            "total_memories": self.state.get_memory_count(),
            "avg_importance": sum(m.importance for m in memories) / len(memories),
            "locations_covered": len(set(m.location for m in memories)),
            "top_location": max(set(m.location for m in memories), key=lambda loc: sum(1 for m in memories if m.location == loc)) if memories else ""
        }


# Export main classes
__all__ = ['MemoryToolkit', 'Memory', 'MemoryState']

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class Memory(BaseModel):
    """A single memory flagged by the LLM as important"""

    turn_number: int           # When this was remembered
    content: str               # What to remember (from AdventurerResponse.remember)
    importance: int            # 1-1000 priority (from AdventurerResponse.rememberImportance)
    location: str              # Where we were when we learned this
    score: int                 # Game score at the time
    moves: int                 # Total moves at the time
    timestamp: str             # ISO timestamp

    def __str__(self) -> str:
        return f"[Turn {self.turn_number} @ {self.location}] {self.content} (importance: {self.importance})"


class MemoryState:
    """In-memory storage for all memories"""

    def __init__(self):
        self.memories: List[Memory] = []

    def add_memory(
        self,
        content: str,
        importance: int,
        turn_number: int,
        location: str,
        score: int,
        moves: int
    ) -> Optional[Memory]:
        """
        Add a new memory.

        Args:
            content: What to remember
            importance: Priority 1-1000
            turn_number: Current turn
            location: Current location name
            score: Current score
            moves: Current move count

        Returns:
            The created Memory, or None if skipped (empty or duplicate)
        """
        # Don't add empty or duplicate memories
        if not content or not content.strip():
            return None

        # Check for near-duplicates (fuzzy match on content)
        for existing in self.memories:
            if self._is_similar(content, existing.content):
                # Update importance if new one is higher
                if importance > existing.importance:
                    existing.importance = importance
                return None  # Don't add duplicate

        memory = Memory(
            turn_number=turn_number,
            content=content.strip(),
            importance=importance,
            location=location,
            score=score,
            moves=moves,
            timestamp=datetime.now().isoformat()
        )

        self.memories.append(memory)

        # Keep sorted by importance (highest first)
        self.memories.sort(key=lambda m: m.importance, reverse=True)

        return memory

    def _is_similar(self, text1: str, text2: str) -> bool:
        """Simple similarity check to avoid duplicates"""
        # Normalize
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # Exact match
        if t1 == t2:
            return True

        # One contains the other (80% threshold)
        if len(t1) > len(t2):
            return t2 in t1 and len(t2) / len(t1) > 0.8
        else:
            return t1 in t2 and len(t1) / len(t2) > 0.8

    def get_top_memories(self, limit: int = 10) -> List[Memory]:
        """Get the N most important memories"""
        return self.memories[:limit]

    def get_memories_by_location(self, location: str) -> List[Memory]:
        """Get all memories associated with a specific location"""
        return [m for m in self.memories if m.location.lower() == location.lower()]

    def get_all_memories(self) -> List[Memory]:
        """Get all memories (sorted by importance)"""
        return self.memories

    def get_memory_count(self) -> int:
        """Total number of memories stored"""
        return len(self.memories)

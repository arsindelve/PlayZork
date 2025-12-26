from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from tools.database import DatabaseManager
from .memory_deduplicator import MemoryDeduplicator
import logging


class Memory(BaseModel):
    """A single memory flagged by the LLM as important"""

    id: Optional[int] = None   # Database ID (None for new memories)
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
    """SQLite-backed storage for all memories"""

    def __init__(self, session_id: str, db: DatabaseManager, deduplicator: Optional[MemoryDeduplicator] = None):
        """
        Initialize memory state with database backend.

        Args:
            session_id: Unique identifier for this game session
            db: DatabaseManager instance for persistence
            deduplicator: Optional MemoryDeduplicator for semantic de-duplication
        """
        self.session_id = session_id
        self.db = db
        self.deduplicator = deduplicator

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
        Add a new memory (persisted to database).

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
        # Don't add empty memories
        if not content or not content.strip():
            return None

        # Ensure importance is an integer (LLM might return string)
        try:
            importance = int(importance)
        except (ValueError, TypeError):
            importance = 500  # Default to medium importance if invalid

        # Clamp to valid range
        importance = max(1, min(1000, importance))

        # Check for exact duplicates in database
        if self.db.check_duplicate_memory(self.session_id, content.strip()):
            return None  # Don't add duplicate

        # Check for semantic duplicates using LLM (if deduplicator available)
        if self.deduplicator:
            existing_memories = self.get_top_memories(limit=50)  # Check against recent memories
            if existing_memories:
                existing_contents = [mem.content for mem in existing_memories]
                is_dup, reason = self.deduplicator.is_duplicate(content.strip(), existing_contents)
                if is_dup:
                    # Log why it was skipped (but don't add to database)
                    logger = logging.getLogger(__name__)
                    logger.info(f"DUPLICATE MEMORY SKIPPED: '{content.strip()}' - Reason: {reason}")
                    return None

        memory = Memory(
            turn_number=turn_number,
            content=content.strip(),
            importance=importance,
            location=location,
            score=score,
            moves=moves,
            timestamp=datetime.now().isoformat()
        )

        # Persist to database
        self.db.add_memory(
            session_id=self.session_id,
            turn_number=turn_number,
            content=content.strip(),
            importance=importance,
            location=location,
            score=score,
            moves=moves
        )

        return memory

    def get_top_memories(self, limit: int = 10) -> List[Memory]:
        """Get the N most important memories (from database)"""
        db_memories = self.db.get_top_memories(self.session_id, limit)
        # Convert database tuples to Memory objects
        # db_memories format: (id, content, importance, turn_number, location)
        memories = []
        for mem_data in db_memories:
            memories.append(Memory(
                id=mem_data[0],
                content=mem_data[1],
                importance=mem_data[2],
                turn_number=mem_data[3],
                location=mem_data[4] or "",
                score=0,  # Not stored separately in query
                moves=0,  # Not stored separately in query
                timestamp=datetime.now().isoformat()
            ))
        return memories

    def get_memories_by_location(self, location: str, limit: int = 5) -> List[Memory]:
        """Get memories associated with a specific location (from database)"""
        db_memories = self.db.get_location_memories(self.session_id, location, limit)
        # db_memories format: (content, importance, turn_number)
        memories = []
        for mem_data in db_memories:
            memories.append(Memory(
                content=mem_data[0],
                importance=mem_data[1],
                turn_number=mem_data[2],
                location=location,
                score=0,
                moves=0,
                timestamp=datetime.now().isoformat()
            ))
        return memories

    def get_memory_count(self) -> int:
        """Total number of memories stored (from database)"""
        # Quick count via get_top_memories (not efficient but works)
        all_mems = self.db.get_top_memories(self.session_id, limit=1000)
        return len(all_mems)

    def remove_memory(self, memory_id: int) -> bool:
        """
        Remove a memory by ID.

        Args:
            memory_id: Database ID of the memory to remove

        Returns:
            True if memory was removed, False otherwise
        """
        return self.db.remove_memory(self.session_id, memory_id)

    def decay_all_importances(self, decay_factor: float = 0.9) -> int:
        """
        Decay all memory importance scores by a factor (default 10% reduction).

        This ensures new issues are naturally higher priority and stale issues
        gradually lose importance, preventing loops on unproductive issues.

        Args:
            decay_factor: Multiply importance by this (0.9 = 10% reduction)

        Returns:
            Number of memories updated
        """
        logger = logging.getLogger(__name__)
        count = self.db.decay_all_importances(self.session_id, decay_factor)
        logger.info(f"[MemoryState] Decayed {count} memories by {int((1-decay_factor)*100)}%")
        return count

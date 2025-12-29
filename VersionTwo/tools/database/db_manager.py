"""
Database manager for persistent storage of game sessions, history, and memory.

Provides SQLite backend for:
- Game sessions
- Turn-by-turn history
- History summaries (recent + long-running)
- Memory entries with importance scoring
"""
import sqlite3
from pathlib import Path
from typing import Optional, List, Tuple
from contextlib import contextmanager


class DatabaseManager:
    """Manages SQLite database for game state persistence"""

    def __init__(self, db_path: str = "data/zork_sessions.db"):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_exists()
        self._create_schema()

    def _ensure_db_exists(self):
        """Create database directory if it doesn't exist"""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dicts
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_schema(self):
        """Create database schema if it doesn't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    final_score INTEGER DEFAULT 0,
                    total_moves INTEGER DEFAULT 0
                )
            """)

            # Turns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    player_command TEXT NOT NULL,
                    game_response TEXT NOT NULL,
                    location TEXT,
                    score INTEGER DEFAULT 0,
                    moves INTEGER DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    UNIQUE(session_id, turn_number)
                )
            """)

            # Summaries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    recent_summary TEXT NOT NULL,
                    long_running_summary TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    UNIQUE(session_id, turn_number)
                )
            """)

            # Memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    importance INTEGER NOT NULL,
                    location TEXT,
                    score INTEGER DEFAULT 0,
                    moves INTEGER DEFAULT 0,
                    closed INTEGER DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Add closed column to existing tables (migration)
            try:
                cursor.execute("ALTER TABLE memories ADD COLUMN closed INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                # Column already exists
                pass

            # Map transitions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS map_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    from_location TEXT NOT NULL,
                    to_location TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    turn_discovered INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    UNIQUE(session_id, from_location, direction)
                )
            """)

            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_session
                ON turns(session_id, turn_number)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_summaries_session
                ON summaries(session_id, turn_number DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_session_importance
                ON memories(session_id, importance DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_closed
                ON memories(session_id, closed)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_map_transitions_session
                ON map_transitions(session_id, from_location)
            """)

    # ===== Session Management =====

    def create_session(self, session_id: str) -> None:
        """Create a new game session OR resume existing session if it already exists"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if session already exists
            cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
            existing = cursor.fetchone()

            if existing:
                # Session exists - RESUME it (don't delete data)
                # Just ensure status is active
                cursor.execute(
                    "UPDATE sessions SET status = ? WHERE session_id = ?",
                    ("active", session_id)
                )
            else:
                # New session - create it
                cursor.execute(
                    "INSERT INTO sessions (session_id, status) VALUES (?, ?)",
                    (session_id, "active")
                )

    def reset_session(self, session_id: str) -> None:
        """Delete all data for a session and create a fresh one"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Delete all old data for this session_id
            cursor.execute("DELETE FROM map_transitions WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM memories WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

            # Now create fresh session
            cursor.execute(
                "INSERT INTO sessions (session_id, status) VALUES (?, ?)",
                (session_id, "active")
            )

    def update_session_status(
        self,
        session_id: str,
        status: str,
        final_score: Optional[int] = None,
        total_moves: Optional[int] = None
    ) -> None:
        """Update session status and final stats"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if final_score is not None and total_moves is not None:
                cursor.execute(
                    """UPDATE sessions
                       SET status = ?, final_score = ?, total_moves = ?
                       WHERE session_id = ?""",
                    (status, final_score, total_moves, session_id)
                )
            else:
                cursor.execute(
                    "UPDATE sessions SET status = ? WHERE session_id = ?",
                    (status, session_id)
                )

    # ===== Turn History =====

    def get_latest_turn_number(self, session_id: str) -> Optional[int]:
        """Get the highest turn number for this session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(turn_number) FROM turns WHERE session_id = ?",
                (session_id,)
            )
            result = cursor.fetchone()
            return result[0] if result and result[0] is not None else None

    def add_turn(
        self,
        session_id: str,
        turn_number: int,
        player_command: str,
        game_response: str,
        location: str,
        score: int,
        moves: int
    ) -> None:
        """Add a turn to history"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO turns
                   (session_id, turn_number, player_command, game_response,
                    location, score, moves)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, turn_number, player_command, game_response,
                 location, score, moves)
            )

    def get_recent_turns(
        self,
        session_id: str,
        n: int = 5
    ) -> List[Tuple[int, str, str, str, int, int]]:
        """
        Get the N most recent turns.

        Returns:
            List of (turn_number, player_command, game_response, location, score, moves)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT turn_number, player_command, game_response,
                          location, score, moves
                   FROM turns
                   WHERE session_id = ?
                   ORDER BY turn_number DESC
                   LIMIT ?""",
                (session_id, n)
            )
            return [(row[0], row[1], row[2], row[3], row[4], row[5])
                    for row in cursor.fetchall()]

    def get_all_turns(self, session_id: str) -> List[dict]:
        """Get all turns for a session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT * FROM turns
                   WHERE session_id = ?
                   ORDER BY turn_number ASC""",
                (session_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # ===== Summaries =====

    def save_summary(
        self,
        session_id: str,
        turn_number: int,
        recent_summary: str,
        long_running_summary: Optional[str] = None
    ) -> None:
        """Save history summary after a turn"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO summaries
                   (session_id, turn_number, recent_summary, long_running_summary)
                   VALUES (?, ?, ?, ?)""",
                (session_id, turn_number, recent_summary, long_running_summary)
            )

    def get_latest_summary(self, session_id: str) -> Optional[Tuple[str, str]]:
        """
        Get the most recent summary.

        Returns:
            (recent_summary, long_running_summary) or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT recent_summary, long_running_summary
                   FROM summaries
                   WHERE session_id = ?
                   ORDER BY turn_number DESC
                   LIMIT 1""",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                return (row[0], row[1] if row[1] else "")
            return None

    # ===== Memories =====

    def add_memory(
        self,
        session_id: str,
        turn_number: int,
        content: str,
        importance: int,
        location: str,
        score: int,
        moves: int
    ) -> None:
        """Add a memory entry"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO memories
                   (session_id, turn_number, content, importance,
                    location, score, moves)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, turn_number, content, importance,
                 location, score, moves)
            )

    def get_top_memories(
        self,
        session_id: str,
        limit: int = 10,
        include_closed: bool = False
    ) -> List[Tuple[int, str, int, int, str]]:
        """
        Get top N memories by importance.

        Args:
            session_id: Game session ID
            limit: Maximum number of memories to return
            include_closed: If True, include closed issues (used for duplicate detection)

        Returns:
            List of (id, content, importance, turn_number, location)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if include_closed:
                # Include all memories (open AND closed) - for duplicate checking
                cursor.execute(
                    """SELECT id, content, importance, turn_number, location
                       FROM memories
                       WHERE session_id = ?
                       ORDER BY importance DESC, turn_number DESC
                       LIMIT ?""",
                    (session_id, limit)
                )
            else:
                # Only open memories - for agent spawning
                cursor.execute(
                    """SELECT id, content, importance, turn_number, location
                       FROM memories
                       WHERE session_id = ? AND (closed = 0 OR closed IS NULL)
                       ORDER BY importance DESC, turn_number DESC
                       LIMIT ?""",
                    (session_id, limit)
                )
            return [(row[0], row[1], row[2], row[3], row[4])
                    for row in cursor.fetchall()]

    def get_location_memories(
        self,
        session_id: str,
        location: str,
        limit: int = 5
    ) -> List[Tuple[str, int, int]]:
        """
        Get memories for a specific location.

        Returns:
            List of (content, importance, turn_number)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT content, importance, turn_number
                   FROM memories
                   WHERE session_id = ? AND location = ? AND (closed = 0 OR closed IS NULL)
                   ORDER BY importance DESC, turn_number DESC
                   LIMIT ?""",
                (session_id, location, limit)
            )
            return [(row[0], row[1], row[2])
                    for row in cursor.fetchall()]

    def check_duplicate_memory(
        self,
        session_id: str,
        content: str
    ) -> bool:
        """
        Check if a memory with similar content already exists.

        IMPORTANT: Checks ALL memories including closed ones to prevent
        creating duplicates of already-solved issues.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT COUNT(*) FROM memories
                   WHERE session_id = ? AND content = ?""",
                (session_id, content)
            )
            count = cursor.fetchone()[0]
            return count > 0

    def close_memory(
        self,
        session_id: str,
        memory_id: int
    ) -> bool:
        """
        Close a memory by setting closed=1 (soft delete).

        Closed memories are:
        - NOT returned when fetching issues for agent spawning
        - Still checked for duplicates (prevent re-creating solved issues)

        Returns:
            True if memory was closed, False otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE memories
                   SET closed = 1
                   WHERE session_id = ? AND id = ? AND (closed = 0 OR closed IS NULL)""",
                (session_id, memory_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def remove_memory(
        self,
        session_id: str,
        memory_id: int
    ) -> bool:
        """
        DEPRECATED: Use close_memory() instead.

        Close a memory by setting closed=1 (soft delete).
        Kept for backwards compatibility.

        Returns:
            True if memory was closed, False otherwise
        """
        return self.close_memory(session_id, memory_id)

    def decay_all_importances(
        self,
        session_id: str,
        decay_factor: float = 0.9
    ) -> int:
        """
        Decay all OPEN memory importance scores by a factor (default 10% reduction).

        Only decays open memories (closed=0). Closed memories keep their importance
        for historical purposes.

        Args:
            session_id: Session ID
            decay_factor: Multiply importance by this (0.9 = 10% reduction)

        Returns:
            Number of memories updated
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE memories
                   SET importance = CAST(importance * ? AS INTEGER)
                   WHERE session_id = ? AND importance > 0 AND (closed = 0 OR closed IS NULL)""",
                (decay_factor, session_id)
            )
            conn.commit()
            return cursor.rowcount

    # ===== Map Management =====

    def add_map_transition(
        self,
        session_id: str,
        from_location: str,
        to_location: str,
        direction: str,
        turn_number: int
    ) -> bool:
        """
        Add a map transition (movement from one location to another).
        Returns True if this is a new transition, False if already known.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """INSERT INTO map_transitions
                       (session_id, from_location, to_location, direction, turn_discovered)
                       VALUES (?, ?, ?, ?, ?)""",
                    (session_id, from_location, to_location, direction, turn_number)
                )
                return True  # New transition discovered
            except Exception:
                # Transition already exists (UNIQUE constraint)
                return False

    def get_all_transitions(
        self,
        session_id: str
    ) -> List[Tuple[str, str, str, int]]:
        """
        Get all known map transitions.

        Returns:
            List of (from_location, to_location, direction, turn_discovered)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT from_location, to_location, direction, turn_discovered
                   FROM map_transitions
                   WHERE session_id = ?
                   ORDER BY turn_discovered ASC""",
                (session_id,)
            )
            return [(row[0], row[1], row[2], row[3])
                    for row in cursor.fetchall()]

    def get_transitions_from_location(
        self,
        session_id: str,
        location: str
    ) -> List[Tuple[str, str]]:
        """
        Get all known exits from a specific location.

        Returns:
            List of (direction, destination)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT direction, to_location
                   FROM map_transitions
                   WHERE session_id = ? AND from_location = ?
                   ORDER BY direction ASC""",
                (session_id, location)
            )
            return [(row[0], row[1]) for row in cursor.fetchall()]

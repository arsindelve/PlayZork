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
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

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
                CREATE INDEX IF NOT EXISTS idx_map_transitions_session
                ON map_transitions(session_id, from_location)
            """)

    # ===== Session Management =====

    def create_session(self, session_id: str) -> None:
        """Create a new game session (or skip if already exists)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO sessions (session_id, status) VALUES (?, ?)",
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
        limit: int = 10
    ) -> List[Tuple[str, int, int, str]]:
        """
        Get top N memories by importance.

        Returns:
            List of (content, importance, turn_number, location)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT content, importance, turn_number, location
                   FROM memories
                   WHERE session_id = ?
                   ORDER BY importance DESC, turn_number DESC
                   LIMIT ?""",
                (session_id, limit)
            )
            return [(row[0], row[1], row[2], row[3])
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
                   WHERE session_id = ? AND location = ?
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
        """Check if a memory with similar content already exists"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT COUNT(*) FROM memories
                   WHERE session_id = ? AND content = ?""",
                (session_id, content)
            )
            count = cursor.fetchone()[0]
            return count > 0

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

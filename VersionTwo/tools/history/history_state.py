from typing import List, Optional
from pydantic import BaseModel
from tools.database import DatabaseManager


class GameTurn(BaseModel):
    """Represents a single turn in the game history"""
    turn_number: int
    player_command: str
    game_response: str
    location: Optional[str] = None
    score: int = 0
    moves: int = 0


class HistoryState:
    """
    Pure state management for game history.
    Maintains raw turns and summary with SQLite persistence.
    """

    def __init__(self, session_id: str, db: DatabaseManager):
        """
        Initialize history state with database backend.

        Args:
            session_id: Unique identifier for this game session
            db: DatabaseManager instance for persistence
        """
        self.session_id = session_id
        self.db = db
        self.previous_command: str = "LOOK"
        self._turn_counter: int = 0

    def add_turn(self, game_response: str, player_command: str,
                 location: Optional[str] = None, score: int = 0,
                 moves: int = 0) -> GameTurn:
        """
        Add a new turn to raw history (persisted to database)

        Args:
            game_response: Text response from the game
            player_command: Command issued by the player
            location: Current location name
            score: Current score
            moves: Current move count

        Returns:
            The created GameTurn object
        """
        self._turn_counter += 1
        turn = GameTurn(
            turn_number=self._turn_counter,
            player_command=player_command,
            game_response=game_response,
            location=location,
            score=score,
            moves=moves
        )

        # Persist to database
        self.db.add_turn(
            session_id=self.session_id,
            turn_number=self._turn_counter,
            player_command=player_command,
            game_response=game_response,
            location=location or "",
            score=score,
            moves=moves
        )

        self.previous_command = player_command
        return turn

    def get_recent_turns(self, n: int) -> List[GameTurn]:
        """
        Get the last N turns from history (from database)

        Args:
            n: Number of recent turns to retrieve

        Returns:
            List of most recent GameTurn objects
        """
        db_turns = self.db.get_recent_turns(self.session_id, n)
        # Convert database tuples to GameTurn objects
        # db_turns format: (turn_number, player_command, game_response, location, score, moves)
        turns = []
        for turn_data in reversed(db_turns):  # Reverse because DB returns DESC order
            turns.append(GameTurn(
                turn_number=turn_data[0],
                player_command=turn_data[1],
                game_response=turn_data[2],
                location=turn_data[3] or None,
                score=turn_data[4],
                moves=turn_data[5]
            ))
        return turns

    def get_full_summary(self) -> str:
        """
        Get the recent summary (last 15 turns) from database

        Returns:
            Recent summary string
        """
        summary_data = self.db.get_latest_summary(self.session_id)
        if summary_data:
            return summary_data[0]  # recent_summary is first element
        return "No history available yet."

    def save_both_summaries(self, recent_summary: str, long_running_summary: str) -> None:
        """
        Save both recent and long-running summaries together in a single operation.
        This avoids race conditions from saving them separately.

        Args:
            recent_summary: The recent summary text
            long_running_summary: The long-running summary text
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"[save_both_summaries] Saving turn {self._turn_counter}")
        logger.info(f"  Recent summary (first 100): {recent_summary[:100]}...")
        logger.info(f"  Long-running (first 100): {long_running_summary[:100]}...")

        # Save both summaries with current turn number in a single operation
        self.db.save_summary(
            session_id=self.session_id,
            turn_number=self._turn_counter,
            recent_summary=recent_summary,
            long_running_summary=long_running_summary
        )

    def update_summary(self, new_summary: str) -> None:
        """
        DEPRECATED: Use save_both_summaries() instead to avoid race conditions.

        Update the recent summary (called by summarizer, persisted to database)

        Args:
            new_summary: New summary text to store
        """
        import logging
        logger = logging.getLogger(__name__)

        # Get existing long-running summary if any
        summary_data = self.db.get_latest_summary(self.session_id)
        long_running = summary_data[1] if summary_data else ""

        logger.warning(f"[update_summary] DEPRECATED - use save_both_summaries instead")
        logger.info(f"  Recent summary (first 100): {new_summary[:100]}...")
        logger.info(f"  Long-running (from DB, first 100): {long_running[:100] if long_running else 'EMPTY'}...")

        # Save both summaries with current turn number
        self.db.save_summary(
            session_id=self.session_id,
            turn_number=self._turn_counter,
            recent_summary=new_summary,
            long_running_summary=long_running
        )

    def get_long_running_summary(self) -> str:
        """
        Get the complete long-running summary of all game history from database

        Returns:
            Long-running summary string
        """
        summary_data = self.db.get_latest_summary(self.session_id)
        if summary_data and summary_data[1]:
            return summary_data[1]  # long_running_summary is second element
        return "No history available yet."

    def update_long_running_summary(self, new_summary: str) -> None:
        """
        DEPRECATED: Use save_both_summaries() instead to avoid race conditions.

        Update the long-running summary (called by summarizer, persisted to database)

        Args:
            new_summary: New long-running summary text to store
        """
        import logging
        logger = logging.getLogger(__name__)

        # Get existing recent summary if any
        summary_data = self.db.get_latest_summary(self.session_id)
        recent = summary_data[0] if summary_data else ""

        logger.warning(f"[update_long_running_summary] DEPRECATED - use save_both_summaries instead")
        logger.info(f"  Recent (from DB, first 100): {recent[:100] if recent else 'EMPTY'}...")
        logger.info(f"  Long-running summary (first 100): {new_summary[:100]}...")

        # Save both summaries with current turn number
        # WARNING: This uses INSERT OR REPLACE, which can cause race conditions!
        self.db.save_summary(
            session_id=self.session_id,
            turn_number=self._turn_counter,
            recent_summary=recent,
            long_running_summary=new_summary
        )

    def get_turn_count(self) -> int:
        """Get the total number of turns recorded"""
        return self._turn_counter

from typing import List, Optional
from pydantic import BaseModel


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
    Maintains raw turns and summary without LLM logic.
    """

    def __init__(self):
        self.raw_turns: List[GameTurn] = []
        self.summary: str = ""  # Recent summary (last 15 turns)
        self.long_running_summary: str = ""  # Complete summary of everything
        self.previous_command: str = "LOOK"
        self._turn_counter: int = 0

    def add_turn(self, game_response: str, player_command: str,
                 location: Optional[str] = None, score: int = 0,
                 moves: int = 0) -> GameTurn:
        """
        Add a new turn to raw history

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
        self.raw_turns.append(turn)
        self.previous_command = player_command
        return turn

    def get_recent_turns(self, n: int) -> List[GameTurn]:
        """
        Get the last N turns from history

        Args:
            n: Number of recent turns to retrieve

        Returns:
            List of most recent GameTurn objects
        """
        n = min(n, len(self.raw_turns))
        return self.raw_turns[-n:] if n > 0 else []

    def get_full_summary(self) -> str:
        """
        Get the recent summary (last 15 turns)

        Returns:
            Recent summary string
        """
        return self.summary if self.summary else "No history available yet."

    def update_summary(self, new_summary: str) -> None:
        """
        Update the recent summary (called by summarizer)

        Args:
            new_summary: New summary text to store
        """
        self.summary = new_summary

    def get_long_running_summary(self) -> str:
        """
        Get the complete long-running summary of all game history

        Returns:
            Long-running summary string
        """
        return self.long_running_summary if self.long_running_summary else "No history available yet."

    def update_long_running_summary(self, new_summary: str) -> None:
        """
        Update the long-running summary (called by summarizer)

        Args:
            new_summary: New long-running summary text to store
        """
        self.long_running_summary = new_summary

    def get_turn_count(self) -> int:
        """Get the total number of turns recorded"""
        return len(self.raw_turns)

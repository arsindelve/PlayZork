from langchain_core.language_models import BaseChatModel
from typing import List

from .history_state import HistoryState, GameTurn
from .history_summarizer import HistorySummarizer
from .history_tools import initialize_history_tools, get_history_tools
from tools.database import DatabaseManager


class HistoryToolkit:
    """
    Facade for the history tool system.
    Manages state, summarization, and provides tools for LangChain agents.
    Now with SQLite persistence.
    """

    def __init__(self, summarizer_llm: BaseChatModel, session_id: str, db: DatabaseManager):
        """
        Initialize the history toolkit with database backend

        Args:
            summarizer_llm: LLM to use for generating summaries (should be cheap)
            session_id: Unique identifier for this game session
            db: DatabaseManager instance for persistence
        """
        self.state = HistoryState(session_id=session_id, db=db)
        self.summarizer = HistorySummarizer(summarizer_llm)

        # Initialize the module-level state for tools
        initialize_history_tools(self.state)

    def update_after_turn(self,
                         game_response: str,
                         player_command: str,
                         location: str,
                         score: int,
                         moves: int) -> None:
        """
        Update history after a game turn completes.
        This should be called by the game loop after each turn.

        Args:
            game_response: Text response from the game
            player_command: Command issued by the player/agent
            location: Current location name
            score: Current game score
            moves: Current move count
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Create and add the turn to state
            turn = self.state.add_turn(
                game_response=game_response,
                player_command=player_command,
                location=location,
                score=score,
                moves=moves
            )

            logger.info(f"Added turn {turn.turn_number}: {player_command}")

            # Generate RECENT summary (last 15 turns only)
            # If we have more than 15 turns, reset and summarize from scratch
            if self.state.get_turn_count() > 15:
                # Get last 15 turns
                recent_turns = self.state.get_recent_turns(15)
                # Build a summary from just these turns
                temp_summary = ""
                for t in recent_turns:
                    # Simple concatenation for now, LLM will summarize
                    temp_summary += f"Turn {t.turn_number}: {t.player_command} -> {t.game_response[:100]}... "

                # Use the summarizer to condense
                logger.info("Generating recent summary (15+ turns)...")
                new_recent_summary = self.summarizer.generate_summary(self.state, turn)
            else:
                # Normal incremental update
                logger.info(f"Generating recent summary (turn {turn.turn_number})...")
                new_recent_summary = self.summarizer.generate_summary(self.state, turn)

            logger.info(f"Recent summary generated: {new_recent_summary[:100]}...")

            # Generate LONG-RUNNING summary (all history, comprehensive)
            logger.info("Generating long-running summary...")
            new_long_summary = self.summarizer.generate_long_running_summary(self.state, turn)
            logger.info(f"Long-running summary generated: {new_long_summary[:100]}...")

            # Save BOTH summaries together in a single operation to avoid race condition
            # Previously, we saved them separately which could cause stale data issues
            self.state.save_both_summaries(new_recent_summary, new_long_summary)
            logger.info("Both summaries saved to database")

        except Exception as e:
            logger.error(f"ERROR in update_after_turn: {e}", exc_info=True)

    def get_tools(self) -> List:
        """
        Get the list of history tools for use with LangChain agents

        Returns:
            List of tool functions that agents can call
        """
        return get_history_tools()


# Export public API
__all__ = ['HistoryToolkit', 'HistoryState', 'GameTurn']

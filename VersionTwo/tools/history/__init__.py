from langchain_openai import ChatOpenAI
from typing import List

from .history_state import HistoryState, GameTurn
from .history_summarizer import HistorySummarizer
from .history_tools import initialize_history_tools, get_history_tools


class HistoryToolkit:
    """
    Facade for the history tool system.
    Manages state, summarization, and provides tools for LangChain agents.
    """

    def __init__(self, summarizer_llm: ChatOpenAI):
        """
        Initialize the history toolkit

        Args:
            summarizer_llm: LLM to use for generating summaries (should be cheap like GPT-3.5)
        """
        self.state = HistoryState()
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
        # Create and add the turn to state
        turn = self.state.add_turn(
            game_response=game_response,
            player_command=player_command,
            location=location,
            score=score,
            moves=moves
        )

        # Generate new summary incorporating this turn
        new_summary = self.summarizer.generate_summary(self.state, turn)

        # Update the state with the new summary
        self.state.update_summary(new_summary)

    def get_tools(self) -> List:
        """
        Get the list of history tools for use with LangChain agents

        Returns:
            List of tool functions that agents can call
        """
        return get_history_tools()


# Export public API
__all__ = ['HistoryToolkit', 'HistoryState', 'GameTurn']

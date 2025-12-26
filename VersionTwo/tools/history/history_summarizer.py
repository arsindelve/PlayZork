from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from .history_state import HistoryState, GameTurn


class HistorySummarizer:
    """
    Handles LLM-based summarization of game history using a cheap model.
    Uses the same prompts as the original HistoryProcessor for consistency.
    """

    def __init__(self, llm: ChatOpenAI):
        """
        Initialize the summarizer with an LLM

        Args:
            llm: Language model for generating summaries (should be cheap like GPT-3.5)
        """
        self.llm = llm
        self.chain = self._create_chain()

    def _create_chain(self) -> Runnable:
        """
        Create the LangChain chain for summarization

        Returns:
            Runnable chain that takes summary context and returns new summary
        """
        # Import here to avoid circular dependency
        from adventurer.prompt_library import PromptLibrary

        system_message = SystemMessagePromptTemplate.from_template(
            PromptLibrary.get_history_processor_system_prompt()
        )

        human_message = HumanMessagePromptTemplate.from_template(
            PromptLibrary.get_history_processor_human_prompt()
        )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message, human_message]
        )

        return chat_prompt_template | self.llm

    def generate_summary(self, history_state: HistoryState, latest_turn: GameTurn) -> str:
        """
        Generate a new RECENT summary (last 15 turns only)

        Args:
            history_state: Current history state
            latest_turn: The most recent turn to incorporate

        Returns:
            New summary text for recent turns
        """
        import logging
        logger = logging.getLogger(__name__)

        # Get previous summary from database
        previous_summary = history_state.get_full_summary()

        # Generate variables for the prompt
        prompt_variables = {
            "summary": previous_summary if previous_summary else "This is the first turn.",
            "player_response": latest_turn.player_command,  # FIX: Use the actual latest command, not previous_command
            "game_response": latest_turn.game_response
        }

        # LOG what we're passing to the LLM
        logger.info(f"=== GENERATING SUMMARY FOR TURN {latest_turn.turn_number} ===")
        logger.info(f"Previous summary (first 100 chars): {previous_summary[:100]}...")
        logger.info(f"Latest command: {latest_turn.player_command}")
        logger.info(f"Latest location: {latest_turn.location}")
        logger.info(f"Latest game response (first 100 chars): {latest_turn.game_response[:100]}...")

        # Invoke the LLM to get a summary with descriptive LangSmith name
        result = self.chain.with_config(
            run_name=f"Summary Generation: Turn {latest_turn.turn_number} @ {latest_turn.location}"
        ).invoke(prompt_variables)

        # Extract content from AIMessage
        new_summary = result.content if hasattr(result, 'content') else str(result)

        logger.info(f"New summary (first 100 chars): {new_summary[:100]}...")

        return new_summary

    def generate_long_running_summary(self, history_state: HistoryState, latest_turn: GameTurn) -> str:
        """
        Generate a comprehensive long-running summary of all game history.
        This is more detailed and comprehensive than the recent summary.

        Args:
            history_state: Current history state
            latest_turn: The most recent turn to incorporate

        Returns:
            New long-running summary text
        """
        from adventurer.prompt_library import PromptLibrary

        # Get previous long-running summary from database
        previous_summary = history_state.get_long_running_summary()

        # Create a prompt for comprehensive summarization
        prompt = ChatPromptTemplate.from_messages([
            ("system", PromptLibrary.get_history_processor_system_prompt()),
            ("human", """Previous comprehensive summary:
{summary}

Latest interaction:
Player: {player_response}
Game: {game_response}
Location: {location}
Score: {score}
Moves: {moves}

Update the comprehensive summary to include this new interaction. Keep track of:
- All locations visited
- All items found and their locations
- All puzzles encountered (solved or unsolved)
- Key discoveries and observations
- Overall progress and score changes

Provide a detailed but concise narrative. Output ONLY the updated summary.""")
        ])

        prompt_variables = {
            "summary": previous_summary if previous_summary else "Game just started.",
            "player_response": history_state.previous_command,
            "game_response": latest_turn.game_response,
            "location": latest_turn.location or "Unknown",
            "score": latest_turn.score,
            "moves": latest_turn.moves
        }

        result = (prompt | self.llm).with_config(
            run_name=f"Long-Running Summary: Turn {latest_turn.turn_number} @ {latest_turn.location}"
        ).invoke(prompt_variables)

        return result.content if hasattr(result, 'content') else str(result)

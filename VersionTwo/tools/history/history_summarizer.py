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
        Generate a new summary incorporating the latest turn

        Args:
            history_state: Current history state
            latest_turn: The most recent turn to incorporate

        Returns:
            New summary text
        """
        # Get previous summary
        previous_summary = history_state.summary

        # Generate variables for the prompt
        prompt_variables = {
            "summary": previous_summary if previous_summary else "This is the first turn.",
            "player_response": history_state.previous_command,
            "game_response": latest_turn.game_response
        }

        # Invoke the LLM to get a summary
        result = self.chain.invoke(prompt_variables)

        # Extract content from AIMessage
        return result.content if hasattr(result, 'content') else str(result)

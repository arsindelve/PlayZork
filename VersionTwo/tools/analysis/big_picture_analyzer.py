"""Big Picture Analyzer - Provides analytical insight into game progress"""
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from tools.history import HistoryToolkit
from tools.database import DatabaseManager
from config import get_expensive_llm
from adventurer.prompt_library import PromptLibrary


class BigPictureAnalyzer:
    """
    Analyzes game state to provide strategic insight rather than just summaries.

    Unlike history summaries which are purely descriptive ("You went north, found a lamp"),
    this analyzer provides analytical insight ("You appear to be stuck in an exploration loop.
    The lamp is critical but you haven't taken it.")

    Analysis is persisted to database so other agents can access it via tools.
    """

    def __init__(
        self,
        history_toolkit: HistoryToolkit,
        session_id: str,
        db: DatabaseManager,
        current_inventory: Optional[List[str]] = None,
        current_location: Optional[str] = None
    ):
        """
        Initialize the analyzer with access to history and database.

        Args:
            history_toolkit: The HistoryToolkit to get game history from
            session_id: Game session ID for database persistence
            db: DatabaseManager for saving analysis
            current_inventory: Current items in player's inventory
            current_location: Current location name
        """
        self.history_toolkit = history_toolkit
        self.session_id = session_id
        self.db = db
        self.current_inventory = current_inventory or []
        self.current_location = current_location or "Unknown"
        self.llm = get_expensive_llm(temperature=0)

    def analyze(self, turn_number: int) -> str:
        """
        Generate a big-picture analysis of the current game state.
        Saves the analysis to database for other agents to access.

        Args:
            turn_number: Current turn number

        Returns:
            Analytical insight string about game progress, blockers, and priorities
        """
        # Get history data
        # Bounded on purpose: this prompt grows with the window until it is
        # full, and it is the dominant term in per-turn latency growth
        # (see STATUS.md, 2026-08-22 checkpoint analysis).
        from config import BIG_PICTURE_HISTORY_TURNS
        recent_turns = self.history_toolkit.state.get_recent_turns(BIG_PICTURE_HISTORY_TURNS)
        full_summary = self.history_toolkit.state.get_long_running_summary()

        # Format recent turns for analysis
        recent_turns_text = self._format_recent_turns(recent_turns)

        # Build and invoke analysis prompt
        prompt = self._build_analysis_prompt(recent_turns_text, full_summary)

        try:
            # Routed through invoke_with_retry like every other LLM call: this
            # runs on the EXPENSIVE model every turn with up to
            # BIG_PICTURE_HISTORY_TURNS turns of raw text in the prompt, and it
            # was previously invisible in the logs (bare .invoke, no markers),
            # so 3 of the ~16 LLM calls per turn could not be measured at all.
            from llm_utils import invoke_with_retry
            response = invoke_with_retry(
                self.llm,
                prompt,
                operation_name=f"BigPictureAnalyzer: Turn {turn_number}",
            )
            analysis = response.content

            # Persist to database for other agents to access
            self.db.save_strategic_analysis(self.session_id, turn_number, analysis)

            return analysis
        except Exception as e:
            return f"Analysis unavailable: {str(e)}"

    def get_latest_analysis(self) -> str:
        """
        Get the most recent strategic analysis from database.
        Used by other agents to access the analysis.

        Returns:
            The latest strategic analysis, or a message if none available
        """
        result = self.db.get_latest_strategic_analysis(self.session_id)
        if result:
            turn_number, analysis = result
            return f"[From Turn {turn_number}]\n\n{analysis}"
        return "No strategic analysis available yet."

    def _format_recent_turns(self, turns) -> str:
        """Format recent turns into readable text for analysis."""
        if not turns:
            return "No recent turns available."

        formatted = []
        for turn in turns:
            formatted.append(
                f"Turn {turn.turn_number} [{turn.location or 'Unknown'}]: "
                f"'{turn.player_command}' -> {turn.game_response[:150]}..."
            )

        return "\n".join(formatted)

    def _build_analysis_prompt(self, recent_turns: str, full_summary: str) -> list:
        """Build the analysis prompt for the LLM."""
        # Format current inventory
        if self.current_inventory:
            inventory_text = ", ".join(self.current_inventory)
        else:
            inventory_text = "Empty (carrying nothing)"

        return [
            SystemMessage(content=PromptLibrary.get_big_picture_system_prompt()),
            HumanMessage(content=PromptLibrary.get_big_picture_human_prompt(
                self.current_location, inventory_text, full_summary, recent_turns
            ))
        ]

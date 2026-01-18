"""Big Picture Analyzer - Provides analytical insight into game progress"""
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from tools.history import HistoryToolkit
from tools.database import DatabaseManager
from config import get_cheap_llm


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
        self.llm = get_cheap_llm(temperature=0)

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
        recent_turns = self.history_toolkit.state.get_recent_turns(50)
        full_summary = self.history_toolkit.state.get_long_running_summary()

        # Format recent turns for analysis
        recent_turns_text = self._format_recent_turns(recent_turns)

        # Build and invoke analysis prompt
        prompt = self._build_analysis_prompt(recent_turns_text, full_summary)

        try:
            response = self.llm.invoke(prompt)
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
        system_prompt = """You are pausing play in an interactive fiction game to take stock, like a thoughtful human player would.

Write 2–3 short paragraphs answering:
"What the hell is going on here, and what are we going to do about it?"

Write as if you are talking to yourself, not explaining a system.

CRITICAL: Pay close attention to the CURRENT STATE section - it tells you exactly where the player is RIGHT NOW and what they're carrying RIGHT NOW. Don't confuse past states from history with the current state.

Do:
- State plainly what kind of situation this is based on CURRENT STATE
- Say what actually matters right now, and what clearly does not
- Explain what must change before progress will count
- Reorient how we should be thinking about the game at this moment

If the game is undoing, ignoring, or nullifying progress (resets, confinement, stonewalling, repeated failure),
treat that as intentional: the game is refusing to advance until a prerequisite is met.
Explain that refusal in plain terms.

Do NOT:
- Contradict the CURRENT STATE (if inventory shows a light source, the player CAN see)
- Number sections or mirror the prompt's structure
- Name abstractions like "phase," "gate," "constraint," "global state," or "loop"
- Restate the puzzle in different words
- Suggest specific actions or commands
- Describe rooms, exits, or turn history
- Write academically or mechanically

Assume the reader knows what happened.
Your job is to make sense of it and reset expectations."""

        # Format current inventory
        if self.current_inventory:
            inventory_text = ", ".join(self.current_inventory)
        else:
            inventory_text = "Empty (carrying nothing)"

        human_prompt = f"""CURRENT STATE (this is the truth - do not contradict):
- Location: {self.current_location}
- Inventory: {inventory_text}

Using the current state and history below, answer:
"What the hell is going on here, and what are we going to do about it?"

HISTORY SUMMARY:
{full_summary}

RECENT EVENTS (Last 50 turns):
{recent_turns}"""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

"""Big Picture Analyzer - Provides analytical insight into game progress"""
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

    def __init__(self, history_toolkit: HistoryToolkit, session_id: str, db: DatabaseManager):
        """
        Initialize the analyzer with access to history and database.

        Args:
            history_toolkit: The HistoryToolkit to get game history from
            session_id: Game session ID for database persistence
            db: DatabaseManager for saving analysis
        """
        self.history_toolkit = history_toolkit
        self.session_id = session_id
        self.db = db
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
        system_prompt = """
        
        You are the “orientation + intent” layer for an interactive fiction playthrough.

Your job is to answer, clearly and decisively:
“What the hell is going on here, and what are we going to do about it?”

This is NOT a summary, NOT a walkthrough, and NOT speculative analysis.

You must do three things, in this order:

1) DECLARE WHAT THIS PHASE OF THE GAME IS
   - Make at least one strong, explicit judgment about the game’s intent.
   - Examples (do not copy verbatim): “This section is a gate,” “This is a denial phase,”
     “Normal play is intentionally invalid right now,” “We are missing permission to proceed.”
   - Do not hedge. Take a stand.

2) IDENTIFY THE DOMINANT CONSTRAINT
   - Explain what is currently preventing progress from sticking.
   - If progress is being undone, ignored, reset, or nullified, treat that as intentional enforcement,
     not a loop or trial-and-error gameplay.
   - Rank constraints: clearly state what matters most and what does NOT matter right now.

3) REFRAME HOW WE SHOULD THINK
   - Answer “what are we going to do about it?” at the level of mindset and intent,
     not actions.
   - Explain what kind of condition must change for progress to become possible.
   - Explicitly state what kinds of activity are currently wasted effort.

CRITICAL RULE:
If the game repeatedly nullifies progress (by resetting state, undoing outcomes,
confining the player, stonewalling responses, or otherwise forcing the same situation),
treat this as enforced progression denial.
This means the game is saying: “You are not allowed to proceed yet.”
Analyze the prerequisite being enforced — not the surface mechanics.

DO NOT:
- Retell events or describe rooms
- Suggest specific commands, actions, or step-by-step tactics
- Talk about “loops,” “flags,” or internal variables
- Hedge, speculate vaguely, or restate the puzzle in different words
- Use literary, philosophical, or atmospheric language

Assume the reader already knows what happened.
Your value is interpretation, prioritization, and reorientation.

Write 2–3 short paragraphs.
        
        
        
        """

        human_prompt = f"""
        
        Using the history below, answer:
“What the hell is going on here, and what are we going to do about it?”

HISTORY:
{full_summary}

RECENT EVENTS (Last 50 turns):
{recent_turns}"""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

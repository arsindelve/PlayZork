"""Death Analyzer - Detects and analyzes player deaths using LLM"""
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from tools.history import HistoryToolkit
from tools.database import DatabaseManager
from config import get_cheap_llm


class DeathAnalysis(BaseModel):
    """Structured output for death analysis"""
    died: bool = Field(description="Whether the player died this turn")
    cause_of_death: str = Field(default="", description="What killed the player")
    events_leading_to_death: str = Field(default="", description="The sequence of events that led to death")
    recommendations: str = Field(default="", description="How to avoid this death in the future")


class DeathAnalyzer:
    """
    Analyzes game state to detect and document player deaths.

    After each turn, checks if the player died. If so, uses LLM to analyze:
    - What caused the death
    - What events led up to it
    - How to avoid similar deaths in the future

    Deaths are persisted to database for learning and display in reports.
    """

    def __init__(self, history_toolkit: HistoryToolkit, session_id: str, db: DatabaseManager):
        """
        Initialize the death analyzer.

        Args:
            history_toolkit: The HistoryToolkit to get game history from
            session_id: Game session ID for database persistence
            db: DatabaseManager for saving death records
        """
        self.history_toolkit = history_toolkit
        self.session_id = session_id
        self.db = db
        self.llm = get_cheap_llm(temperature=0)

    def analyze_turn(
        self,
        turn_number: int,
        game_response: str,
        player_command: str,
        location: str,
        score: int,
        moves: int
    ) -> Optional[DeathAnalysis]:
        """
        Analyze a turn for death and persist if found.

        Args:
            turn_number: Current turn number
            game_response: The game's response text for this turn
            player_command: The command that was executed
            location: Current location
            score: Current score
            moves: Current move count

        Returns:
            DeathAnalysis if death was detected and analyzed, None otherwise
        """
        # First, detect if death occurred using LLM
        analysis = self._analyze_for_death(game_response, player_command)

        if not analysis.died:
            return None

        # Get recent history for context
        recent_turns = self.history_toolkit.state.get_recent_turns(10)
        recent_context = self._format_recent_turns(recent_turns)

        # Do full analysis with context
        full_analysis = self._analyze_death_with_context(
            game_response,
            player_command,
            recent_context
        )

        # Persist to database
        self.db.add_death(
            session_id=self.session_id,
            turn_number=turn_number,
            location=location,
            score=score,
            moves=moves,
            cause_of_death=full_analysis.cause_of_death,
            events_leading_to_death=full_analysis.events_leading_to_death,
            recommendations=full_analysis.recommendations,
            game_response=game_response,
            player_command=player_command
        )

        return full_analysis

    def get_all_deaths(self) -> list:
        """
        Get all deaths from this session.

        Returns:
            List of death records from database
        """
        return self.db.get_all_deaths(self.session_id)

    def get_death_count(self) -> int:
        """Get the number of deaths in this session."""
        return self.db.get_death_count(self.session_id)

    def _analyze_for_death(self, game_response: str, player_command: str) -> DeathAnalysis:
        """
        Quick check if the player died this turn.

        Args:
            game_response: The game's response text
            player_command: The command that was executed

        Returns:
            DeathAnalysis with died=True/False (other fields may be empty)
        """
        llm_with_structure = self.llm.with_structured_output(DeathAnalysis)

        prompt = [
            SystemMessage(content="""You are analyzing a text adventure game response to determine if the player died.

Look for death indicators such as:
- "You have died"
- "You are dead"
- "You died"
- "Your adventure is over"
- "You have been killed"
- "You have been slain"
- Score resetting to 0 with death message
- Game over messages
- Being eaten, drowned, crushed, etc.

If the player died, set died=True. Otherwise set died=False.
For this quick check, you can leave the other fields empty."""),
            HumanMessage(content=f"""Did the player die from this command?

COMMAND: {player_command}

GAME RESPONSE:
{game_response}

Determine if the player died.""")
        ]

        try:
            return llm_with_structure.invoke(prompt)
        except Exception:
            # If analysis fails, assume no death
            return DeathAnalysis(died=False)

    def _analyze_death_with_context(
        self,
        game_response: str,
        player_command: str,
        recent_context: str
    ) -> DeathAnalysis:
        """
        Full analysis of a confirmed death.

        Args:
            game_response: The game response containing the death
            player_command: The command that led to death
            recent_context: Recent turn history for context

        Returns:
            Complete DeathAnalysis with cause, events, and recommendations
        """
        llm_with_structure = self.llm.with_structured_output(DeathAnalysis)

        prompt = [
            SystemMessage(content="""You are analyzing a player death in a text adventure game.

Your job is to:
1. Identify the CAUSE of death - what specifically killed the player (e.g., "eaten by a grue", "fell into a pit", "drowned")
2. Trace the EVENTS leading to death - what decisions or circumstances led to this outcome
3. Provide RECOMMENDATIONS - specific, actionable advice for avoiding this death in future playthroughs

Be concise but thorough. Focus on practical lessons learned."""),
            HumanMessage(content=f"""The player just died. Analyze this death.

FATAL COMMAND: {player_command}

GAME RESPONSE (containing death):
{game_response}

RECENT HISTORY (events leading up to death):
{recent_context}

Provide a complete analysis with:
- cause_of_death: What killed the player
- events_leading_to_death: The sequence of events/decisions that led here
- recommendations: How to avoid this death in the future""")
        ]

        try:
            analysis = llm_with_structure.invoke(prompt)
            analysis.died = True
            return analysis
        except Exception as e:
            # Return basic analysis if LLM fails
            return DeathAnalysis(
                died=True,
                cause_of_death="Unknown - analysis failed",
                events_leading_to_death=f"Command '{player_command}' resulted in death",
                recommendations=f"Avoid the action that caused death: {player_command}"
            )

    def _format_recent_turns(self, turns) -> str:
        """Format recent turns into readable text for analysis."""
        if not turns:
            return "No recent turns available."

        formatted = []
        for turn in turns:
            formatted.append(
                f"Turn {turn.turn_number} [{turn.location or 'Unknown'}]: "
                f"'{turn.player_command}' -> {turn.game_response[:200]}..."
            )

        return "\n".join(formatted)

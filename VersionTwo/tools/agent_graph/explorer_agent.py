"""ExplorerAgent - Single agent per turn that advocates for exploring unexplored directions"""
from typing import Optional, List
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from adventurer.prompt_library import PromptLibrary
from .tool_execution import invoke_tool_safely
import logging


class ExplorerProposal(BaseModel):
    """Proposed exploration action from an ExplorerAgent"""
    proposed_action: str      # e.g., "GO NORTH" or "NORTH"
    reason: str               # Why explore this direction
    confidence: int           # 1-100 score based on unexplored count + mention


class ExplorerAgent:
    """Single agent per turn that advocates for exploring the best unexplored direction."""

    # Cardinal directions to check (in priority order)
    CARDINAL_DIRECTIONS = [
        "NORTH", "SOUTH", "EAST", "WEST",
        "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST",
        "UP", "DOWN"
    ]

    def __init__(
        self,
        current_location: str,
        unexplored_directions: List[str],
        mentioned_directions: List[str],
        turn_number: int,
        game_exits: Optional[List[str]] = None
    ):
        """
        Initialize the single ExplorerAgent for this turn.

        Args:
            current_location: Current location name
            unexplored_directions: List of all unexplored cardinal directions
            mentioned_directions: List of directions mentioned in game text (subset of unexplored)
            turn_number: Current turn number
        """
        self.current_location = current_location
        self.unexplored_directions = unexplored_directions
        self.mentioned_directions = mentioned_directions
        self.turn_number = turn_number
        # Directions the game itself reports (#30); ranks candidates.
        self.game_exits = list(game_exits or [])

        # Proposal fields (populated after research)
        self.proposed_action: Optional[str] = None
        self.reason: Optional[str] = None
        self.confidence: Optional[int] = None
        self.research_context: Optional[str] = None

        # Tool call history (for reporting)
        self.tool_calls_history: list = []

        # Pick the best direction and calculate confidence
        self.best_direction = self._pick_best_direction()

    def _pick_best_direction(self) -> str:
        """Pick the direction most likely to lead somewhere.

        The old rule fell back to a FIXED cardinal order — NORTH, SOUTH, EAST,
        WEST — whenever the room description mentioned nothing. That is not a
        tiebreak, it is a systematic northward bias, and it showed: over a
        26-turn run the agent walked north into the forest and mapped Clearing,
        Canyon View and Rocky Ledge while never returning to the house it
        started beside. The ExplorerAgent won 64% of contested turns, so its
        bias was effectively the agent's policy.

        Candidates are now SCORED on real evidence:

          +3  the game's own exits array says this direction exists (#30)
          +2  the room description mentions it
          +1  a cardinal rather than a diagonal (cheaper to describe, and
              Zork's world is mostly cardinal)

        The exits array is not a perfect oracle — North of House advertises an
        exit that is then refused — so it ranks candidates rather than
        restricting them, and a direction it omits can still be chosen if
        nothing better is on offer.
        """
        if not self.unexplored_directions:
            return "NORTH"

        cardinals = {"NORTH", "SOUTH", "EAST", "WEST"}
        game_exits = {d.upper() for d in (self.game_exits or [])}
        mentioned = {d.upper() for d in (self.mentioned_directions or [])}

        def score(direction: str) -> tuple:
            points = 0
            if direction in game_exits:
                points += 3
            if direction in mentioned:
                points += 2
            if direction in cardinals:
                points += 1
            # Stable tiebreak on the canonical order, so a run is reproducible.
            return (-points, self.unexplored_directions.index(direction))

        return min(self.unexplored_directions, key=score)

    def _calculate_confidence(self, chosen_direction: str) -> int:
        """
        Calculate confidence score for the chosen direction.

        Args:
            chosen_direction: The direction we've chosen to propose

        Returns:
            Confidence score (1-100)
        """
        unexplored_count = len(self.unexplored_directions)

        # Base confidence from unexplored count
        if unexplored_count >= 6:
            base = 75
        elif unexplored_count >= 4:
            base = 65
        elif unexplored_count >= 2:
            base = 55
        else:
            base = 45

        # Bonus if chosen direction was mentioned
        bonus = 20 if chosen_direction in self.mentioned_directions else 0

        # Cap at 95 (never 100% certain)
        return min(base + bonus, 95)

    async def propose(
        self,
        decision_llm: BaseChatModel,
        context,
    ) -> None:
        """Generate the exploration proposal for `best_direction`.

        The research round-trip is gone (#25): it asked the model to "use the
        mapper tools to understand this location", executed whatever came
        back once, and never iterated. TurnContext already holds the map.
        """
        logger = logging.getLogger(__name__)
        # Function-local import: tests monkeypatch llm_utils.ainvoke_with_retry
        from llm_utils import ainvoke_with_retry

        logger.info(f"[ExplorerAgent:{self.best_direction}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[ExplorerAgent] CURRENT LOCATION: {self.current_location}")
        logger.info(f"[ExplorerAgent] BEST DIRECTION: {self.best_direction}")
        logger.info(f"[ExplorerAgent] UNEXPLORED COUNT: {len(self.unexplored_directions)}")
        logger.info(f"[ExplorerAgent:{self.best_direction}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        current_game_response = context.game_text
        self.research_context = context.research_context_for()

        # Calculate confidence for the chosen direction
        self.confidence = self._calculate_confidence(self.best_direction)

        # Phase 2: Generate proposal
        # Build system message with conditional text
        if self.best_direction in self.mentioned_directions:
            why_chosen = "It was mentioned in the location description"
        else:
            why_chosen = "It is a cardinal direction to try systematically"

        proposal_prompt = ChatPromptTemplate.from_messages([
            ("system", PromptLibrary.get_explorer_agent_system_prompt(why_chosen)),
            ("human", PromptLibrary.get_explorer_agent_human_prompt())
        ])

        proposal_chain = proposal_prompt | decision_llm.with_structured_output(ExplorerProposal)

        proposal = await ainvoke_with_retry(
            proposal_chain.with_config(
                run_name=f"ExplorerAgent Proposal: {self.best_direction} from {self.current_location}"
            ),
            {
                "best_direction": self.best_direction,
                "current_location": self.current_location,
                "unexplored_count": len(self.unexplored_directions),
                "all_unexplored": ", ".join(self.unexplored_directions),
                "mentioned_dirs": ", ".join(self.mentioned_directions) if self.mentioned_directions else "None",
                "confidence": self.confidence,
                "game_response": current_game_response,
                "research_context": self.research_context
            },
            operation_name="ExplorerAgent Proposal"
        )

        self.proposed_action = proposal.proposed_action
        self.reason = proposal.reason

        # Log proposal summary
        logger.info(f"[ExplorerAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[ExplorerAgent] PROPOSAL SUMMARY")
        logger.info(f"[ExplorerAgent] Direction: {self.best_direction} from {self.current_location}")
        logger.info(f"[ExplorerAgent] Proposed Action: '{self.proposed_action}' (confidence: {self.confidence}/100)")
        if self.reason:
            logger.info(f"[ExplorerAgent] Reason: {self.reason}")
        logger.info(f"[ExplorerAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

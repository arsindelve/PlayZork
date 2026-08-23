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
        turn_number: int
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
        """
        Pick the best direction to explore based on priority rules.

        Priority:
        1. Mentioned in description (any mentioned direction)
        2. Cardinal directions (NORTH, SOUTH, EAST, WEST) - prefer first
        3. Diagonal directions (NE, NW, SE, SW)
        4. UP/DOWN

        Returns:
            Best direction to explore
        """
        # First priority: Directions mentioned in description
        if self.mentioned_directions:
            # Pick first mentioned direction
            return self.mentioned_directions[0]

        # Second priority: Cardinal directions
        cardinals = ["NORTH", "SOUTH", "EAST", "WEST"]
        for direction in cardinals:
            if direction in self.unexplored_directions:
                return direction

        # Third priority: Diagonals
        diagonals = ["NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST"]
        for direction in diagonals:
            if direction in self.unexplored_directions:
                return direction

        # Last priority: UP/DOWN
        for direction in ["UP", "DOWN"]:
            if direction in self.unexplored_directions:
                return direction

        # Fallback (shouldn't reach here if unexplored_directions is non-empty)
        return self.unexplored_directions[0] if self.unexplored_directions else "NORTH"

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

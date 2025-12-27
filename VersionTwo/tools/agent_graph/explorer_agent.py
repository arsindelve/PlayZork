"""ExplorerAgent - Single agent per turn that advocates for exploring unexplored directions"""
from typing import Optional, List
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable


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

        # Pick best direction and calculate confidence
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

    def research_and_propose(
        self,
        research_agent: Runnable,
        decision_llm: BaseChatModel,
        history_tools: list,
        mapper_tools: list,
        current_game_response: str,
        current_score: int,
        current_moves: int
    ) -> None:
        """
        Phase 1: Research using tools (optional - understand map topology)
        Phase 2: Generate exploration proposal for best_direction
        """
        # Phase 1: Research (call mapper tools to understand geography)
        research_input = {
            "input": f"You are planning exploration from '{self.current_location}'. "
                     f"There are {len(self.unexplored_directions)} unexplored directions: {', '.join(self.unexplored_directions)}. "
                     f"Use the mapper tools to understand what we know about this location and surrounding areas.",
            "score": current_score,
            "locationName": self.current_location,
            "moves": current_moves,
            "game_response": current_game_response
        }

        try:
            from config import invoke_with_retry
            research_response = invoke_with_retry(
                research_agent.with_config(
                    run_name=f"ExplorerAgent Research: {self.best_direction} from {self.current_location}"
                ),
                research_input,
                operation_name="ExplorerAgent Research"
            )

            # Execute tool calls if present
            if hasattr(research_response, 'tool_calls') and research_response.tool_calls:
                tool_results = []
                all_tools = history_tools + mapper_tools
                tools_map = {tool.name: tool for tool in all_tools}

                logger.info(f"[ExplorerAgent] Made {len(research_response.tool_calls)} tool calls:")

                for tool_call in research_response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call.get('args', {})

                    logger.info(f"  -> {tool_name}({tool_args})")

                    if tool_name in tools_map:
                        tool_result = tools_map[tool_name].invoke(tool_args)
                        logger.info(f"     Result: {str(tool_result)[:150]}...")
                        tool_results.append(f"{tool_name} result: {tool_result}")

                self.research_context = "\n\n".join(tool_results) if tool_results else "No tools called"
            else:
                logger.info(f"[ExplorerAgent] No tool calls made")
                self.research_context = research_response.content if hasattr(research_response, 'content') else str(research_response)

        except Exception as e:
            self.research_context = f"Research failed: {str(e)}"

        # Calculate confidence for the chosen direction
        self.confidence = self._calculate_confidence(self.best_direction)

        # Phase 2: Generate proposal
        # Build system message with conditional text
        if self.best_direction in self.mentioned_directions:
            why_chosen = "It was mentioned in the location description"
        else:
            why_chosen = "It is a cardinal direction to try systematically"

        proposal_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an ExplorerAgent focused on systematic exploration in Zork.

Your task: Advocate for exploring the {{best_direction}} direction from the current location.

This direction was chosen as the best option from {{unexplored_count}} unexplored directions because:
- It follows priority rules (mentioned > cardinals > diagonals > up/down)
- {why_chosen}

Rules for proposed_action:
- Propose exploring {{best_direction}} (e.g., "GO {{best_direction}}" or just "{{best_direction}}")
- Use standard Zork command format

Rules for reason:
- Explain why exploring this direction makes sense now
- Mention if it was in the location description
- Note how many other directions remain unexplored
- Keep it concise (1-2 sentences)

Output format: ExplorerProposal with proposed_action, reason, and confidence={{confidence}}.
"""),
            ("human", """BEST DIRECTION TO EXPLORE:
{best_direction}

CURRENT LOCATION:
{current_location}

ALL UNEXPLORED DIRECTIONS ({unexplored_count}):
{all_unexplored}

MENTIONED DIRECTIONS:
{mentioned_dirs}

CURRENT GAME STATE:
{game_response}

RESEARCH CONTEXT:
{research_context}

Propose exploring {best_direction}.""")
        ])

        try:
            from config import invoke_with_retry
            proposal_chain = proposal_prompt | decision_llm.with_structured_output(ExplorerProposal)

            proposal = invoke_with_retry(
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
            # Confidence already calculated

        except Exception as e:
            self.proposed_action = self.best_direction
            self.reason = f"Explore {self.best_direction} (unexplored direction, {len(self.unexplored_directions)} total remaining)"
            # Keep calculated confidence

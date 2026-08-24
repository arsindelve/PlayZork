"""IssueAgent - Sub-agent for tracking individual puzzles/obstacles/issues"""
from tools.memory import Memory
from typing import Optional
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from adventurer.prompt_library import PromptLibrary
from .tool_execution import invoke_tool_safely, TOOL_ERROR_PREFIX
from tools.mapping.locations import UNKNOWN_LOCATION, is_known_location
import logging


class IssueProposal(BaseModel):
    """Proposed action and confidence from an IssueAgent"""
    proposed_action: str  # What the adventurer should do this turn (or "nothing")
    reason: str  # Why this action will help solve the issue
    confidence: int  # 1-100 score of how much this will help solve the issue


class IssueAgent:
    """
    Represents a sub-agent focused on a single strategic issue.

    Each IssueAgent performs its own research cycle and proposes actions
    to solve its specific puzzle/obstacle.
    """

    def __init__(self, memory: Memory):
        """
        Initialize an IssueAgent for a specific issue.

        Args:
            memory: The Memory object containing the issue details
        """
        self.memory = memory

        # Core issue properties (for easy access)
        self.issue_content = memory.content
        self.importance = memory.importance
        self.turn_number = memory.turn_number
        self.location = memory.location
        self.score = memory.score
        self.moves = memory.moves

        # Proposal fields (populated after research)
        self.proposed_action: Optional[str] = None
        self.reason: Optional[str] = None
        self.confidence: Optional[int] = None
        self.research_context: Optional[str] = None

        # Tool call history (for reporting)
        self.tool_calls_history: list = []

    def __str__(self) -> str:
        return f"IssueAgent[{self.importance}/1000] tracking: '{self.issue_content}' (from turn {self.turn_number})"

    def __repr__(self) -> str:
        return self.__str__()

    def get_issue_summary(self) -> str:
        """Get a formatted summary of this issue"""
        return (
            f"Issue: {self.issue_content}\n"
            f"Importance: {self.importance}/1000\n"
            f"Location: {self.location}\n"
            f"Discovered: Turn {self.turn_number}"
        )

    @staticmethod
    def _declined(proposal) -> bool:
        """True when the agent effectively proposed nothing.

        Local models express this several ways — an empty string, the literal
        word "nothing", or a real-looking action at zero confidence — and all
        three reach the arbiter as an unusable proposal.
        """
        action = (getattr(proposal, "proposed_action", "") or "").strip().lower()
        confidence = getattr(proposal, "confidence", 0) or 0
        return (not action) or action in ("nothing", "none", "n/a") or confidence <= 0

    async def propose(
        self,
        decision_llm: BaseChatModel,
        context,
    ) -> IssueProposal:
        """Generate a proposal for solving this issue.

        The old phase-1 "research" LLM round-trip is gone (#25). It asked a
        14B model for permission to run SQLite queries whose arguments the
        code already knew, executed them once, never fed results back for a
        second round, and discarded `response.content` whenever tool calls
        existed. TurnContext now supplies the same facts deterministically —
        and completely, where the LLM route could silently return any subset
        (#4, #5, #6).

        Args:
            decision_llm: LLM for generating the structured proposal
            context: This turn's TurnContext

        Returns:
            IssueProposal with proposed_action and confidence score
        """
        logger = logging.getLogger(__name__)
        # Function-local import: tests monkeypatch llm_utils.ainvoke_with_retry
        from llm_utils import ainvoke_with_retry

        logger.info(f"[IssueAgent ID:{self.memory.id}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[IssueAgent ID:{self.memory.id}] AGENT: IssueAgent")
        logger.info(f"[IssueAgent ID:{self.memory.id}] ISSUE: {self.issue_content}")
        logger.info(f"[IssueAgent ID:{self.memory.id}] IMPORTANCE: {self.importance}/1000")
        logger.info(f"[IssueAgent ID:{self.memory.id}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        current_location = context.location
        current_game_response = context.game_text

        # Where the issue actually points, which is not always where it was
        # noticed. "Ensign Blather at Reactor Lobby — return to Deck Nine as
        # ordered" was navigated toward Reactor Lobby, the room we were already
        # standing in, so the route was NO PATH and this agent had nothing to
        # offer while the escape pod sat two rooms away.
        from tools.memory.issue_target import resolve_issue_target
        self.target_location = resolve_issue_target(
            self.issue_content, self.location,
            getattr(context, "known_locations", None)) or self.location

        # Everything the research phase used to fetch, fetched in code.
        navigation_direction = context.direction_to(self.target_location)
        inventory_summary = context.inventory_summary
        self.research_context = context.research_context_for(self.target_location)

        logger.info(f"[IssueAgent ID:{self.memory.id}] Navigation direction: {navigation_direction}")
        logger.info(f"[IssueAgent ID:{self.memory.id}] Inventory: {inventory_summary}")

        proposal_prompt = ChatPromptTemplate.from_messages([
            ("system", PromptLibrary.get_issue_agent_system_prompt()),
            ("human", PromptLibrary.get_issue_agent_human_prompt())
        ])

        proposal_chain = proposal_prompt | decision_llm.with_structured_output(IssueProposal)

        logger.info(f"[IssueAgent ID:{self.memory.id}] Calling proposal_chain.invoke()...")

        # Calculate location status for spatial reasoning
        if self.target_location and is_known_location(current_location):
            issue_loc_normalized = self.target_location.strip().lower()
            current_loc_normalized = current_location.strip().lower()
            location_status = "SAME LOCATION" if issue_loc_normalized == current_loc_normalized else "DIFFERENT LOCATION"
        else:
            location_status = "UNKNOWN"

        proposal = await ainvoke_with_retry(
            proposal_chain.with_config(
                run_name=f"IssueAgent Proposal: {self.issue_content[:60]}"
            ),
            {
                "issue": self.issue_content,
                "issue_location": self.target_location or "Unknown",
                "current_location": current_location,
                "location_status": location_status,
                "navigation_direction": navigation_direction,
                "inventory_summary": inventory_summary,
                "game_response": current_game_response,
                "research_context": self.research_context
            },
            operation_name=f"IssueAgent Proposal: {self.issue_content[:40]}"
        )
        logger.info(f"[IssueAgent ID:{self.memory.id}] Proposal generated: {proposal.proposed_action} (confidence: {proposal.confidence})")

        # Store proposal
        self.proposed_action = proposal.proposed_action
        self.reason = proposal.reason
        self.confidence = proposal.confidence

        # An issue you are not standing at has a deterministic answer: walk
        # toward it. The prompt already carries `navigation_direction` and
        # `location_status`, and the model declined anyway — in pf-20260824 it
        # returned "nothing" at confidence 0 for a 900-importance escape pod
        # two rooms away, while the ship's clock ran. Same shape as #21: a rule
        # stated only in prose is not a mechanism, so enforce it here.
        #
        # Confidence describes the RELIABILITY OF THE ACTION, not the worth of
        # the issue: one step along a BFS shortest path over edges we recorded
        # ourselves is about as dependable as a proposal gets. Whether the
        # issue deserves pursuing is already priced in by the importance term
        # of the expected value — 900 importance gives EV 63 and outranks
        # exploration's 47.5, a decayed 300 gives 21 and does not.
        if self._declined(proposal) and location_status == "DIFFERENT LOCATION":
            step = (navigation_direction or "").strip().upper()
            if step and step not in ("NO PATH", "NOT AVAILABLE", "UNKNOWN"):
                self.proposed_action = step
                self.confidence = 70
                self.reason = (
                    f"Cannot act on this issue from {current_location}; it is "
                    f"at {self.target_location}. {step} is the next step on the known "
                    f"route there.")
                logger.info(
                    f"[IssueAgent ID:{self.memory.id}] Declined with no action; "
                    f"substituting route step {step} toward {self.target_location}")

        # Log proposal summary
        logger.info(f"[IssueAgent ID:{self.memory.id}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[IssueAgent ID:{self.memory.id}] PROPOSAL SUMMARY")
        logger.info(f"[IssueAgent ID:{self.memory.id}] Issue: [{self.importance}/1000] {self.issue_content}")
        logger.info(f"[IssueAgent ID:{self.memory.id}] Proposed Action: '{self.proposed_action}' (confidence: {self.confidence}/100)")
        if self.reason:
            logger.info(f"[IssueAgent ID:{self.memory.id}] Reason: {self.reason}")
        logger.info(f"[IssueAgent ID:{self.memory.id}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return proposal

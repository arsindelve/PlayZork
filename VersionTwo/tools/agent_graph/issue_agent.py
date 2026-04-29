"""IssueAgent - Sub-agent for tracking individual puzzles/obstacles/issues"""
from tools.memory import Memory
from typing import Optional
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from adventurer.prompt_library import PromptLibrary
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

    async def research_and_propose(
        self,
        research_agent: Runnable,
        decision_llm: BaseChatModel,
        history_tools: list,
        current_location: str,
        current_game_response: str,
        current_score: int,
        current_moves: int
    ) -> IssueProposal:
        """
        Execute research cycle and generate a proposal for solving this issue.

        Args:
            research_agent: LLM chain with tools for calling history
            decision_llm: LLM for generating structured proposal
            history_tools: List of available history tools
            current_location: Current game location
            current_game_response: Latest game response text
            current_score: Current game score
            current_moves: Current move count

        Returns:
            IssueProposal with proposed_action and confidence score
        """
        logger = logging.getLogger(__name__)

        logger.info(f"[IssueAgent ID:{self.memory.id}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[IssueAgent ID:{self.memory.id}] AGENT: IssueAgent")
        logger.info(f"[IssueAgent ID:{self.memory.id}] ID: {self.memory.id}")
        logger.info(f"[IssueAgent ID:{self.memory.id}] ISSUE: {self.issue_content}")
        logger.info(f"[IssueAgent ID:{self.memory.id}] IMPORTANCE: {self.importance}/1000")
        logger.info(f"[IssueAgent ID:{self.memory.id}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[IssueAgent ID:{self.memory.id}] Phase 1: Research for '{self.issue_content}'")

        # Determine location status for research instructions
        if self.location and current_location:
            issue_loc_normalized = self.location.strip().lower()
            current_loc_normalized = current_location.strip().lower()
            is_same_location = issue_loc_normalized == current_loc_normalized
        else:
            is_same_location = True

        # Build research instruction based on location
        if not is_same_location and self.location:
            research_instruction = (
                f"You are investigating this strategic issue: '{self.issue_content}'. "
                f"The issue is at '{self.location}' but you are at '{current_location}'. "
                f"REQUIRED: 1) Call get_direction_to_location(from_location='{current_location}', to_location='{self.location}') to find path. "
                f"2) Call get_current_inventory() to check if you have items that could solve this issue. "
                f"3) Call get_full_summary() for context."
            )
        else:
            research_instruction = (
                f"You are investigating this strategic issue: '{self.issue_content}'. "
                f"Use the available tools to gather relevant history context."
            )

        # Phase 1: Research using history tools
        # Must match the research agent prompt parameters: score, locationName, moves, game_response
        research_input = {
            "input": research_instruction,
            "score": current_score,
            "locationName": current_location,
            "moves": current_moves,
            "game_response": current_game_response
        }

        logger.info(f"[IssueAgent] Calling research_agent.ainvoke()...")
        from llm_utils import ainvoke_with_retry
        research_response = await ainvoke_with_retry(
            research_agent.with_config(
                run_name=f"IssueAgent Research: {self.issue_content[:60]}"
            ),
            research_input,
            operation_name=f"IssueAgent Research: {self.issue_content[:40]}"
        )
        logger.info(f"[IssueAgent ID:{self.memory.id}] Research agent responded successfully")

        # Execute tool calls if present
        if hasattr(research_response, 'tool_calls') and research_response.tool_calls:
            tool_results = []
            tools_map = {tool.name: tool for tool in history_tools}

            logger.info(f"[IssueAgent ID:{self.memory.id}] Made {len(research_response.tool_calls)} tool calls:")

            for tool_call in research_response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})

                logger.info(f"[IssueAgent ID:{self.memory.id}]   -> {tool_name}({tool_args})")

                if tool_name in tools_map:
                    tool_result = tools_map[tool_name].invoke(tool_args)
                    logger.info(f"[IssueAgent ID:{self.memory.id}]      Result: {str(tool_result)[:150]}...")
                    tool_results.append(f"{tool_name} result: {tool_result}")

                    # Store tool call history for reporting
                    self.tool_calls_history.append({
                        "tool_name": tool_name,
                        "input": str(tool_args),
                        "output": str(tool_result)
                    })

            self.research_context = "\n\n".join(tool_results) if tool_results else "No tools executed."
        else:
            logger.info(f"[IssueAgent ID:{self.memory.id}] No tool calls made")
            self.research_context = research_response.content if hasattr(research_response, 'content') else str(research_response)

        # Extract navigation direction and inventory from research
        navigation_direction = "NOT CHECKED"
        inventory_items = []

        for tool_call in self.tool_calls_history:
            tool_name = tool_call.get("tool_name", "")
            output = tool_call.get("output", "")

            if tool_name == "get_direction_to_location":
                if output in ["NO PATH", "ALREADY THERE"] or output.startswith("Error"):
                    navigation_direction = output
                else:
                    navigation_direction = output  # e.g., "SOUTH"

            elif tool_name == "get_current_inventory":
                # Parse inventory output
                if "empty" not in output.lower() and "no items" not in output.lower():
                    # Try to extract items from the inventory output
                    inventory_items = [item.strip() for item in output.replace("\n", ",").split(",") if item.strip()]

        logger.info(f"[IssueAgent ID:{self.memory.id}] Navigation direction: {navigation_direction}")
        logger.info(f"[IssueAgent ID:{self.memory.id}] Inventory items: {inventory_items}")

        # Phase 2: Generate proposal based on research
        logger.info(f"[IssueAgent ID:{self.memory.id}] Phase 2: Generating proposal for '{self.issue_content}'")
        logger.info(f"[IssueAgent ID:{self.memory.id}] Research context length: {len(self.research_context)} chars")

        proposal_prompt = ChatPromptTemplate.from_messages([
            ("system", PromptLibrary.get_issue_agent_system_prompt()),
            ("human", PromptLibrary.get_issue_agent_human_prompt())
        ])

        proposal_chain = proposal_prompt | decision_llm.with_structured_output(IssueProposal)

        logger.info(f"[IssueAgent ID:{self.memory.id}] Calling proposal_chain.invoke()...")

        # Calculate location status for spatial reasoning
        if self.location and current_location:
            issue_loc_normalized = self.location.strip().lower()
            current_loc_normalized = current_location.strip().lower()
            location_status = "SAME LOCATION" if issue_loc_normalized == current_loc_normalized else "DIFFERENT LOCATION"
        else:
            location_status = "UNKNOWN"

        # Prepare inventory summary for proposal
        inventory_summary = ", ".join(inventory_items) if inventory_items else "empty"

        proposal = await ainvoke_with_retry(
            proposal_chain.with_config(
                run_name=f"IssueAgent Proposal: {self.issue_content[:60]}"
            ),
            {
                "issue": self.issue_content,
                "issue_location": self.location or "Unknown",
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

        # Log proposal summary
        logger.info(f"[IssueAgent ID:{self.memory.id}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[IssueAgent ID:{self.memory.id}] PROPOSAL SUMMARY")
        logger.info(f"[IssueAgent ID:{self.memory.id}] Issue: [{self.importance}/1000] {self.issue_content}")
        logger.info(f"[IssueAgent ID:{self.memory.id}] Proposed Action: '{self.proposed_action}' (confidence: {self.confidence}/100)")
        if self.reason:
            logger.info(f"[IssueAgent ID:{self.memory.id}] Reason: {self.reason}")
        logger.info(f"[IssueAgent ID:{self.memory.id}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return proposal

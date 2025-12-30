"""IssueAgent - Sub-agent for tracking individual puzzles/obstacles/issues"""
from tools.memory import Memory
from typing import Optional
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from config import GAME_NAME
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

    def research_and_propose(
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

        # Phase 1: Research using history tools
        # Must match the research agent prompt parameters: score, locationName, moves, game_response
        research_input = {
            "input": f"You are investigating this strategic issue: '{self.issue_content}'. Use the available tools to gather relevant history context.",
            "score": current_score,
            "locationName": current_location,
            "moves": current_moves,
            "game_response": current_game_response
        }

        logger.info(f"[IssueAgent] Calling research_agent.invoke()...")
        # Call research agent (can call tools) with timeout and retry
        try:
            from config import invoke_with_retry
            research_response = invoke_with_retry(
                research_agent.with_config(
                    run_name=f"IssueAgent Research: {self.issue_content[:60]}"
                ),
                research_input,
                operation_name=f"IssueAgent Research: {self.issue_content[:40]}"
            )
            logger.info(f"[IssueAgent ID:{self.memory.id}] Research agent responded successfully")
        except Exception as e:
            logger.error(f"[IssueAgent ID:{self.memory.id}] Research agent failed: {e}")
            raise

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

        # Phase 2: Generate proposal based on research
        logger.info(f"[IssueAgent ID:{self.memory.id}] Phase 2: Generating proposal for '{self.issue_content}'")
        logger.info(f"[IssueAgent ID:{self.memory.id}] Research context length: {len(self.research_context)} chars")

        proposal_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an IssueAgent tasked with solving ONE SPECIFIC puzzle/obstacle in {GAME_NAME}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL QUESTION YOU MUST ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Does the action I'm proposing DIRECTLY solve MY SPECIFIC issue?"

If YES → Give high confidence (70-100)
If NO → Give confidence 0 or very low (1-20)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR RESPONSIBILITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You ONLY propose actions that solve YOUR specific issue.
You do NOT propose actions for other issues, exploration, or general progress.

EXAMPLES OF CORRECT BEHAVIOR:

Your Issue: "Locked door in Kitchen - need key"
Action: "UNLOCK DOOR WITH KEY" → Confidence 90 ✓ (solves YOUR issue)
Action: "GO NORTH" → Confidence 0 ✗ (doesn't solve YOUR issue)
Action: "EXAMINE ROOM" → Confidence 0 ✗ (doesn't solve YOUR issue)

Your Issue: "Troll blocking Bridge - need to defeat or bypass"
Action: "KILL TROLL WITH SWORD" → Confidence 85 ✓ (solves YOUR issue)
Action: "TAKE LAMP" → Confidence 0 ✗ (doesn't solve YOUR issue)
Action: "GO EAST" → Confidence 0 ✗ (doesn't solve YOUR issue)

Your Issue: "Small mailbox at West of House"
Action: "OPEN MAILBOX" → Confidence 80 ✓ (solves YOUR issue)
Action: "GO WEST" → Confidence 0 ✗ (doesn't solve YOUR issue)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOCATION AWARENESS - CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALID PROPOSALS FROM DIFFERENT LOCATIONS:
✓ Issue: "Locked door at Kitchen" / Current: "Garden"
  → "TAKE KEY" (if key is here) - Confidence 90 ✓

✓ Issue: "Need light source" / Current: "Cellar"
  → "TAKE LAMP" (if lamp is here) - Confidence 85 ✓

INVALID PROPOSALS FROM DIFFERENT LOCATIONS:
✗ Issue: "Window at Behind House" / Current: "Forest"
  → "OPEN WINDOW" - Confidence 0 ✗ (window not in current location!)

✗ Issue: "Grating at Clearing" / Current: "Forest Path"
  → "OPEN GRATING" - Confidence 0 ✗ (grating not in current location!)

RULE: If your action directly interacts with an object (OPEN, PUSH, EXAMINE, etc.)
      and that object is NOT in the current location, confidence MUST be 0.

      You can only propose taking items, finding clues, or gathering tools
      that exist in the CURRENT location to solve issues elsewhere.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES FOR PROPOSED_ACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. If you have a DIRECT solution for YOUR issue → propose it clearly
2. If you DON'T have a solution for YOUR issue → propose "nothing" and confidence 0
3. NEVER propose actions that help other issues or general exploration
4. NEVER propose movement commands unless movement DIRECTLY solves YOUR issue

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE SCORING (BE BRUTALLY HONEST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ask yourself: "Will this action DIRECTLY solve MY specific issue?"

90-100: Definite solution - this action will solve MY issue right now
70-89:  Very likely - this action should solve MY issue
50-69:  Moderate - this action might solve MY issue
20-49:  Weak - this action probably won't solve MY issue
0-19:   No solution - this action doesn't solve MY issue at all

CRITICAL: If your proposed action doesn't DIRECTLY address YOUR SPECIFIC issue,
your confidence MUST be 0 or very low (1-20).

Don't give 70+ confidence unless the action DIRECTLY solves YOUR issue!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASON FIELD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Explain EXACTLY HOW this action solves YOUR SPECIFIC issue.
If you can't explain how it solves YOUR issue → confidence should be 0.

Respond with structured output."""),
            ("human", """ISSUE YOU ARE SOLVING:
{issue}

ISSUE LOCATION:
{issue_location}

YOUR CURRENT LOCATION:
{current_location}

LOCATION STATUS:
{location_status}

CURRENT GAME STATE:
{game_response}

RESEARCH CONTEXT:
{research_context}

CRITICAL: Consider whether your proposed action can be performed from your CURRENT location.
- Direct object interaction (OPEN, TAKE, PUSH, EXAMINE, etc.) usually requires being AT the object's location
- Finding items/clues that solve the issue CAN happen in other locations
- If you need to interact with the object but are in the wrong location, confidence should be 0

What should the adventurer do THIS TURN to make progress on YOUR issue?""")
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

        try:
            from config import invoke_with_retry
            proposal = invoke_with_retry(
                proposal_chain.with_config(
                    run_name=f"IssueAgent Proposal: {self.issue_content[:60]}"
                ),
                {
                    "issue": self.issue_content,
                    "issue_location": self.location or "Unknown",
                    "current_location": current_location,
                    "location_status": location_status,
                    "game_response": current_game_response,
                    "research_context": self.research_context
                },
                operation_name=f"IssueAgent Proposal: {self.issue_content[:40]}"
            )
            logger.info(f"[IssueAgent ID:{self.memory.id}] Proposal generated: {proposal.proposed_action} (confidence: {proposal.confidence})")
        except Exception as e:
            logger.error(f"[IssueAgent ID:{self.memory.id}] Proposal generation failed: {e}")
            raise

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

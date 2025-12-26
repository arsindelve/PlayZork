"""IssueAgent - Sub-agent for tracking individual puzzles/obstacles/issues"""
from tools.memory import Memory
from typing import Optional
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
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
        decision_llm: ChatOpenAI,
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

        logger.info(f"[IssueAgent] Phase 1: Research for '{self.issue_content}'")

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
        # Call research agent (can call tools) with descriptive LangSmith name
        try:
            research_response = research_agent.with_config(
                run_name=f"IssueAgent Research: {self.issue_content[:60]}"
            ).invoke(research_input)
            logger.info(f"[IssueAgent] Research agent responded successfully")
        except Exception as e:
            logger.error(f"[IssueAgent] Research agent failed: {e}")
            raise

        # Execute tool calls if present
        if hasattr(research_response, 'tool_calls') and research_response.tool_calls:
            tool_results = []
            tools_map = {tool.name: tool for tool in history_tools}

            for tool_call in research_response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})

                if tool_name in tools_map:
                    tool_result = tools_map[tool_name].invoke(tool_args)
                    tool_results.append(f"{tool_name} result: {tool_result}")

            self.research_context = "\n\n".join(tool_results) if tool_results else "No tools executed."
        else:
            self.research_context = research_response.content if hasattr(research_response, 'content') else str(research_response)

        # Phase 2: Generate proposal based on research
        logger.info(f"[IssueAgent] Phase 2: Generating proposal for '{self.issue_content}'")
        logger.info(f"[IssueAgent] Research context length: {len(self.research_context)} chars")

        proposal_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an IssueAgent tasked with solving a specific puzzle/obstacle in Zork.

YOUR RESPONSIBILITY:
Propose the BEST solution for YOUR specific issue based on available information.

CRITICAL: Be HONEST about your confidence. Your job is NOT to advocate for your issue.
Your job is to provide the BEST idea for solving it. If you don't have a good solution, give a LOW score.

Rules for proposed_action:
- If there's a clear, actionable step to solve YOUR issue, propose it (e.g., "OPEN WINDOW", "TAKE KEY", "GO NORTH")
- If you need more information or have no viable solution right now, respond with "nothing"
- Be specific to YOUR issue only - don't propose actions for other problems

Rules for reason:
- Explain HOW this action will solve YOUR specific issue
- Reference relevant history or evidence from your research
- Be concise (1-2 sentences)
- If action is "nothing", explain why you lack a viable solution

Confidence score (1-100) - BE HONEST:
- 80-100: Strong, actionable solution with clear path to solving YOUR issue
- 50-79: Reasonable approach, worth trying, likely to make progress
- 20-49: Weak idea, uncertain if it will help, better than nothing
- 1-19: No good solution available, just guessing

IMPORTANT: If you don't have a confident solution for YOUR issue, give a LOW score (20 or below).
The Decision Agent will choose the IssueAgent with the MOST CONFIDENT, VIABLE solution.

Respond with structured output."""),
            ("human", """ISSUE YOU ARE SOLVING:
{issue}

CURRENT LOCATION:
{current_location}

CURRENT GAME STATE:
{game_response}

RESEARCH CONTEXT:
{research_context}

What should the adventurer do THIS TURN to make progress on YOUR issue?""")
        ])

        proposal_chain = proposal_prompt | decision_llm.with_structured_output(IssueProposal)

        logger.info(f"[IssueAgent] Calling proposal_chain.invoke()...")
        try:
            proposal = proposal_chain.with_config(
                run_name=f"IssueAgent Proposal: {self.issue_content[:60]}"
            ).invoke({
                "issue": self.issue_content,
                "current_location": current_location,
                "game_response": current_game_response,
                "research_context": self.research_context
            })
            logger.info(f"[IssueAgent] Proposal generated: {proposal.proposed_action} (confidence: {proposal.confidence})")
        except Exception as e:
            logger.error(f"[IssueAgent] Proposal generation failed: {e}")
            raise

        # Store proposal
        self.proposed_action = proposal.proposed_action
        self.reason = proposal.reason
        self.confidence = proposal.confidence

        # Log the proposal
        logger.info(f"IssueAgent[{self.importance}/1000]: '{self.issue_content}'")
        logger.info(f"  > Proposed: '{self.proposed_action}' (confidence: {self.confidence}/100)")
        logger.info(f"  > Reason: {self.reason}")

        return proposal

"""
LoopDetectionAgent - Detects when the AI is stuck in unproductive loops.

This agent analyzes raw game history to identify:
- Stuck in same location without progress
- Oscillating between locations (NORTH→SOUTH→NORTH pattern)

When a loop is detected, proposes a breaking action with high confidence.
When no loop, proposes "nothing" with zero confidence.
"""
from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from .loop_detection_response import LoopDetectionResponse
import logging


class LoopDetectionAgent:
    """
    Agent that detects unproductive loops and proposes breaking actions.

    Always runs in parallel with IssueAgents and ExplorerAgent.
    """

    def __init__(self):
        """Initialize the LoopDetectionAgent"""
        self.proposed_action: str = "nothing"
        self.reason: str = ""
        self.confidence: int = 0
        self.loop_detected: bool = False
        self.loop_type: str = ""

    def research_and_propose(
        self,
        research_agent: Runnable,
        decision_llm: BaseChatModel,
        history_tools: list,
        mapper_tools: list,
        current_location: str,
        current_game_response: str,
        current_score: int,
        current_moves: int
    ):
        """
        Analyze raw history for loops and propose breaking action if needed.

        Args:
            research_agent: LLM chain with tools for calling history
            decision_llm: LLM for generating structured proposal
            history_tools: List of available history tools
            mapper_tools: List of available mapper tools
            current_location: Current game location
            current_game_response: Latest game response text
            current_score: Current game score
            current_moves: Current move count
        """
        logger = logging.getLogger(__name__)

        logger.info(f"[LoopDetectionAgent] Phase 1: Gathering raw history (last 10 turns)")

        # Phase 1: Get raw history (last 10 turns) using history tools
        research_input = {
            "input": "Use get_recent_turns to retrieve the last 10 turns of RAW game history. We need to analyze for stuck/oscillating patterns.",
            "score": current_score,
            "locationName": current_location,
            "moves": current_moves,
            "game_response": current_game_response
        }

        try:
            from config import invoke_with_retry
            research_response = invoke_with_retry(
                research_agent.with_config(
                    run_name="LoopDetectionAgent Research"
                ),
                research_input,
                operation_name="LoopDetectionAgent Research"
            )
        except Exception as e:
            logger.error(f"[LoopDetectionAgent] Research agent failed: {e}")
            self.proposed_action = "nothing"
            self.confidence = 0
            return

        # Execute tool calls to get raw history
        raw_history = ""
        if hasattr(research_response, 'tool_calls') and research_response.tool_calls:
            tools_map = {tool.name: tool for tool in history_tools}

            logger.info(f"[LoopDetectionAgent] Made {len(research_response.tool_calls)} tool calls:")

            for tool_call in research_response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})

                if tool_name == "get_recent_turns" and tool_name in tools_map:
                    # Force last 10 turns
                    tool_args['n'] = 10
                    logger.info(f"  -> {tool_name}(n={tool_args['n']})")
                    tool_result = tools_map[tool_name].invoke(tool_args)
                    logger.info(f"     Result: {str(tool_result)[:150]}...")
                    raw_history = tool_result
                    break

        if not raw_history:
            logger.warning("[LoopDetectionAgent] No raw history retrieved")
            self.proposed_action = "nothing"
            self.confidence = 0
            return

        # Phase 2: Get available exits from mapper
        logger.info(f"[LoopDetectionAgent] Phase 2: Getting available exits from mapper")

        available_exits = []
        tools_map = {tool.name: tool for tool in mapper_tools}
        if "get_exits_from" in tools_map:
            try:
                logger.info(f"  -> get_exits_from(location='{current_location}')")
                exits_result = tools_map["get_exits_from"].invoke({"location": current_location})
                logger.info(f"     Result: {str(exits_result)[:150]}...")
                # Parse exits (format: "NORTH → Kitchen, SOUTH → Hallway")
                if exits_result and exits_result.strip():
                    for line in exits_result.split(','):
                        parts = line.strip().split('→')
                        if len(parts) >= 1:
                            direction = parts[0].strip()
                            available_exits.append(direction)
            except Exception as e:
                logger.warning(f"[LoopDetectionAgent] Failed to get exits: {e}")

        # Phase 3: Analyze for loops using LLM
        logger.info(f"[LoopDetectionAgent] Phase 3: Analyzing for loop patterns")

        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the LoopDetectionAgent in a Zork-playing AI system.

YOUR RESPONSIBILITY:
Analyze raw game history to detect unproductive loops and propose actions to break them.

LOOP TYPES TO DETECT:

1. **Stuck in Location** (loop_type: "stuck_location")
   - Same location for 5+ turns
   - No score increase during this time
   - Agent is repeatedly trying similar failed actions (e.g., GO EAST fails, GO WEST fails)

2. **Oscillating Between Locations** (loop_type: "oscillating")
   - Moving back and forth between 2-3 locations
   - Example: NORTH → SOUTH → NORTH → SOUTH
   - Example: Kitchen → Hallway → Kitchen → Hallway
   - No meaningful progress being made

WHEN LOOP DETECTED:
- Set loop_detected = true
- Set appropriate loop_type
- Propose a RADICALLY DIFFERENT action to break the loop:
  * If location description mentions interactive verbs (climbable, openable, takeable), try that verb!
    Example: "cliff appears climbable" → propose "CLIMB CLIFF" or "CLIMB UP"
    Example: "door can be opened" → propose "OPEN DOOR"
  * Try an unexplored exit from available_exits
  * Try examining objects mentioned in description (EXAMINE item)
  * Try different action categories: if moving failed → interact with objects, if attacking → examine
  * Try INVENTORY to check what you have
- Confidence: 95-100 (very high - loops are bad, must break them!)
- Reason: Explain the loop pattern clearly and why this action breaks it

WHEN NO LOOP DETECTED:
- Set loop_detected = false
- Set loop_type = ""
- Set proposed_action = "nothing"
- Confidence: 0
- Reason: "No loop pattern detected in recent history"

========================================================
CRITICAL: BE VERY AGGRESSIVE
========================================================

Loops waste turns and prevent progress. Detect them early!

SIGNS OF LOOPS (detect ANY of these):
+ Same location 5+ turns with no score change
+ Trying different movement commands that all fail
+ Bouncing between 2 locations repeatedly
+ Doing similar actions that don't change the situation

If you see ANY loop pattern, SET loop_detected=true and confidence=95-100.
Don't wait for 5+ turns - catch loops early at turn 5!

Respond with structured output."""),
            ("human", """CURRENT LOCATION: {current_location}
CURRENT SCORE: {current_score}

AVAILABLE EXITS (from mapper):
{available_exits}

RAW GAME HISTORY (Last 10 Turns):
{raw_history}

Analyze for loops and propose breaking action if needed:""")
        ])

        analysis_chain = analysis_prompt | decision_llm.with_structured_output(LoopDetectionResponse)

        try:
            from config import invoke_with_retry
            response = invoke_with_retry(
                analysis_chain.with_config(
                    run_name="LoopDetectionAgent Analysis"
                ),
                {
                    "current_location": current_location,
                    "current_score": current_score,
                    "available_exits": ", ".join(available_exits) if available_exits else "None available",
                    "raw_history": raw_history[:3000]  # Truncate if too long
                },
                operation_name="LoopDetectionAgent Analysis"
            )

            # Store results
            self.loop_detected = response.loop_detected
            self.loop_type = response.loop_type
            self.proposed_action = response.proposed_action
            self.reason = response.reason
            self.confidence = response.confidence

            if self.loop_detected:
                logger.info(f"[LoopDetectionAgent] LOOP DETECTED ({self.loop_type})")
                logger.info(f"  > Proposed: '{self.proposed_action}' (confidence: {self.confidence}/100)")
                logger.info(f"  > Reason: {self.reason}")
            else:
                logger.info(f"[LoopDetectionAgent] No loop detected")

        except Exception as e:
            logger.error(f"[LoopDetectionAgent] Analysis failed: {e}")
            self.proposed_action = "nothing"
            self.confidence = 0

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
            research_response = research_agent.with_config(
                run_name="LoopDetectionAgent Research"
            ).invoke(research_input)
        except Exception as e:
            logger.error(f"[LoopDetectionAgent] Research agent failed: {e}")
            self.proposed_action = "nothing"
            self.confidence = 0
            return

        # Execute tool calls to get raw history
        raw_history = ""
        if hasattr(research_response, 'tool_calls') and research_response.tool_calls:
            tools_map = {tool.name: tool for tool in history_tools}

            for tool_call in research_response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})

                if tool_name == "get_recent_turns" and tool_name in tools_map:
                    # Force last 10 turns
                    tool_args['n'] = 10
                    tool_result = tools_map[tool_name].invoke(tool_args)
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
                exits_result = tools_map["get_exits_from"].invoke({"location": current_location})
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
   - Agent is repeatedly trying similar actions without progress

2. **Oscillating Between Locations** (loop_type: "oscillating")
   - Moving back and forth between 2-3 locations
   - Example: NORTH → SOUTH → NORTH → SOUTH
   - Example: Kitchen → Hallway → Kitchen → Hallway
   - No meaningful progress being made

WHEN LOOP DETECTED:
- Set loop_detected = true
- Set appropriate loop_type
- Propose a RADICALLY DIFFERENT action to break the loop:
  * Try an unexplored exit from available_exits
  * Try opposite/different direction than recent pattern
  * Try completely different action type (if attacking → examine; if moving → interact)
  * Try examining surroundings carefully (LOOK, INVENTORY, EXAMINE item)
- Confidence: 90-100 (high confidence that loop exists)
- Reason: Explain the loop pattern and why this action breaks it

WHEN NO LOOP DETECTED:
- Set loop_detected = false
- Set loop_type = ""
- Set proposed_action = "nothing"
- Confidence: 0
- Reason: "No loop pattern detected in recent history"

CRITICAL: Be aggressive in detecting loops. If there's repetitive behavior without progress, flag it.

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
            response = analysis_chain.with_config(
                run_name="LoopDetectionAgent Analysis"
            ).invoke({
                "current_location": current_location,
                "current_score": current_score,
                "available_exits": ", ".join(available_exits) if available_exits else "None available",
                "raw_history": raw_history[:3000]  # Truncate if too long
            })

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

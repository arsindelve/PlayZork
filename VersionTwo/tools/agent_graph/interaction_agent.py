"""
InteractionAgent - Identifies and proposes interactions with objects in current location.

This agent analyzes the current game response to detect interactive objects like:
- Takeable items (TAKE LAMP)
- Containers (OPEN DOOR, UNLOCK CHEST)
- Interactive objects (PRESS BUTTON, PULL LEVER)
- Readable items (READ NOTE, EXAMINE SIGN)
- Inventory combinations (UNLOCK DOOR WITH KEY)

Always runs in parallel with other agents.
"""
from typing import List, Optional, Dict
import re
import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from .interaction_response import InteractionResponse
from config import GAME_NAME


class InteractionAgent:
    """
    Agent that identifies and proposes interactions with objects in current location.

    Runs every turn to detect:
    - Takeable items
    - Openable/closeable containers
    - Interactive objects (buttons, levers, dials)
    - Readable items
    - Inventory items that can be used in environment

    Always runs in parallel with other agents.
    """

    def __init__(self):
        """Initialize the InteractionAgent"""
        self.proposed_action: str = "nothing"
        self.reason: str = ""
        self.confidence: int = 0
        self.detected_objects: List[str] = []
        self.inventory_items: List[str] = []
        self.current_location: str = ""

        # Tool call history (for reporting)
        self.tool_calls_history: list = []

    def research_and_propose(
        self,
        research_agent: Runnable,
        decision_llm: BaseChatModel,
        history_tools: list,
        mapper_tools: list,
        current_location: str,
        current_game_response: str,
        current_score: int,
        current_moves: int,
        inventory_tools: list = None  # Optional for backward compatibility
    ):
        """
        Analyze current location for interactive objects and propose action.

        Phases:
        1. Get current inventory (via research_agent with tools)
        2. Parse game response for interactive objects (deterministic)
        3. If unclear, use LLM to analyze interactions
        4. Generate proposal with confidence

        Args:
            research_agent: LLM chain with tools for calling inventory
            decision_llm: LLM for generating structured proposals
            history_tools: Available history tools
            mapper_tools: Available mapper tools
            current_location: Current game location
            current_game_response: Latest game response text
            current_score: Current game score
            current_moves: Current move count
            inventory_tools: Inventory toolkit tools (optional)
        """
        logger = logging.getLogger(__name__)

        # Store current location for reporting
        self.current_location = current_location

        logger.info(f"[InteractionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[InteractionAgent] AGENT: InteractionAgent")
        logger.info(f"[InteractionAgent] PURPOSE: Identify and propose interactions with local objects")
        logger.info(f"[InteractionAgent] CURRENT LOCATION: {current_location}")
        logger.info(f"[InteractionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[InteractionAgent] Phase 1: Getting current inventory")

        # Phase 1: Get inventory using research agent with tools
        inventory_list: List[str] = []

        if inventory_tools:
            # Combine all available tools
            all_tools = history_tools + mapper_tools + inventory_tools

            research_input = {
                "input": "Use get_inventory to list all items currently in inventory.",
                "score": current_score,
                "locationName": current_location,
                "moves": current_moves,
                "game_response": current_game_response
            }

            try:
                from llm_utils import invoke_with_retry
                research_response = invoke_with_retry(
                    research_agent.with_config(
                        run_name="InteractionAgent Inventory Check",
                        configurable={"tools": all_tools}
                    ),
                    research_input,
                    operation_name="InteractionAgent Inventory Check"
                )

                # Execute any tool calls to get inventory
                if hasattr(research_response, 'tool_calls') and research_response.tool_calls:
                    tools_map = {tool.name: tool for tool in all_tools}

                    logger.info(f"[InteractionAgent] Made {len(research_response.tool_calls)} tool calls:")

                    for tool_call in research_response.tool_calls:
                        tool_name = tool_call['name']
                        tool_args = tool_call.get('args', {})

                        if tool_name == "get_inventory" and tool_name in tools_map:
                            logger.info(f"[InteractionAgent]   -> {tool_name}()")
                            tool_result = tools_map[tool_name].invoke(tool_args)
                            logger.info(f"[InteractionAgent]      Result: {str(tool_result)}")

                            # Store tool call history for reporting
                            self.tool_calls_history.append({
                                "tool_name": tool_name,
                                "input": str(tool_args),
                                "output": str(tool_result)
                            })

                            # Parse inventory (comma-separated list or "empty" message)
                            if tool_result and "empty" not in tool_result.lower():
                                inventory_list = [item.strip() for item in tool_result.split(',')]
                            break
            except Exception as e:
                logger.warning(f"[InteractionAgent] Failed to get inventory: {e}")
                # Continue without inventory

        logger.info(f"[InteractionAgent] Inventory: {inventory_list if inventory_list else 'empty'}")

        # Phase 2: Deterministic parsing for common interactions
        logger.info(f"[InteractionAgent] Phase 2: Running deterministic parsing")

        deterministic_result = self._deterministic_parse(current_game_response, inventory_list)

        if deterministic_result:
            # Found a clear interaction deterministically!
            logger.info(f"[InteractionAgent] ⚡ DETERMINISTIC MATCH: {deterministic_result['action']}")

            self.proposed_action = deterministic_result['action']
            self.reason = deterministic_result['reason']
            self.confidence = deterministic_result['confidence']
            self.detected_objects = deterministic_result.get('objects', [])
            self.inventory_items = deterministic_result.get('items_used', [])

            # Log proposal summary
            logger.info(f"[InteractionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"[InteractionAgent] PROPOSAL SUMMARY - DETERMINISTIC")
            logger.info(f"[InteractionAgent] Proposed Action: '{self.proposed_action}' (confidence: {self.confidence}/100)")
            logger.info(f"[InteractionAgent] Detected Objects: {self.detected_objects}")
            logger.info(f"[InteractionAgent] Reason: {self.reason}")
            logger.info(f"[InteractionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # Skip LLM - we have a clear match
            return

        logger.info(f"[InteractionAgent] No deterministic match, proceeding to LLM analysis")

        # Phase 3: LLM analysis for complex interactions
        logger.info(f"[InteractionAgent] Phase 3: Analyzing interactions with LLM")

        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are the InteractionAgent in a {GAME_NAME}-playing AI system.

YOUR RESPONSIBILITY:
Identify and propose interactions with objects in the current location.

INTERACTION TYPES:

1. **Take Items**
   - Look for: "There is X here", "You see X", "X sits/lies here"
   - Action: TAKE [item]
   - Confidence: 85-95 (very likely to succeed)

2. **Open/Close Containers**
   - Look for: doors, boxes, chests, mailboxes (especially if "closed" or "locked")
   - Action: OPEN [container], CLOSE [container]
   - If locked + have key: UNLOCK [container] WITH KEY
   - Confidence: 80-90 if openable, 60-70 if locked without key

3. **Use Interactive Objects**
   - Look for: buttons, levers, dials, switches, knobs
   - Action: PRESS BUTTON, PULL LEVER, TURN DIAL, etc.
   - Confidence: 70-85 (might trigger puzzles)

4. **Read/Examine**
   - Look for: papers, notes, books, signs, inscriptions
   - Action: READ [item], EXAMINE [item]
   - Confidence: 60-75 (informational, not always critical)

5. **Combine Inventory with Environment**
   - Check inventory for items that might interact with location
   - Examples: key+door, torch+darkness, rope+pit
   - Action: USE [item] ON [object], UNLOCK [door] WITH [key]
   - Confidence: 80-95 if clear match

CONFIDENCE SCORING:
- 90-100: Clear, unambiguous interaction (TAKE visible item)
- 70-89: Likely useful interaction (OPEN closed door, PRESS button)
- 50-69: Possible interaction (EXAMINE unusual object)
- 20-49: Speculative interaction (try random commands)
- 0: No interactions available

WHEN NO INTERACTIONS:
- Set proposed_action = "nothing"
- Set confidence = 0
- Reason: "No interactive objects detected in current location"

CRITICAL RULES:
- Prioritize TAKING items over exploring (items might be needed for puzzles)
- Don't propose movement commands (that's ExplorerAgent's job)
- Don't try to solve tracked issues (that's IssueAgent's job)
- Focus ONLY on interacting with objects mentioned in current location
- ALWAYS check inventory first - many interactions require items

CRITICAL COMMAND RULES:
- NEVER use semicolons (;) in your proposed action
- NEVER combine multiple commands - propose ONE simple command only
- Use the SIMPLEST possible version of each command
- Examples: 'TAKE LAMP', 'OPEN DOOR', 'PRESS BUTTON'
- NOT allowed: 'OPEN KIT; TAKE ROPE', 'TAKE LAMP AND EXAMINE IT'

Respond with structured output."""),
            ("human", """CURRENT LOCATION: {current_location}
CURRENT SCORE: {current_score}

INVENTORY:
{inventory}

CURRENT GAME RESPONSE:
{game_response}

Analyze the game response for interactive objects and propose the best interaction.""")
        ])

        analysis_chain = analysis_prompt | decision_llm.with_structured_output(InteractionResponse)

        try:
            from llm_utils import invoke_with_retry
            response = invoke_with_retry(
                analysis_chain.with_config(
                    run_name="InteractionAgent LLM Analysis"
                ),
                {
                    "current_location": current_location,
                    "current_score": current_score,
                    "inventory": ", ".join(inventory_list) if inventory_list else "Your inventory is empty.",
                    "game_response": current_game_response[:1000]  # Truncate if too long
                },
                operation_name="InteractionAgent LLM Analysis"
            )

            # Store results
            self.proposed_action = response.proposed_action
            self.reason = response.reason
            self.confidence = response.confidence
            self.detected_objects = response.detected_objects or []
            self.inventory_items = response.inventory_items or []

            # Log proposal summary
            logger.info(f"[InteractionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"[InteractionAgent] PROPOSAL SUMMARY - LLM")
            logger.info(f"[InteractionAgent] Proposed Action: '{self.proposed_action}' (confidence: {self.confidence}/100)")
            logger.info(f"[InteractionAgent] Detected Objects: {self.detected_objects}")
            if self.reason:
                logger.info(f"[InteractionAgent] Reason: {self.reason}")
            logger.info(f"[InteractionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        except Exception as e:
            logger.error(f"[InteractionAgent] LLM analysis failed: {e}")
            # Set safe defaults
            self.proposed_action = "nothing"
            self.confidence = 0
            self.reason = f"Analysis failed: {str(e)}"
            raise

    def _deterministic_parse(self, game_response: str, inventory: List[str]) -> Optional[Dict]:
        """
        Fast pattern matching for common interaction types.
        Returns proposal dict if clear match, None if ambiguous.

        Args:
            game_response: Current game response text
            inventory: List of items in inventory

        Returns:
            Dict with 'action', 'reason', 'confidence', 'objects', 'items_used' if match found, else None
        """
        text = game_response.lower()

        # Pattern 1: Takeable items
        takeable_patterns = [
            r"there (?:is|are) (?:a |an )?(\w+)(?: and (?:a |an )?(\w+))? here",
            r"you (?:see|notice) (?:a |an )?(\w+)",
            r"(?:a |an )?(\w+) (?:sits|lies|rests) (?:here|on the \w+)"
        ]

        for pattern in takeable_patterns:
            match = re.search(pattern, text)
            if match:
                item = match.group(1).upper()
                # Skip if it's a location descriptor (common false positives)
                if item.lower() not in ['door', 'room', 'hallway', 'corridor', 'wall', 'floor', 'ceiling']:
                    return {
                        'action': f'TAKE {item}',
                        'reason': f'Found takeable item: {item}',
                        'confidence': 90,
                        'objects': [item]
                    }

        # Pattern 2: Closed containers
        if 'closed' in text:
            container_match = re.search(r'(\w+) (?:is |are )?closed', text)
            if container_match:
                container = container_match.group(1).upper()
                return {
                    'action': f'OPEN {container}',
                    'reason': f'Found closed container: {container}',
                    'confidence': 85,
                    'objects': [container]
                }

        # Pattern 3: Locked objects (check if we have key)
        if 'locked' in text:
            locked_match = re.search(r'(\w+) (?:is |are )?locked', text)
            if locked_match:
                obj = locked_match.group(1).upper()

                # Check inventory for key
                if any('key' in item.lower() for item in inventory):
                    return {
                        'action': f'UNLOCK {obj} WITH KEY',
                        'reason': f'Found locked {obj} and have key in inventory',
                        'confidence': 95,
                        'objects': [obj],
                        'items_used': ['KEY']
                    }
                else:
                    return {
                        'action': f'EXAMINE {obj}',
                        'reason': f'Found locked {obj} but no key yet',
                        'confidence': 60,
                        'objects': [obj]
                    }

        # Pattern 4: Interactive objects
        interactive_keywords = ['button', 'lever', 'dial', 'switch', 'knob']
        for keyword in interactive_keywords:
            if keyword in text:
                action_verb = {
                    'button': 'PRESS',
                    'lever': 'PULL',
                    'dial': 'TURN',
                    'switch': 'FLIP',
                    'knob': 'TURN'
                }.get(keyword, 'EXAMINE')

                return {
                    'action': f'{action_verb} {keyword.upper()}',
                    'reason': f'Found interactive object: {keyword}',
                    'confidence': 80,
                    'objects': [keyword.upper()]
                }

        # No clear pattern found
        return None

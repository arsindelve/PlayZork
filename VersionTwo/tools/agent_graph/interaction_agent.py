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
from .tool_execution import invoke_tool_safely, TOOL_ERROR_PREFIX
from adventurer.prompt_library import PromptLibrary


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

    async def propose(
        self,
        decision_llm: BaseChatModel,
        context,
    ) -> None:
        """Propose an interaction with something in this room.

        Phase 1 used to be a whole LLM round-trip asking the model to call
        get_inventory — 106s on the measured turn, to retrieve a list the code
        already had (#25). TurnContext supplies it, and on the hosted backends
        it comes from the game itself (#30).
        """
        logger = logging.getLogger(__name__)
        # Function-local import: tests monkeypatch llm_utils.ainvoke_with_retry
        from llm_utils import ainvoke_with_retry

        current_location = context.location
        current_game_response = context.game_text
        inventory_list = list(context.inventory)

        logger.info(f"[InteractionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[InteractionAgent] CURRENT LOCATION: {current_location}")
        logger.info(f"[InteractionAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

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
            ("system", PromptLibrary.get_interaction_agent_system_prompt()),
            ("human", PromptLibrary.get_interaction_agent_human_prompt())
        ])

        analysis_chain = analysis_prompt | decision_llm.with_structured_output(InteractionResponse)

        response = await ainvoke_with_retry(
            analysis_chain.with_config(
                run_name="InteractionAgent LLM Analysis"
            ),
            {
                "current_location": current_location,
                "current_score": context.score,
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

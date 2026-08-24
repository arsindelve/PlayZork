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

        # The game itself reports which commands it will accept here (#30).
        # When it does, that is ground truth and the regex guesswork below is
        # not needed at all.
        available = getattr(context, "available_actions", {}) or {}
        logger.info(f"[InteractionAgent] Game-reported objects: {list(available) or 'none'}")

        # The deterministic parser is a HINT ONLY and must never short-circuit
        # the LLM (#16). It used to return early on a match, which meant the
        # LLM never ran on precisely the turns the regexes misfired — and they
        # misfired on Zork's most common reply. "You see nothing special about
        # the mailbox." produced `TAKE NOTHING` at confidence 90, above almost
        # every real proposal on the arbiter's list.
        hint = self._deterministic_parse(current_game_response, inventory_list)
        if hint:
            logger.info(f"[InteractionAgent] Parser hint (not authoritative): {hint['action']}")

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
                "game_response": current_game_response[:1000],  # Truncate if too long
                "available_actions": context.available_actions_summary,
                "already_tried": context.unproductive_summary,
                "parser_hint": hint["action"] if hint else "none",
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

        # A negated sentence describes what is NOT here. Proposing an action on
        # it is guaranteed to fail: "You cannot see any button here." used to
        # yield PRESS BUTTON at confidence 80 (#16).
        if re.search(r"\b(?:cannot|can't|couldn't|don't|doesn't|isn't|aren't|"
                     r"no longer|not\s+here|nothing|anything)\b", text):
            return None

        # Pattern 1: Takeable items.
        # Capture the FULL noun phrase and keep its head noun: the old patterns
        # took the first word after the article, so "a brass lantern" became
        # TAKE BRASS and "a red button" became TAKE RED (#16).
        # Stop the noun phrase at the first delimiter, so "a brass lantern and
        # a sword here" yields "brass lantern" rather than running on.
        # NOTE both quantifiers are lazy. A greedy `{0,3}` swallows the words
        # before the lazy tail gets a chance, so "a small mailbox here." was
        # captured as "small mailbox here" and proposed TAKE HERE.
        NOUN = r"((?:\w+ ){0,3}?\w+?)(?=\s+(?:and|here|on|in|is|are)\b|[.,])"
        takeable_patterns = [
            rf"there (?:is|are) (?:a |an |the )?{NOUN}",
            rf"you (?:see|notice) (?:a |an |the )?{NOUN}",
            rf"(?:a |an |the )?{NOUN}\s+(?:sits|lies|rests)\b",
        ]

        for pattern in takeable_patterns:
            match = re.search(pattern, text)
            if match:
                phrase = (match.group(1) or "").strip()
                # The head noun is the last word of the phrase.
                item = phrase.split()[-1].upper() if phrase else ""
                # Skip if it's a location descriptor (common false positives)
                # Head nouns that are never a takeable object. Belt and braces
                # behind the negation guard above: a bad head noun produces a
                # command the game cannot parse, and this agent's proposals
                # reach the arbiter at high confidence (#16).
                NOT_OBJECTS = {
                    'door', 'room', 'hallway', 'corridor', 'wall', 'floor', 'ceiling',
                    'special', 'unusual', 'here', 'there', 'it', 'you', 'nothing',
                    'anything', 'something', 'this', 'that', 'them',
                }
                if item and item.lower() not in NOT_OBJECTS:
                    return {
                        'action': f'TAKE {item}',
                        'reason': f'Found takeable item: {item}',
                        'confidence': 90,
                        'objects': [item]
                    }

        # Pattern 2: Closed containers
        if 'closed' in text:
            # Copula is MANDATORY and the subject may not be "you": without it,
            # "You closed the wooden door." — the game confirming your OWN
            # command — produced OPEN YOU at confidence 85 (#16).
            container_match = re.search(r'\b(?!you\b)(\w+) (?:is|are) closed', text)
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
            locked_match = re.search(r'\b(?!you\b)(\w+) (?:is|are) locked', text)
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

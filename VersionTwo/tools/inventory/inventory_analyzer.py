"""InventoryAnalyzer - LLM-based analyzer for detecting inventory changes"""
from typing import List
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from config import GAME_NAME
import logging


class InventoryChange(BaseModel):
    """Structured output for inventory changes"""
    items_added: List[str] = []      # Items added to inventory
    items_removed: List[str] = []    # Items removed from inventory
    reasoning: str = ""              # Why these changes occurred


class InventoryAnalyzer:
    """LLM-based analyzer for detecting inventory changes"""

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the inventory analyzer.

        Args:
            llm: Language model for analysis (should be cheap model like GPT-3.5)
        """
        self.llm = llm
        self.logger = logging.getLogger(__name__)

        # Create prompt for inventory analysis
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You analyze {GAME_NAME} game turns to detect inventory changes.

Your job is to determine what items were ADDED TO or REMOVED FROM the player's inventory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES FOR INVENTORY TRACKING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. TAKING ITEMS (ADD to inventory):
   ✓ "TAKE LAMP" + "Taken." → ADD "lamp"
   ✓ "GET SWORD" + "Taken." → ADD "sword"
   ✓ "OPEN MAILBOX" + "Opening the mailbox reveals a leaflet which you take." → ADD "leaflet"
   ✓ "TAKE ALL" + "lamp: Taken. sword: Taken." → ADD "lamp", "sword"
   ✓ "GET SWORD FROM CHEST" + "Taken." → ADD "sword"
   ✓ "EXAMINE CHEST" + "Inside the chest is a brass key which you take." → ADD "brass key"

2. DROPPING ITEMS (REMOVE from inventory):
   ✓ "DROP LAMP" + "Dropped." → REMOVE "lamp"
   ✓ "PUT SWORD IN CHEST" + "Done." → REMOVE "sword" (putting in container = dropping)
   ✓ "PLACE LEAFLET IN MAILBOX" + "Done." → REMOVE "leaflet"
   ✓ "GIVE LAMP TO TROLL" + "The troll takes the lamp." → REMOVE "lamp" (giving = dropping)
   ✓ "INSERT COIN IN SLOT" + "Done." → REMOVE "coin"

3. CONTAINERS - CRITICAL UNDERSTANDING:
   - Putting something IN a container = REMOVE from inventory
   - Taking something FROM a container = ADD to inventory
   - If an item goes into a bag, chest, mailbox, pocket, etc. → REMOVE it
   - If an item comes out of a container → ADD it

4. FAILED ACTIONS (NO CHANGE):
   ✗ "TAKE LAMP" + "You can't see any lamp here." → NO CHANGE
   ✗ "TAKE LAMP" + "You're already carrying the lamp." → NO CHANGE
   ✗ "DROP SWORD" + "You aren't carrying a sword." → NO CHANGE
   ✗ "TAKE LAMP" + "You're carrying too much." → NO CHANGE
   ✗ "TAKE LAMP" + "The lamp is too heavy." → NO CHANGE

5. IMPLICIT CHANGES:
   ✓ "The troll takes your lamp and runs away." → REMOVE "lamp"
   ✓ "You find a sword on the ground and pick it up." → ADD "sword"
   ✓ "The wizard gives you a magic ring." → ADD "magic ring"
   ✓ "Your torch burns out and crumbles to dust." → REMOVE "torch"

6. ITEM NAME EXTRACTION:
   - Use EXACT item names from the game text
   - DO NOT normalize or change names
   - "brass lantern" → use "brass lantern" (not "lantern")
   - "rusty key" → use "rusty key" (not "key")
   - Keep adjectives and full descriptions as game provides them

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analyze the turn carefully and return:
- items_added: List of items ADDED to inventory (empty list if none)
- items_removed: List of items REMOVED from inventory (empty list if none)
- reasoning: Brief explanation of what happened

If nothing changed, return empty lists for both.

Respond with structured JSON output."""),
            ("human", """PLAYER COMMAND: {player_command}

GAME RESPONSE: {game_response}

What items were added to or removed from inventory this turn?""")
        ])

        # Create chain with structured output
        self.chain = self.prompt | self.llm.with_structured_output(InventoryChange)

    def analyze_turn(self, player_command: str, game_response: str) -> InventoryChange:
        """
        Analyze a turn and return inventory changes.

        Args:
            player_command: Command issued by the player
            game_response: Game's response to the command

        Returns:
            InventoryChange object with items added/removed and reasoning
        """
        try:
            result = self.chain.invoke({
                "player_command": player_command,
                "game_response": game_response
            })

            self.logger.info(f"[InventoryAnalyzer] Command: {player_command}")
            self.logger.info(f"[InventoryAnalyzer] Items added: {result.items_added}")
            self.logger.info(f"[InventoryAnalyzer] Items removed: {result.items_removed}")
            self.logger.info(f"[InventoryAnalyzer] Reasoning: {result.reasoning}")

            return result

        except Exception as e:
            self.logger.error(f"[InventoryAnalyzer] Analysis failed: {e}")
            # Return empty change on error
            return InventoryChange(
                items_added=[],
                items_removed=[],
                reasoning=f"Analysis failed: {str(e)}"
            )

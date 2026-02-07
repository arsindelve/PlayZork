"""InventoryAnalyzer - LLM-based analyzer for detecting inventory changes"""
from typing import List
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from adventurer.prompt_library import PromptLibrary
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
            ("system", PromptLibrary.get_inventory_analyzer_system_prompt()),
            ("human", PromptLibrary.get_inventory_analyzer_human_prompt())
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

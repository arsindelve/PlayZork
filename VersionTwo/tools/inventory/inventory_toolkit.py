"""InventoryToolkit - Provides tools for agents to query inventory"""
from tools.database import DatabaseManager
from .inventory_state import InventoryState
from langchain_core.tools import tool
from typing import List


class InventoryToolkit:
    """Provides LangChain tools for querying inventory"""

    def __init__(self, session_id: str, db: DatabaseManager):
        """
        Initialize the inventory toolkit.

        Args:
            session_id: Game session ID
            db: DatabaseManager instance
        """
        self.state = InventoryState(session_id, db)

    def get_tools(self) -> List:
        """
        Return LangChain tools for agents to use.

        Returns:
            List of LangChain tool functions
        """
        # Capture state in closure
        state = self.state

        @tool
        def get_inventory() -> str:
            """
            Get the list of items currently in the adventurer's inventory.
            Returns a simple comma-separated list of item names.
            Use this to check what items are available for solving puzzles.
            """
            items = state.get_items()
            if not items:
                return "Your inventory is empty."
            return ", ".join(items)

        return [get_inventory]

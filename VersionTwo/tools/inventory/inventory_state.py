"""InventoryState - Manages the player's inventory with database persistence"""
from typing import List
from tools.database import DatabaseManager
import logging


class InventoryState:
    """Manages the list of items in the player's inventory"""

    def __init__(self, session_id: str, db: DatabaseManager):
        """
        Initialize inventory state with database backend.

        Args:
            session_id: Unique identifier for this game session
            db: DatabaseManager instance for persistence
        """
        self.session_id = session_id
        self.db = db
        self.items: List[str] = []  # Cache for current items
        self.logger = logging.getLogger(__name__)

        # Load current inventory from database
        self._load_from_db()

    def _load_from_db(self):
        """Load current inventory from database"""
        self.items = self.db.get_current_inventory(self.session_id)
        self.logger.info(f"[InventoryState] Loaded {len(self.items)} items from database: {self.items}")

    def get_items(self) -> List[str]:
        """
        Return simple list of item names.

        Returns:
            List of item names currently in inventory
        """
        return self.items.copy()

    def add_item(self, item_name: str, turn_number: int):
        """
        Add item to inventory (with DB persistence).

        Args:
            item_name: Name of the item to add
            turn_number: Turn when item was acquired
        """
        if not item_name or not item_name.strip():
            return

        item_name = item_name.strip()

        # Add to database
        self.db.add_inventory_item(self.session_id, item_name, turn_number)

        # Update cache
        self.items.append(item_name)

        self.logger.info(f"[InventoryState] Added '{item_name}' to inventory (turn {turn_number})")

    def remove_item(self, item_name: str, turn_number: int) -> bool:
        """
        Remove item from inventory (mark as dropped in DB).

        Args:
            item_name: Name of the item to remove
            turn_number: Turn when item was dropped

        Returns:
            True if item was found and removed, False otherwise
        """
        if not item_name or not item_name.strip():
            return False

        item_name = item_name.strip()

        # Remove from database
        success = self.db.remove_inventory_item(self.session_id, item_name, turn_number)

        if success:
            # Update cache
            if item_name in self.items:
                self.items.remove(item_name)
                self.logger.info(f"[InventoryState] Removed '{item_name}' from inventory (turn {turn_number})")
                return True
            else:
                self.logger.warning(f"[InventoryState] Item '{item_name}' removed from DB but not found in cache")
                # Reload from DB to sync
                self._load_from_db()
                return True
        else:
            self.logger.warning(f"[InventoryState] Failed to remove '{item_name}' - not in inventory")
            return False

    def sync_with_game(self, game_inventory: List[str], turn_number: int):
        """
        Sync our tracking with actual game INVENTORY output.

        This is called during bootstrap to ensure our tracking matches game state.

        Args:
            game_inventory: List of items from game's INVENTORY command
            turn_number: Current turn number (usually 0 for bootstrap)
        """
        self.logger.info(f"[InventoryState] Syncing with game inventory: {game_inventory}")

        # Get current tracked items
        current_items = set(self.items)
        game_items = set(game_inventory)

        # Items in game but not tracked = need to add
        items_to_add = game_items - current_items
        for item in items_to_add:
            self.logger.info(f"[InventoryState] Sync: Adding missing item '{item}'")
            self.add_item(item, turn_number)

        # Items tracked but not in game = need to remove
        items_to_remove = current_items - game_items
        for item in items_to_remove:
            self.logger.info(f"[InventoryState] Sync: Removing extra item '{item}'")
            self.remove_item(item, turn_number)

        self.logger.info(f"[InventoryState] Sync complete. Current inventory: {self.items}")

"""InventoryState - Manages the player's inventory with database persistence"""
import re
from typing import List, Optional, Tuple
from tools.database import DatabaseManager
import logging


_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def normalize_item_name(name: str) -> str:
    """
    Canonical form used for *matching only* (never for display or storage).

    Lowercases, strips punctuation, collapses whitespace, and drops a leading
    article. The game's INVENTORY listing says "A brass lantern" while the
    analyzer extracts "brass lantern"; both normalize to "brass lantern".
    """
    s = _PUNCT_RE.sub(" ", (name or "").lower())
    s = _WS_RE.sub(" ", s).strip()
    return _ARTICLE_RE.sub("", s).strip()


def _tokens(name: str) -> set:
    return set(normalize_item_name(name).split())


class InventoryState:
    """Manages the list of items in the player's inventory"""

    def __init__(self, session_id: str, db: DatabaseManager):
        self.session_id = session_id
        self.db = db
        self.items: List[str] = []  # Read-only mirror of the DB; never mutated directly
        self.logger = logging.getLogger(__name__)
        self._load_from_db()

    def _load_from_db(self):
        """Refresh the mirror from the database (the single source of truth)."""
        self.items = self.db.get_current_inventory(self.session_id)

    def get_items(self) -> List[str]:
        """Return the item names currently held, as recorded in the database."""
        return self.items.copy()

    def _open_rows(self) -> List[Tuple[int, str]]:
        return self.db.get_open_inventory_rows(self.session_id)

    def _find_row(self, item_name: str, rows: List[Tuple[int, str]]) -> Tuple[Optional[int], str]:
        """
        Resolve a loosely-named item against the rows currently held.

        Deliberately conservative: it resolves case, punctuation, article and
        head-noun drift ("lantern" -> "brass lantern") but never synonyms
        ("lamp" -> "brass lantern"). A request matching two held items is
        refused rather than guessed, because removing an item the player still
        carries is worse than leaving a stale one: the agents simply stop
        considering it, with nothing in the game text to correct them.

        Returns (row_id, reason); row_id is None when nothing was resolved.
        """
        wanted = normalize_item_name(item_name)
        if not wanted:
            return None, "empty name"

        exact = [rid for rid, raw in rows if normalize_item_name(raw) == wanted]
        if exact:
            # Identical duplicates are interchangeable: drop the oldest.
            return exact[0], "exact"

        wanted_tokens = _tokens(item_name)
        if wanted_tokens:
            partial = [
                rid for rid, raw in rows
                if wanted_tokens <= _tokens(raw) or _tokens(raw) <= wanted_tokens
            ]
            if len(partial) == 1:
                return partial[0], "partial"
            if len(partial) > 1:
                return None, f"ambiguous ({len(partial)} held items match)"

        return None, "not held"

    def add_item(self, item_name: str, turn_number: int):
        """Record an item as held, unless an equivalent item is already held."""
        if not item_name or not item_name.strip():
            return

        item_name = item_name.strip()
        rows = self._open_rows()

        wanted = normalize_item_name(item_name)
        if any(normalize_item_name(raw) == wanted for _, raw in rows):
            self.logger.info(
                f"[InventoryState] Skipped duplicate add of '{item_name}' (already held)"
            )
            return

        self.db.add_inventory_item(self.session_id, item_name, turn_number)
        self._load_from_db()
        self.logger.info(f"[InventoryState] Added '{item_name}' to inventory (turn {turn_number})")

    def remove_item(self, item_name: str, turn_number: int) -> bool:
        """Mark one held item as dropped. Returns True if a row was dropped."""
        if not item_name or not item_name.strip():
            return False

        item_name = item_name.strip()
        rows = self._open_rows()
        row_id, reason = self._find_row(item_name, rows)

        if row_id is None:
            self.logger.warning(
                f"[InventoryState] Failed to remove '{item_name}' - {reason}. "
                f"Held: {[raw for _, raw in rows]}"
            )
            return False

        dropped = self.db.drop_inventory_row(row_id, turn_number)
        self._load_from_db()
        if dropped:
            self.logger.info(
                f"[InventoryState] Removed '{item_name}' from inventory "
                f"(turn {turn_number}, match={reason})"
            )
        return dropped

    def sync_with_game(self, game_inventory: List[str], turn_number: int):
        """
        Reconcile stored inventory against the game's own INVENTORY listing.

        The game is authoritative: anything it lists is added, anything it does
        not list is dropped. Reconciliation runs against the database rows, not
        against the in-memory mirror, so it can repair state the mirror agrees
        with but the database does not.
        """
        self.logger.info(f"[InventoryState] Syncing with game inventory: {game_inventory}")

        rows = self._open_rows()
        unmatched_rows = list(rows)
        missing: List[str] = []

        for game_item in game_inventory:
            row_id, _ = self._find_row(game_item, unmatched_rows)
            if row_id is None:
                missing.append(game_item)
            else:
                unmatched_rows = [(rid, raw) for rid, raw in unmatched_rows if rid != row_id]

        for row_id, raw in unmatched_rows:
            self.logger.info(f"[InventoryState] Sync: dropping untracked item '{raw}'")
            self.db.drop_inventory_row(row_id, turn_number)

        for item in missing:
            self.logger.info(f"[InventoryState] Sync: adding missing item '{item}'")
            self.db.add_inventory_item(self.session_id, item, turn_number)

        self._load_from_db()
        self.logger.info(f"[InventoryState] Sync complete. Current inventory: {self.items}")

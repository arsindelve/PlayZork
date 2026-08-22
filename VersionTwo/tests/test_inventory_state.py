"""Tests for InventoryState: DB-backed inventory with conservative name matching."""
import pytest

from tools.database import DatabaseManager
from tools.inventory import InventoryState
from tools.inventory.inventory_state import normalize_item_name


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(str(tmp_path / "test.db"))


@pytest.fixture
def inv(db):
    return InventoryState("test-session", db)


def held(state):
    """Inventory as the database records it, bypassing the in-memory mirror."""
    return state.db.get_current_inventory(state.session_id)


class TestNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("brass lantern", "brass lantern"),
        ("Brass Lantern", "brass lantern"),
        ("A brass lantern", "brass lantern"),
        ("  the  brass   lantern  ", "brass lantern"),
        ("jewel-encrusted egg", "jewel encrusted egg"),
        ("", ""),
    ])
    def test_normalize(self, raw, expected):
        assert normalize_item_name(raw) == expected

    def test_article_only_stripped_at_start(self):
        # "a" inside a name is a token, not an article to strip
        assert normalize_item_name("map a") == "map a"


class TestMirrorMatchesDatabase:
    def test_get_items_matches_db_after_add(self, inv):
        inv.add_item("leaflet", 1)
        assert inv.get_items() == held(inv) == ["leaflet"]

    def test_get_items_matches_db_after_remove(self, inv):
        inv.add_item("leaflet", 1)
        inv.remove_item("leaflet", 2)
        assert inv.get_items() == held(inv) == []

    def test_failed_remove_leaves_both_consistent(self, inv):
        inv.add_item("brass lantern", 1)
        assert inv.remove_item("lamp", 2) is False
        assert inv.get_items() == held(inv) == ["brass lantern"]

    def test_state_survives_reconstruction(self, inv, db):
        """Regression: the mirror used to diverge, so inventory changed on resume."""
        inv.add_item("leaflet", 1)
        inv.add_item("leaflet", 1)
        inv.remove_item("leaflet", 2)
        before = inv.get_items()
        after = InventoryState("test-session", db).get_items()
        assert before == after == []


class TestAddDedupe:
    def test_duplicate_add_is_ignored(self, inv):
        inv.add_item("leaflet", 1)
        inv.add_item("leaflet", 2)
        assert held(inv) == ["leaflet"]

    def test_dedupe_is_normalized_not_exact(self, inv):
        inv.add_item("brass lantern", 1)
        inv.add_item("A Brass Lantern", 2)
        assert held(inv) == ["brass lantern"]

    def test_first_name_is_kept_verbatim(self, inv):
        inv.add_item("Brass Lantern", 1)
        inv.add_item("brass lantern", 2)
        assert held(inv) == ["Brass Lantern"]

    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_blank_adds_ignored(self, inv, bad):
        inv.add_item(bad, 1)
        assert held(inv) == []

    def test_distinct_items_both_added(self, inv):
        inv.add_item("skeleton key", 1)
        inv.add_item("rusty key", 2)
        assert held(inv) == ["skeleton key", "rusty key"]


class TestRemoveMatching:
    def test_exact(self, inv):
        inv.add_item("brass lantern", 1)
        assert inv.remove_item("brass lantern", 2) is True
        assert held(inv) == []

    def test_case_and_article_drift(self, inv):
        inv.add_item("Brass Lantern", 1)
        assert inv.remove_item("a brass lantern", 2) is True
        assert held(inv) == []

    def test_head_noun_drift(self, inv):
        inv.add_item("brass lantern", 1)
        assert inv.remove_item("lantern", 2) is True
        assert held(inv) == []

    def test_extra_adjective_drift(self, inv):
        inv.add_item("sword", 1)
        assert inv.remove_item("elvish sword", 2) is True
        assert held(inv) == []

    def test_synonym_is_not_guessed(self, inv):
        """'lamp' must NOT match 'brass lantern': a wrong removal is worse than a stale item."""
        inv.add_item("brass lantern", 1)
        assert inv.remove_item("lamp", 2) is False
        assert held(inv) == ["brass lantern"]

    def test_ambiguous_removal_is_refused(self, inv):
        inv.add_item("skeleton key", 1)
        inv.add_item("rusty key", 2)
        assert inv.remove_item("key", 3) is False
        assert held(inv) == ["skeleton key", "rusty key"]

    def test_ambiguity_resolved_by_full_name(self, inv):
        inv.add_item("skeleton key", 1)
        inv.add_item("rusty key", 2)
        assert inv.remove_item("rusty key", 3) is True
        assert held(inv) == ["skeleton key"]

    def test_unrelated_item_not_removed(self, inv):
        inv.add_item("skeleton key", 1)
        assert inv.remove_item("brass key", 2) is False
        assert held(inv) == ["skeleton key"]

    def test_remove_from_empty_inventory(self, inv):
        assert inv.remove_item("leaflet", 1) is False

    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_blank_removals_ignored(self, inv, bad):
        inv.add_item("leaflet", 1)
        assert inv.remove_item(bad, 2) is False
        assert held(inv) == ["leaflet"]

    def test_removal_affects_one_row_only(self, inv, db):
        """Regression: the UPDATE had no LIMIT and closed every matching row."""
        db.add_inventory_item("test-session", "candle", 1)
        db.add_inventory_item("test-session", "candle", 1)
        inv._load_from_db()
        assert inv.remove_item("candle", 2) is True
        assert held(inv) == ["candle"]

    def test_sessions_are_isolated(self, db):
        a = InventoryState("a", db)
        b = InventoryState("b", db)
        a.add_item("leaflet", 1)
        assert b.get_items() == []
        assert b.remove_item("leaflet", 2) is False
        assert held(a) == ["leaflet"]


class TestSyncWithGame:
    def test_sync_adds_missing(self, inv):
        inv.sync_with_game(["brass lantern", "sword"], 1)
        assert held(inv) == ["brass lantern", "sword"]

    def test_sync_drops_items_game_does_not_list(self, inv):
        inv.add_item("brass lantern", 1)
        inv.add_item("leaflet", 2)
        inv.sync_with_game(["brass lantern"], 3)
        assert held(inv) == ["brass lantern"]

    def test_sync_tolerates_article_and_case_from_game_text(self, inv):
        inv.add_item("brass lantern", 1)
        inv.sync_with_game(["A Brass Lantern"], 2)
        assert held(inv) == ["brass lantern"]  # not re-added, not dropped

    def test_sync_repairs_a_diverged_mirror(self, inv, db):
        """Regression: sync diffed the mirror, so it could never see divergence."""
        inv.add_item("leaflet", 1)
        row_id = db.get_open_inventory_rows("test-session")[0][0]
        db.drop_inventory_row(row_id, 2)      # DB drifts behind the mirror
        inv.items = ["leaflet"]               # stale mirror, as the old code left it
        inv.sync_with_game(["leaflet"], 3)
        assert inv.get_items() == held(inv) == ["leaflet"]

    def test_sync_to_empty_clears_everything(self, inv):
        inv.add_item("sword", 1)
        inv.sync_with_game([], 2)
        assert held(inv) == []


class TestDatabaseRowHelpers:
    def test_open_rows_exclude_dropped(self, inv, db):
        inv.add_item("sword", 1)
        inv.add_item("leaflet", 2)
        inv.remove_item("sword", 3)
        assert [name for _, name in db.get_open_inventory_rows("test-session")] == ["leaflet"]

    def test_drop_row_is_idempotent(self, inv, db):
        inv.add_item("sword", 1)
        row_id = db.get_open_inventory_rows("test-session")[0][0]
        assert db.drop_inventory_row(row_id, 2) is True
        assert db.drop_inventory_row(row_id, 3) is False


# ---------------------------------------------------------------------------
# The analyzer must actually receive what we hold (GitHub issue #21)
# ---------------------------------------------------------------------------


def test_analyzer_is_given_the_current_inventory():
    """The player types "DROP LAMP" while the DB holds "brass lantern"; no
    string-matching rule reconciles those, so the model is given the held
    items and asked to name the removal using our string.

    This also pins the signature: persist_node passes current_inventory=, and
    a mismatch would raise TypeError inside the #1 try/except — silently
    disabling inventory tracking for the whole run rather than failing loudly.
    """
    import logging

    from tools.inventory.inventory_analyzer import InventoryAnalyzer, InventoryChange

    captured = {}

    class FakeChain:
        def invoke(self, payload):
            captured.update(payload)
            return InventoryChange(items_added=[], items_removed=["brass lantern"], reasoning="")

    analyzer = object.__new__(InventoryAnalyzer)
    analyzer.logger = logging.getLogger("test")
    analyzer.chain = FakeChain()

    analyzer.analyze_turn(
        player_command="DROP LAMP",
        game_response="Dropped.",
        current_inventory=["brass lantern", "leaflet"],
    )

    assert captured["current_inventory"] == "brass lantern, leaflet"


def test_analyzer_renders_an_empty_inventory_readably():
    import logging

    from tools.inventory.inventory_analyzer import InventoryAnalyzer, InventoryChange

    captured = {}

    class FakeChain:
        def invoke(self, payload):
            captured.update(payload)
            return InventoryChange()

    analyzer = object.__new__(InventoryAnalyzer)
    analyzer.logger = logging.getLogger("test")
    analyzer.chain = FakeChain()

    analyzer.analyze_turn(player_command="LOOK", game_response="You see nothing.")

    assert captured["current_inventory"] == "(empty)"

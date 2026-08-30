"""MemoryDeduplicator must never end the run (GitHub issue #2).

It was the only LLM call in the turn path with no retry, no timeout and no
try/except, and it runs *after* the turn's command has already executed.
"""

from types import SimpleNamespace

import pytest

from tools.memory.memory_deduplicator import (
    DEDUP_MAX_RETRIES,
    MemoryDeduplicator,
    _as_bool,
)
from tools.memory.memory_state import MemoryState


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    """Skip invoke_with_retry's exponential backoff so tests stay fast."""
    monkeypatch.setattr("llm_utils.time", SimpleNamespace(sleep=lambda seconds: None))


class FakeChain:
    """Stands in for `prompt | llm.with_structured_output(...)`."""

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.invocations = []

    def with_config(self, **kwargs):
        return self

    def invoke(self, payload):
        self.invocations.append(payload)
        if self._error is not None:
            raise self._error
        return self._result


def _make_deduplicator(chain):
    """Build a deduplicator without constructing a real LLM chain."""
    dedup = object.__new__(MemoryDeduplicator)
    dedup.chain = chain
    dedup.logger = SimpleNamespace(error=lambda *a, **k: None, info=lambda *a, **k: None)
    return dedup


def test_returns_llm_verdict_when_healthy():
    chain = FakeChain(SimpleNamespace(is_duplicate=True, reason="same troll"))
    dedup = _make_deduplicator(chain)

    is_dup, reason = dedup.is_duplicate("troll demands payment", ["troll wants money"])

    assert is_dup is True
    assert reason == "same troll"


def test_llm_error_fails_open_instead_of_raising():
    chain = FakeChain(error=RuntimeError("connection reset by peer"))
    dedup = _make_deduplicator(chain)

    is_dup, reason = dedup.is_duplicate("troll demands payment", ["troll wants money"])

    # Fail open: the issue gets stored, the run continues.
    assert is_dup is False
    assert "unavailable" in reason
    # It retried before giving up.
    assert len(chain.invocations) == DEDUP_MAX_RETRIES


def test_parse_error_fails_open():
    """ChatOllama's json_schema path raises rather than returning None."""
    chain = FakeChain(error=ValueError("Failed to parse DeduplicationResult from ''"))
    dedup = _make_deduplicator(chain)

    is_dup, _ = dedup.is_duplicate("new issue", ["old issue"])

    assert is_dup is False


def test_none_result_fails_open():
    chain = FakeChain(result=None)
    dedup = _make_deduplicator(chain)

    is_dup, reason = dedup.is_duplicate("new issue", ["old issue"])

    assert is_dup is False
    assert "unavailable" in reason


def test_dict_result_is_handled():
    chain = FakeChain(result={"is_duplicate": True, "reason": "already tracked"})
    dedup = _make_deduplicator(chain)

    assert dedup.is_duplicate("new issue", ["old issue"]) == (True, "already tracked")


def test_string_false_is_not_treated_as_duplicate():
    """A truthy "false" string would silently discard a real new issue."""
    chain = FakeChain(result={"is_duplicate": "false", "reason": "different puzzle"})
    dedup = _make_deduplicator(chain)

    is_dup, _ = dedup.is_duplicate("new issue", ["old issue"])

    assert is_dup is False


def test_no_existing_issues_skips_the_llm_entirely():
    chain = FakeChain(error=RuntimeError("should never be called"))
    dedup = _make_deduplicator(chain)

    is_dup, _ = dedup.is_duplicate("first issue ever", [])

    assert is_dup is False
    assert chain.invocations == []


def test_as_bool_coercions():
    assert _as_bool(True) is True
    assert _as_bool(False) is False
    assert _as_bool("true") is True
    assert _as_bool("FALSE") is False
    assert _as_bool("no") is False
    assert _as_bool(None) is False


class FakeDb:
    """Minimal DatabaseManager stand-in for MemoryState."""

    def __init__(self):
        self.added = []

    def check_duplicate_memory(self, session_id, content):
        return False

    def get_top_memories(self, session_id, limit, **kwargs):
        # (id, content, importance, turn_number, location)
        return [(1, "troll wants money", 600, 4, "Troll Room")]

    def add_memory(self, **kwargs):
        self.added.append(kwargs)


def test_memory_is_still_stored_when_dedup_fails():
    """End-to-end fail-open: persist keeps working through a dedup outage."""
    db = FakeDb()
    dedup = _make_deduplicator(FakeChain(error=RuntimeError("ollama down")))
    state = MemoryState(session_id="s1", db=db, deduplicator=dedup)

    memory = state.add_memory(
        content="troll demands payment",
        importance=700,
        turn_number=12,
        location="Troll Room",
        score=10,
        moves=12,
    )

    assert memory is not None
    assert len(db.added) == 1
    assert db.added[0]["content"] == "troll demands payment"

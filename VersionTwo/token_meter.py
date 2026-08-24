"""Per-turn token accounting.

Wall-clock alone cannot compare two architectures. On fixed serving, turn time
is total tokens processed divided by a roughly fixed tokens-per-second — this
machine's Ollama was benchmarked at flat throughput across 1/2/4/8 concurrent
requests, so concurrency changes nothing and only token volume moves the
number (see STATUS.md, 2026-08-24).

That makes `score@wall-clock` a proxy for token volume on any one rig, and it
makes cross-rig comparison meaningless without a common unit. Tokens are that
unit: they are hardware-independent, so a run on a laptop and a run on a GPU
are directly comparable, and the multi-agent arm can be charged for what it
actually costs rather than for how fast the box happens to be.

Counts come from the provider's own metadata (`usage_metadata` on the LangChain
response), not from an estimate.
"""
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnTokens:
    """Token cost of a single turn, broken down by the operation that spent it."""

    turn_number: int
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    # operation name -> (input, output, calls)
    by_operation: Dict[str, tuple] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> str:
        return (f"turn {self.turn_number}: {self.calls} calls, "
                f"{self.input_tokens} in + {self.output_tokens} out = "
                f"{self.total_tokens} tokens")


class TokenMeter:
    """Accumulates token usage for the turn currently being played.

    Thread-safe: LangGraph runs sync nodes on executor threads, and the graph
    fans out, so several branches report concurrently.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._turn = 0
        self._input = 0
        self._output = 0
        self._calls = 0
        self._by_op = defaultdict(lambda: [0, 0, 0])

    def start_turn(self, turn_number: int) -> None:
        with self._lock:
            self._turn = turn_number
            self._input = self._output = self._calls = 0
            self._by_op.clear()

    def record(self, response, operation_name: str = "") -> None:
        """Record one LLM response. Never raises — accounting must not be able
        to cost a turn."""
        try:
            usage = getattr(response, "usage_metadata", None) or {}
            tin = int(usage.get("input_tokens") or 0)
            tout = int(usage.get("output_tokens") or 0)
            if not tin and not tout:
                meta = getattr(response, "response_metadata", None) or {}
                tin = int(meta.get("prompt_eval_count") or 0)
                tout = int(meta.get("eval_count") or 0)
            if not tin and not tout:
                return
            with self._lock:
                self._input += tin
                self._output += tout
                self._calls += 1
                slot = self._by_op[operation_name or "unnamed"]
                slot[0] += tin
                slot[1] += tout
                slot[2] += 1
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[TokenMeter] could not record usage: {e}")

    def snapshot(self) -> TurnTokens:
        with self._lock:
            return TurnTokens(
                turn_number=self._turn,
                input_tokens=self._input,
                output_tokens=self._output,
                calls=self._calls,
                by_operation={k: tuple(v) for k, v in self._by_op.items()},
            )


# One meter per process; the game plays one turn at a time.
_METER = TokenMeter()


def get_token_meter() -> TokenMeter:
    return _METER

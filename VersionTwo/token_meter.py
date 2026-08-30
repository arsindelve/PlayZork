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
import contextvars
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# The operation whose LLM calls are currently in flight. A contextvar so it
# survives both the asyncio-task boundary (the graph fans out) and the
# thread-pool boundary (LangChain may run callbacks on a worker).
_OPERATION: contextvars.ContextVar[str] = contextvars.ContextVar(
    "playzork_llm_operation", default="")


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

    def set_operation(self, name: str) -> None:
        """Label the operation whose LLM calls follow.

        A CONTEXTVAR, not a thread-local. The graph fans out, so several
        branches label and call concurrently — but they do so as asyncio tasks
        sharing one thread, where a thread-local would let them overwrite each
        other, and LangChain may run the callback on a worker thread, where a
        thread-local set on the event loop would be invisible entirely (it
        was: every call came back "unattributed"). Contextvars propagate
        across both boundaries.
        """
        _OPERATION.set(name)

    @staticmethod
    def _current_operation() -> str:
        return _OPERATION.get() or "unattributed"

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
                slot = self._by_op[operation_name or self._current_operation()]
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


from langchain_core.callbacks import BaseCallbackHandler


class TokenCallbackHandler(BaseCallbackHandler):
    """Meters EVERY LLM call, including structured-output ones.

    `with_structured_output(...)` returns a parsed Pydantic model, which throws
    away the AIMessage — and with it `usage_metadata`. Metering the return
    value therefore counted only the calls that hand back a raw message
    (summaries, big-picture analysis) and missed every agent proposal and the
    decision itself: precisely the architecture's own work, and precisely what
    the experiment needs to charge it for.

    A callback sees the raw LLMResult before parsing, so it catches all of
    them. Attached at the LLM factory, so no call site can opt out by accident.
    """

    def __init__(self, meter: "TokenMeter"):
        super().__init__()
        self._meter = meter

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            name = self._meter._current_operation()
            for generation_list in getattr(response, "generations", []) or []:
                for generation in generation_list:
                    message = getattr(generation, "message", None)
                    if message is not None:
                        self._meter.record(message, name)
        except Exception:  # noqa: BLE001 - accounting must never cost a turn
            pass


# One meter per process; the game plays one turn at a time.
_METER = TokenMeter()
_CALLBACK = TokenCallbackHandler(_METER)


def get_token_callback() -> TokenCallbackHandler:
    """The handler to attach to every LLM instance."""
    return _CALLBACK


def get_token_meter() -> TokenMeter:
    return _METER

"""Per-turn token accounting (thesis instrumentation).

Wall-clock cannot compare architectures across machines, and on one machine it
is a proxy for token volume anyway — this box's Ollama benchmarks at flat
throughput across 1/2/4/8 concurrent requests, so only token count moves the
number. Tokens are the unit that survives a hardware change.
"""
import threading
from types import SimpleNamespace

from token_meter import TokenMeter, get_token_meter


def _msg(tin, tout):
    return SimpleNamespace(usage_metadata={"input_tokens": tin, "output_tokens": tout})


def test_counts_come_from_provider_metadata_not_an_estimate():
    m = TokenMeter()
    m.start_turn(3)
    m.record(_msg(1500, 80), "Decision Agent")
    m.record(_msg(1200, 60), "IssueAgent Proposal")

    s = m.snapshot()
    assert s.input_tokens == 2700
    assert s.output_tokens == 140
    assert s.total_tokens == 2840
    assert s.calls == 2


def test_falls_back_to_ollama_response_metadata():
    """Some paths surface prompt_eval_count/eval_count rather than
    usage_metadata."""
    m = TokenMeter()
    m.start_turn(1)
    m.record(SimpleNamespace(response_metadata={"prompt_eval_count": 900, "eval_count": 40}), "Observer")

    s = m.snapshot()
    assert (s.input_tokens, s.output_tokens) == (900, 40)


def test_a_response_without_usage_is_skipped_not_guessed():
    """Structured-output chains return a Pydantic model carrying no usage
    metadata. Totals are a floor, never an estimate — an invented number would
    be worse than a missing one for a thesis measurement."""
    m = TokenMeter()
    m.start_turn(1)
    m.record(SimpleNamespace(), "structured output")
    m.record(_msg(100, 10), "real call")

    s = m.snapshot()
    assert s.calls == 1
    assert s.total_tokens == 110


def test_accounting_never_raises():
    """It must not be able to cost a turn."""
    m = TokenMeter()
    m.start_turn(1)
    m.record(None, "none")
    m.record(SimpleNamespace(usage_metadata={"input_tokens": "not a number"}), "junk")
    m.record(object(), "bare object")

    assert m.snapshot().calls == 0


def test_usage_is_attributed_per_operation():
    m = TokenMeter()
    m.start_turn(2)
    m.record(_msg(100, 10), "Decision Agent")
    m.record(_msg(200, 20), "Decision Agent")
    m.record(_msg(300, 30), "Observer")

    by_op = m.snapshot().by_operation
    assert by_op["Decision Agent"] == (300, 30, 2)
    assert by_op["Observer"] == (300, 30, 1)


def test_starting_a_turn_clears_the_previous_one():
    m = TokenMeter()
    m.start_turn(1)
    m.record(_msg(999, 99), "old")
    m.start_turn(2)
    m.record(_msg(10, 1), "new")

    s = m.snapshot()
    assert s.turn_number == 2
    assert s.total_tokens == 11
    assert "old" not in s.by_operation


def test_concurrent_branches_can_report_safely():
    """The graph fans out and LangGraph runs sync nodes on executor threads,
    so several branches report at once."""
    m = TokenMeter()
    m.start_turn(1)

    def worker():
        for _ in range(200):
            m.record(_msg(1, 1), "branch")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()

    s = m.snapshot()
    assert s.calls == 1600
    assert s.input_tokens == 1600


def test_every_guarded_call_labels_its_operation():
    """llm_utils supplies the LABEL; the metering itself happens in a callback,
    because a structured-output chain discards the message carrying usage."""
    import inspect

    import llm_utils

    for fn in (llm_utils.invoke_with_retry, llm_utils.ainvoke_with_retry):
        assert "_label(operation_name)" in inspect.getsource(fn), fn.__name__


def test_structured_output_calls_are_counted_via_the_callback():
    """The gap this closed: metering the RETURN VALUE missed every agent
    proposal and the decision, since those return a parsed Pydantic model."""
    from types import SimpleNamespace

    from token_meter import TokenCallbackHandler, TokenMeter

    meter = TokenMeter()
    meter.start_turn(1)
    meter.set_operation("Decision Agent")
    handler = TokenCallbackHandler(meter)

    handler.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(
        message=SimpleNamespace(usage_metadata={"input_tokens": 1500, "output_tokens": 80}))]]))

    snapshot = meter.snapshot()
    assert snapshot.total_tokens == 1580
    assert snapshot.by_operation["Decision Agent"] == (1500, 80, 1)


def test_the_handler_never_raises_on_a_malformed_result():
    from token_meter import TokenCallbackHandler, TokenMeter

    handler = TokenCallbackHandler(TokenMeter())
    handler.on_llm_end(None)
    handler.on_llm_end(object())


def test_operation_labels_survive_the_asyncio_task_boundary():
    """The graph fans out as asyncio tasks. Each must keep its own label.

    A thread-local fails here twice over: the tasks share one thread so they
    would overwrite each other, and LangChain may run the callback on a worker
    thread where a label set on the event loop is invisible — which is exactly
    what happened, every call came back "unattributed".
    """
    import asyncio

    from token_meter import TokenMeter

    meter = TokenMeter()
    meter.start_turn(1)
    seen = {}

    async def branch(name, delay):
        meter.set_operation(name)
        await asyncio.sleep(delay)          # let the other branch interleave
        seen[name] = meter._current_operation()

    async def main():
        await asyncio.gather(branch("spawn", 0.02), branch("observe", 0.01))

    asyncio.run(main())

    assert seen == {"spawn": "spawn", "observe": "observe"}


def test_an_unlabelled_call_is_recorded_as_unattributed_not_dropped():
    """Losing the label must cost attribution, never the count — the total is
    what the experiment compares."""
    from token_meter import TokenMeter

    meter = TokenMeter()
    meter.start_turn(1)
    meter.set_operation("")          # no label in scope
    meter.record(_msg(100, 10))

    snapshot = meter.snapshot()
    assert snapshot.total_tokens == 110
    assert "unattributed" in snapshot.by_operation


def test_the_process_meter_is_a_singleton():
    assert get_token_meter() is get_token_meter()

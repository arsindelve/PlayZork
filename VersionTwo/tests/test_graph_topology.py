"""The graph runs independent work in parallel (#23, #26).

`close_issues` and `observe` were chained AFTER `decide`, so ~20% of every
turn was spent on bookkeeping once the command was already chosen — measured
at 18-24% on the 2026-08-24 run. Neither has any data dependency on the
decision: both read the game response, available at the top of the turn.

They now run BESIDE the spawn->decide chain and finish inside its shadow. This
is also the first real use of LangGraph in the project: it was previously a
straight line that would have behaved identically as sequential awaits.
"""
import asyncio
from types import SimpleNamespace

import pytest

import tools.agent_graph.decision_graph as dg


def _graph():
    stub = SimpleNamespace(get_tools=lambda: [], state=SimpleNamespace())
    return dg.create_decision_graph(
        decision_chain=object(), decision_llm=object(), history_toolkit=stub,
        memory_toolkit=stub, mapper_toolkit=stub, inventory_toolkit=stub,
        turn_number_ref={"current": 0},
    )


def _edges():
    return {(e.source, e.target) for e in _graph().get_graph().edges}


def test_bookkeeping_branches_start_from_the_context_not_the_decision():
    edges = _edges()

    assert ("build_context", "close_issues") in edges
    assert ("build_context", "observe") in edges
    # The old chain must be gone, or the work is still after the command.
    assert ("decide", "close_issues") not in edges
    assert ("close_issues", "observe") not in edges


def test_all_three_branches_join_at_persist():
    edges = _edges()

    assert ("decide", "persist") in edges
    assert ("close_issues", "persist") in edges
    assert ("observe", "persist") in edges


def test_persist_remains_the_single_writer():
    """The parallel branches must stay read-only with respect to memory:
    close_issues stages closures (#3) and persist commits them, which is what
    makes running them beside the decision safe."""
    import inspect

    source = inspect.getsource(dg.create_close_issues_node)

    assert "remove_memory" not in source
    assert "pending_closures" in source


def test_nodes_return_only_their_own_keys():
    """Required for parallel branches: a node returning the whole state would
    look to LangGraph like every node writing every key."""
    import inspect

    writes = {
        "create_spawn_agents_node": {"issue_agents", "explorer_agent",
                                     "loop_detection_agent", "interaction_agent"},
        "create_close_issues_node": {"issue_closed_response", "pending_closures"},
        "create_observe_node": {"observer_response"},
        "create_decision_node": {"decision", "decision_prompt", "decision_tool_calls"},
    }
    seen = {}
    for factory, expected in writes.items():
        source = inspect.getsource(getattr(dg, factory))
        assert "return state" not in source, f"{factory} returns the whole state"
        seen[factory] = expected

    # And the branches that run concurrently must not overlap at all.
    concurrent = [seen["create_spawn_agents_node"],
                  seen["create_close_issues_node"],
                  seen["create_observe_node"]]
    for i, a in enumerate(concurrent):
        for b in concurrent[i + 1:]:
            assert not (a & b), f"concurrent branches both write {a & b}"


def test_the_three_branches_actually_run_concurrently():
    """Executes a compiled graph with the real topology and asserts the
    branches overlap in time — the property the whole refactor is for.

    Note the state type is a TypedDict with per-key fields, not a bare dict.
    With `dict`, LangGraph treats the whole mapping as one `__root__` value and
    parallel branches raise InvalidUpdateError. That is precisely why the real
    nodes were changed to return only their own (disjoint) keys.
    """
    from typing import TypedDict

    from langgraph.graph import END, StateGraph

    class ParallelState(TypedDict, total=False):
        spawned: bool
        decided: bool
        closed: bool
        observed: bool

    running = []
    peak = {"n": 0}

    def track(name, delay, key):
        async def node(state):
            running.append(name)
            peak["n"] = max(peak["n"], len(running))
            await asyncio.sleep(delay)
            running.remove(name)
            return {key: True}
        return node

    graph = StateGraph(ParallelState)
    graph.add_node("build_context", lambda s: {})
    graph.add_node("spawn_agents", track("spawn", 0.06, "spawned"))
    graph.add_node("decide", track("decide", 0.01, "decided"))
    graph.add_node("close_issues", track("close", 0.04, "closed"))
    graph.add_node("observe", track("observe", 0.04, "observed"))
    graph.add_node("persist", lambda s: {})
    graph.set_entry_point("build_context")
    for target in ("spawn_agents", "close_issues", "observe"):
        graph.add_edge("build_context", target)
    graph.add_edge("spawn_agents", "decide")
    for source in ("decide", "close_issues", "observe"):
        graph.add_edge(source, "persist")
    graph.add_edge("persist", END)

    result = asyncio.run(graph.compile().ainvoke({}))

    assert peak["n"] >= 3, f"branches did not overlap (peak concurrency {peak['n']})"
    # Every branch's write survived the join — no concurrent-update conflict.
    assert result["spawned"] and result["decided"] and result["closed"] and result["observed"]


def test_the_real_state_type_keeps_branch_keys_separate():
    """A regression guard on the state declaration itself: if DecisionState
    ever became a bare dict, the parallel branches would start colliding."""
    assert hasattr(dg.DecisionState, "__annotations__")
    keys = dg.DecisionState.__annotations__
    for key in ("issue_agents", "issue_closed_response", "observer_response", "decision"):
        assert key in keys, f"{key} must be its own state field, not nested"


def test_persist_runs_exactly_once_per_turn():
    """A fan-in built from separate edges is NOT a join. The branches have
    different depths, so persist was scheduled once when close/observe
    finished and again when decide finished — verified live, PERSIST ran twice
    per turn, double-applying the turn's bookkeeping.

    LangGraph's list start_key waits for ALL named nodes.
    """
    from typing import TypedDict

    from langgraph.graph import END, StateGraph

    class S(TypedDict, total=False):
        spawned: bool
        decided: bool
        closed: bool
        observed: bool

    runs = {"persist": 0}

    async def slow(key, delay):
        async def node(state):
            await asyncio.sleep(delay)
            return {key: True}
        return node

    def mk(key, delay):
        async def node(state):
            await asyncio.sleep(delay)
            return {key: True}
        return node

    def persist(state):
        runs["persist"] += 1
        return {}

    graph = StateGraph(S)
    graph.add_node("build_context", lambda s: {})
    graph.add_node("spawn_agents", mk("spawned", 0.05))
    graph.add_node("decide", mk("decided", 0.01))
    graph.add_node("close_issues", mk("closed", 0.01))
    graph.add_node("observe", mk("observed", 0.01))
    graph.add_node("persist", persist)
    graph.set_entry_point("build_context")
    for t in ("spawn_agents", "close_issues", "observe"):
        graph.add_edge("build_context", t)
    graph.add_edge("spawn_agents", "decide")
    graph.add_edge(["decide", "close_issues", "observe"], "persist")
    graph.add_edge("persist", END)

    asyncio.run(graph.compile().ainvoke({}))

    assert runs["persist"] == 1, f"persist ran {runs['persist']} times"


def test_the_real_graph_uses_a_join_not_three_separate_edges():
    import inspect

    source = inspect.getsource(dg.create_decision_graph)

    assert '["decide", "close_issues", "observe"], "persist"' in source, \
        "persist must join all three branches, or it runs twice per turn"

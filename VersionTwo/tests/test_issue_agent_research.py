"""IssueAgent research parsing (GitHub issue #6).

The agent instructed the research LLM to call `get_current_inventory`, which
does not exist — the real tool is `get_inventory` — so IssueAgents never saw
inventory. After the #1 tool-execution guard landed, the failure got worse
rather than better: the resulting "Error: unknown tool ..." string was split
on commas into phantom carried items, injecting the name of every available
tool into the proposal prompt as inventory.
"""
import asyncio
from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda

import llm_utils
from tools.agent_graph.issue_agent import IssueAgent, IssueProposal


class FakeTool:
    def __init__(self, name, result=None, error=None):
        self.name = name
        self._result = result
        self._error = error

    def invoke(self, args):
        if self._error is not None:
            raise self._error
        return self._result


def make_agent(issue_location="Exit Hallway"):
    memory = SimpleNamespace(
        id=1, content="Locked metal door at Exit Hallway - need to unlock",
        importance=800, turn_number=4, location=issue_location, score=0, moves=4,
    )
    return IssueAgent(memory)


def run_agent(monkeypatch, tool_calls, tools):
    """Drive research_and_propose with fakes; return the proposal prompt inputs."""
    captured = {}

    async def fake_ainvoke_with_retry(chain, payload, operation_name="", **kwargs):
        if operation_name.startswith("IssueAgent Research"):
            captured["research_instruction"] = payload["input"]
            return SimpleNamespace(tool_calls=tool_calls, content="")
        captured["proposal_input"] = payload
        return IssueProposal(proposed_action="SOUTH", reason="r", confidence=90)

    monkeypatch.setattr(llm_utils, "ainvoke_with_retry", fake_ainvoke_with_retry)

    agent = make_agent()
    decision_llm = SimpleNamespace(
        with_structured_output=lambda schema: RunnableLambda(lambda _: None)
    )
    asyncio.run(agent.research_and_propose(
        research_agent=RunnableLambda(lambda _: None),
        decision_llm=decision_llm,
        history_tools=tools,
        current_location="Storage Closet",
        current_game_response="You are in a storage closet.",
        current_score=0,
        current_moves=4,
    ))
    return captured


REAL_TOOLS = [
    FakeTool("get_inventory", result="brass key, leaflet"),
    FakeTool("get_direction_to_location", result="SOUTH"),
    FakeTool("get_full_summary", result="summary"),
]


def test_real_inventory_tool_result_reaches_the_proposal_prompt(monkeypatch):
    """#6: a successful get_inventory call must populate inventory_summary."""
    captured = run_agent(monkeypatch, [{"name": "get_inventory", "args": {}}], REAL_TOOLS)
    assert captured["proposal_input"]["inventory_summary"] == "brass key, leaflet"


def test_research_instruction_only_names_tools_that_exist(monkeypatch):
    """#6: the instruction must not tell the model to call a phantom tool."""
    captured = run_agent(monkeypatch, [], REAL_TOOLS)
    instruction = captured["research_instruction"]
    assert "get_current_inventory" not in instruction
    assert "get_inventory(" in instruction


def test_unknown_tool_error_does_not_leak_into_inventory_summary(monkeypatch):
    """#6 x #1: an 'Error: unknown tool ...' string must not be parsed as items."""
    captured = run_agent(
        monkeypatch, [{"name": "get_current_inventory", "args": {}}], REAL_TOOLS
    )
    assert captured["proposal_input"]["inventory_summary"] == "empty"


def test_failing_inventory_tool_does_not_leak_into_inventory_summary(monkeypatch):
    """A get_inventory that raises must yield 'empty', not the error text."""
    tools = [FakeTool("get_inventory", error=RuntimeError("database is locked"))]
    captured = run_agent(monkeypatch, [{"name": "get_inventory", "args": {}}], tools)
    assert captured["proposal_input"]["inventory_summary"] == "empty"


def test_empty_inventory_renders_as_empty(monkeypatch):
    tools = [FakeTool("get_inventory", result="Your inventory is empty.")]
    captured = run_agent(monkeypatch, [{"name": "get_inventory", "args": {}}], tools)
    assert captured["proposal_input"]["inventory_summary"] == "empty"


def test_failed_pathfinding_does_not_render_as_a_direction(monkeypatch):
    """The nav branch had the same defect: an "Error: ..." result was passed
    through verbatim as NAVIGATION DIRECTION, which the proposal prompt's
    routing rules cannot interpret."""
    tools = [FakeTool("get_direction_to_location", error=ValueError("to_location Field required"))]
    captured = run_agent(
        monkeypatch, [{"name": "get_direction_to_location", "args": {"from_location": "A"}}], tools
    )
    assert captured["proposal_input"]["navigation_direction"] == "NOT AVAILABLE"


def test_successful_pathfinding_still_reaches_the_prompt(monkeypatch):
    captured = run_agent(
        monkeypatch,
        [{"name": "get_direction_to_location", "args": {"from_location": "A", "to_location": "B"}}],
        REAL_TOOLS,
    )
    assert captured["proposal_input"]["navigation_direction"] == "SOUTH"


def test_no_path_result_is_preserved(monkeypatch):
    tools = [FakeTool("get_direction_to_location", result="NO PATH")]
    captured = run_agent(
        monkeypatch,
        [{"name": "get_direction_to_location", "args": {"from_location": "A", "to_location": "B"}}],
        tools,
    )
    assert captured["proposal_input"]["navigation_direction"] == "NO PATH"

"""Tests for safe execution of model-supplied tool calls (GitHub issue #1)."""

import pytest

from tools.agent_graph.tool_execution import TOOL_ERROR_PREFIX, invoke_tool_safely


class FakeTool:
    """Minimal stand-in for a LangChain tool."""

    def __init__(self, name, result=None, error=None):
        self.name = name
        self._result = result
        self._error = error
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        if self._error is not None:
            raise self._error
        return self._result


def test_returns_tool_result_on_success():
    tool = FakeTool("get_map", result="West Of House --[NORTH]--> North Of House")
    tools_map = {tool.name: tool}

    result = invoke_tool_safely(tools_map, "get_map", {})

    assert result == "West Of House --[NORTH]--> North Of House"
    assert tool.calls == [{}]


def test_missing_required_argument_does_not_raise():
    """The confirmed trigger from issue #1: the model drops a required arg."""
    tool = FakeTool(
        "get_direction_to_location",
        error=ValueError("to_location Field required"),
    )
    tools_map = {tool.name: tool}

    result = invoke_tool_safely(
        tools_map,
        "get_direction_to_location",
        {"from_location": "A"},
    )

    assert isinstance(result, str)
    assert result.startswith(TOOL_ERROR_PREFIX)
    assert "get_direction_to_location" in result
    assert "to_location Field required" in result


def test_unknown_tool_returns_error_naming_available_tools():
    """A hallucinated tool name (e.g. #6's get_current_inventory) is reported
    back to the model instead of being silently dropped."""
    tools_map = {"get_inventory": FakeTool("get_inventory", result="lamp")}

    result = invoke_tool_safely(tools_map, "get_current_inventory", {})

    assert result.startswith(TOOL_ERROR_PREFIX)
    assert "get_current_inventory" in result
    assert "get_inventory" in result


def test_empty_tools_map_does_not_raise():
    result = invoke_tool_safely({}, "get_map", {})

    assert result.startswith(TOOL_ERROR_PREFIX)


def test_none_args_are_passed_as_empty_dict():
    tool = FakeTool("get_full_summary", result="summary")
    tools_map = {tool.name: tool}

    assert invoke_tool_safely(tools_map, "get_full_summary", None) == "summary"
    assert tool.calls == [{}]


@pytest.mark.parametrize(
    "error",
    [
        TypeError("unhashable type"),
        RuntimeError("database is locked"),
        KeyError("location"),
    ],
)
def test_any_tool_exception_is_contained(error):
    tool = FakeTool("get_exits_from_location", error=error)

    result = invoke_tool_safely({tool.name: tool}, "get_exits_from_location", {"x": 1})

    assert result.startswith(TOOL_ERROR_PREFIX)

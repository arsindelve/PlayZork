"""Safe execution of LLM-requested tool calls.

Every tool call executed on the per-turn path is *model-supplied*: the tool
name and all of its arguments come out of an LLM response. A malformed call
(missing required argument, wrong type, hallucinated tool name) makes
``tool.invoke(...)`` raise — historically far enough up the stack to end the
whole game session (GitHub issue #1).

`invoke_tool_safely` is the single choke point for those calls. It never
raises: a failure is turned into an error string that is handed back to the
model as the tool result, exactly the way a real tool-calling loop reports
errors, so the agent can react instead of the run dying.
"""

import logging

logger = logging.getLogger(__name__)

# Error strings start with this prefix so downstream parsing (e.g. IssueAgent's
# navigation-direction extraction) can recognize a failed call.
TOOL_ERROR_PREFIX = "Error:"


def invoke_tool_safely(tools_map, tool_name, tool_args=None, *, label="", log=None):
    """Execute one model-requested tool call without ever raising.

    Args:
        tools_map: Mapping of tool name -> LangChain tool.
        tool_name: Tool name requested by the model.
        tool_args: Arguments dict requested by the model (may be None/empty).
        label: Caller name used as a log prefix (e.g. "IssueAgent ID:3").
        log: Logger to use (defaults to this module's logger).

    Returns:
        The tool's result on success, or a string starting with "Error:"
        describing what went wrong. The error string is safe to feed back to
        the model as the tool result.
    """
    active_log = log or logger
    prefix = f"[{label}] " if label else ""

    tool = tools_map.get(tool_name) if tools_map else None
    if tool is None:
        available = ", ".join(sorted(tools_map)) if tools_map else "none"
        active_log.warning(
            f"{prefix}Model requested unknown tool '{tool_name}'. Available: {available}"
        )
        return (
            f"{TOOL_ERROR_PREFIX} unknown tool '{tool_name}'. "
            f"Available tools: {available}"
        )

    try:
        return tool.invoke(tool_args or {})
    except Exception as exc:  # noqa: BLE001 - model-supplied args; must not escape
        active_log.warning(
            f"{prefix}Tool '{tool_name}' failed with args {tool_args}: {exc}",
            exc_info=True,
        )
        return f"{TOOL_ERROR_PREFIX} tool '{tool_name}' failed: {exc}"

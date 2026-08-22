"""
IssueClosedAgent - Identifies resolved issues for persist_node to close.

This agent analyzes recent game history to determine if any tracked
strategic issues have been solved. It does NOT write to memory itself:
closures are staged and applied by persist_node, so a turn cancelled by
the turn budget cannot leave memory half-applied (GitHub issue #3).

Responsibility: Aggressively close resolved issues to keep memory clean.
Runs BEFORE ObserverAgent to avoid confusion with stale issues.
"""
from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from .issue_closed_response import IssueClosedResponse
from tools.memory import MemoryToolkit
from tools.history import HistoryToolkit
from adventurer.prompt_library import PromptLibrary
from .tool_execution import invoke_tool_safely, TOOL_ERROR_PREFIX
import logging


class IssueClosedAgent:
    """
    Analyzes recent game history to identify and close resolved issues.

    This agent is called AFTER:
    - Decision Agent chooses a command
    - Command is executed
    - Game response is received

    But BEFORE:
    - Observer Agent identifies new issues

    Its job: Close any tracked issues that have been solved.
    """

    def __init__(self):
        """Initialize the IssueClosedAgent"""
        self.logger = logging.getLogger(__name__)

    def analyze(
        self,
        game_response: str,
        location: str,
        score: int,
        moves: int,
        decision_llm: BaseChatModel,
        history_toolkit: HistoryToolkit,
        memory_toolkit: MemoryToolkit
    ) -> tuple[IssueClosedResponse, List[dict]]:
        """
        Analyze recent history and decide which issues are resolved.

        READ-ONLY with respect to memory: this never closes anything itself
        (GitHub issue #3). It returns the closures for persist_node to apply,
        so a turn cancelled by the budget leaves memory untouched rather than
        half-applied.

        Args:
            game_response: The game's response after command execution
            location: Current location name
            score: Current game score
            moves: Current move count
            decision_llm: The LLM to use for analysis
            history_toolkit: HistoryToolkit for accessing recent turns
            memory_toolkit: MemoryToolkit for accessing and removing tracked issues

        Returns:
            Tuple of (IssueClosedResponse, pending_closures), where
            pending_closures is a list of {"id": int, "display": str | None}
            dicts for persist_node to apply.
        """
        self.logger.info(f"[IssueClosedAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.logger.info(f"[IssueClosedAgent] AGENT: IssueClosedAgent")
        self.logger.info(f"[IssueClosedAgent] PURPOSE: Identify and remove resolved issues from memory")
        self.logger.info(f"[IssueClosedAgent] LOCATION: {location}")
        self.logger.info(f"[IssueClosedAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.logger.info(f"[IssueClosedAgent] Analyzing recent history at {location}")

        # Phase 1: Get tracked issues
        self.logger.info(f"[IssueClosedAgent] Phase 1: Retrieving tracked issues...")
        tracked_issues = memory_toolkit.state.get_top_memories(limit=30)  # Get top 30

        if not tracked_issues:
            self.logger.info(f"[IssueClosedAgent] No tracked issues to analyze")
            return IssueClosedResponse(closed_issue_ids=[], closed_issue_contents=[], reasoning="No issues tracked yet."), []

        tracked_issues_text = "\n".join([
            f"- [ID:{mem.id}, Importance:{mem.importance}/1000] {mem.content}"
            for mem in tracked_issues
        ])

        self.logger.info(f"[IssueClosedAgent] Found {len(tracked_issues)} tracked issues")

        # Phase 2: Gather recent history (last 5 turns)
        self.logger.info(f"[IssueClosedAgent] Phase 2: Gathering recent history...")

        # Use history toolkit to get recent turns. invoke_tool_safely covers
        # both "tool missing" and "tool raised" without ending the turn (#1).
        tools_map = {tool.name: tool for tool in history_toolkit.get_tools()}

        self.logger.info(f"[IssueClosedAgent]   -> get_recent_turns(n=5)")
        recent_history = invoke_tool_safely(
            tools_map,
            "get_recent_turns",
            {"n": 5},
            label="IssueClosedAgent",
            log=self.logger,
        )
        if str(recent_history).startswith(TOOL_ERROR_PREFIX):
            recent_history = "No recent history available."
        self.logger.info(f"[IssueClosedAgent]      Result: {str(recent_history)[:150]}...")
        self.logger.info(f"[IssueClosedAgent] Recent history length: {len(recent_history)} chars")

        # Phase 3: Analyze which issues are resolved
        self.logger.info(f"[IssueClosedAgent] Phase 3: Analyzing for resolved issues...")

        # Create prompt for analysis
        prompt = self._create_analysis_prompt(
            game_response, location, recent_history, tracked_issues_text
        )

        # Use structured output to get IssueClosedResponse
        analysis_chain = decision_llm.with_structured_output(IssueClosedResponse)

        # Invoke with timeout and retry
        from llm_utils import invoke_with_retry
        response = invoke_with_retry(
            analysis_chain.with_config(
                run_name=f"IssueClosedAgent: {location}"
            ),
            prompt,
            operation_name="IssueClosedAgent Analysis"
        )

        # Phase 4: STAGE the closures. Nothing is written here.
        #
        # This agent used to close issues immediately, which made a cancelled
        # turn corrupting: if the turn budget expired during `observe`, issues
        # were already closed while the turn's new issue and inventory changes
        # never landed, and the next session resumed from that half-applied
        # state (GitHub issue #3). Closures are now handed to persist_node,
        # which applies them last, after all cancellable LLM work.
        self.logger.info(f"[IssueClosedAgent] Phase 4: Staging closures for persist...")

        pending_closures = []
        if response.closed_issue_ids:
            for issue_id in response.closed_issue_ids:
                # Find the memory with this ID to get its content for display
                mem_content = None
                mem_importance = None
                for mem in tracked_issues:
                    if mem.id == issue_id:
                        mem_content = mem.content
                        mem_importance = mem.importance
                        break

                display = (
                    f"[ID:{issue_id}, {mem_importance}/1000] {mem_content}"
                    if mem_content and mem_importance is not None
                    else None
                )
                pending_closures.append({"id": issue_id, "display": display})
                self.logger.info(f"[IssueClosedAgent] STAGED close of ID {issue_id}: '{mem_content}'")
        else:
            self.logger.info(f"[IssueClosedAgent] No issues to close this turn")

        # closed_issue_contents stays empty until persist_node confirms the
        # writes: the report must show what was actually closed, not intended.
        response.closed_issue_contents = []

        # Log summary
        self.logger.info(f"[IssueClosedAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.logger.info(f"[IssueClosedAgent] SUMMARY")
        if pending_closures:
            self.logger.info(f"[IssueClosedAgent] ISSUES STAGED FOR CLOSE: {len(pending_closures)}")
            for closure in pending_closures:
                self.logger.info(f"[IssueClosedAgent]   - STAGED: '{closure['display'] or closure['id']}'")
            if response.reasoning:
                self.logger.info(f"[IssueClosedAgent]   Reasoning: {response.reasoning}")
        else:
            self.logger.info(f"[IssueClosedAgent] No issues closed this turn")
        self.logger.info(f"[IssueClosedAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return response, pending_closures

    def _create_analysis_prompt(
        self, game_response: str, location: str, recent_history: str, tracked_issues: str
    ) -> str:
        """Create the prompt for analyzing resolved issues"""
        return PromptLibrary.get_issue_closed_analysis_prompt(
            tracked_issues, recent_history, location, game_response
        )


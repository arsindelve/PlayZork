"""
IssueClosedAgent - Identifies resolved issues and removes them from memory.

This agent analyzes recent game history to determine if any tracked
strategic issues have been solved and should be removed from memory.

Responsibility: Aggressively close resolved issues to keep memory clean.
Runs BEFORE ObserverAgent to avoid confusion with stale issues.
"""
from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from .issue_closed_response import IssueClosedResponse
from tools.memory import MemoryToolkit
from tools.history import HistoryToolkit
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
    ) -> IssueClosedResponse:
        """
        Analyze recent history and close resolved issues.

        Args:
            game_response: The game's response after command execution
            location: Current location name
            score: Current game score
            moves: Current move count
            decision_llm: The LLM to use for analysis
            history_toolkit: HistoryToolkit for accessing recent turns
            memory_toolkit: MemoryToolkit for accessing and removing tracked issues

        Returns:
            IssueClosedResponse with list of closed issues
        """
        self.logger.info(f"[IssueClosedAgent] Analyzing recent history at {location}")

        # Phase 1: Get tracked issues
        self.logger.info(f"[IssueClosedAgent] Phase 1: Retrieving tracked issues...")
        tracked_issues = memory_toolkit.state.get_top_memories(limit=30)  # Get top 30

        if not tracked_issues:
            self.logger.info(f"[IssueClosedAgent] No tracked issues to analyze")
            return IssueClosedResponse(closed_issue_ids=[], closed_issue_contents=[], reasoning="No issues tracked yet.")

        tracked_issues_text = "\n".join([
            f"- [ID:{mem.id}, Importance:{mem.importance}/1000] {mem.content}"
            for mem in tracked_issues
        ])

        self.logger.info(f"[IssueClosedAgent] Found {len(tracked_issues)} tracked issues")

        # Phase 2: Gather recent history (last 5 turns)
        self.logger.info(f"[IssueClosedAgent] Phase 2: Gathering recent history...")

        # Use history toolkit to get recent turns
        recent_turns_tool = None
        for tool in history_toolkit.get_tools():
            if tool.name == "get_recent_turns":
                recent_turns_tool = tool
                break

        recent_history = ""
        if recent_turns_tool:
            self.logger.info(f"  -> get_recent_turns(n=5)")
            recent_history = recent_turns_tool.invoke({"n": 5})
            self.logger.info(f"     Result: {str(recent_history)[:150]}...")
            self.logger.info(f"[IssueClosedAgent] Recent history length: {len(recent_history)} chars")
        else:
            recent_history = "No recent history available."
            self.logger.warning(f"[IssueClosedAgent] get_recent_turns tool not found")

        # Phase 3: Analyze which issues are resolved
        self.logger.info(f"[IssueClosedAgent] Phase 3: Analyzing for resolved issues...")

        # Create prompt for analysis
        prompt = self._create_analysis_prompt(
            game_response, location, recent_history, tracked_issues_text
        )

        # Use structured output to get IssueClosedResponse
        analysis_chain = decision_llm.with_structured_output(IssueClosedResponse)

        # Invoke with timeout and retry
        from config import invoke_with_retry
        response = invoke_with_retry(
            analysis_chain.with_config(
                run_name=f"IssueClosedAgent: {location}"
            ),
            prompt,
            operation_name="IssueClosedAgent Analysis"
        )

        # Phase 4: Remove closed issues from memory
        self.logger.info(f"[IssueClosedAgent] Phase 4: Removing closed issues from memory...")

        closed_contents = []
        if response.closed_issue_ids:
            for issue_id in response.closed_issue_ids:
                self.logger.info(f"[IssueClosedAgent] Attempting to close issue ID: {issue_id}")

                # Find the memory with this ID to get its content for logging/display
                mem_content = None
                mem_importance = None
                for mem in tracked_issues:
                    if mem.id == issue_id:
                        mem_content = mem.content
                        mem_importance = mem.importance
                        break

                success = memory_toolkit.state.remove_memory(issue_id)
                if success:
                    self.logger.info(f"[IssueClosedAgent] [OK] REMOVED ID {issue_id}: '{mem_content}'")
                    # Add to closed_contents for display (include ID for debugging)
                    if mem_content and mem_importance is not None:
                        closed_contents.append(f"[ID:{issue_id}, {mem_importance}/1000] {mem_content}")
                else:
                    self.logger.warning(f"[IssueClosedAgent] [FAIL] Database removal failed for ID {issue_id}: '{mem_content}'")
        else:
            self.logger.info(f"[IssueClosedAgent] No issues closed this turn")

        # Update response with content strings for display
        response.closed_issue_contents = closed_contents

        self.logger.info(f"[IssueClosedAgent] Analysis complete:")
        self.logger.info(f"  Closed {len(response.closed_issue_ids)} issue(s)")
        self.logger.info(f"  Reasoning: {response.reasoning}")

        return response

    def _create_analysis_prompt(
        self, game_response: str, location: str, recent_history: str, tracked_issues: str
    ) -> str:
        """Create the prompt for analyzing resolved issues"""
        return f"""You are the IssueClosedAgent in a Zork-playing AI system.

YOUR SINGLE RESPONSIBILITY:
Analyze recent game history and identify which tracked issues have been SOLVED/RESOLVED.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENTLY TRACKED ISSUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{tracked_issues}

These issues are being tracked. Your job is to identify which ones are NOW SOLVED.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECENT GAME HISTORY (Last 5 Turns)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{recent_history}

Use this to see what actions were taken and their results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT TURN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Location: {location}
Game Response: {game_response}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT MAKES AN ISSUE "RESOLVED"?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Locked doors/gates → OPENED or UNLOCKED
✓ Items to collect → TAKEN and in inventory
✓ Obstacles/blockages → REMOVED or BYPASSED
✓ Puzzles → SOLVED (evidence of success)
✓ Exploration goals → COMPLETED (location fully explored)
✓ Containers (mailbox, chest) → OPENED and contents examined/taken

EXAMPLES:

Tracked: "Locked grating at Clearing - need key"
Recent history shows: "You unlock the grating with the key" or "The grating is now open"
→ CLOSE THIS ISSUE ✓

Tracked: "Small mailbox at West Of House"
Recent history shows: "You open the mailbox" AND "You take the leaflet"
→ CLOSE THIS ISSUE ✓

Tracked: "Troll blocking path at Troll Room"
Recent history shows: "The troll is defeated" or "The troll has left"
→ CLOSE THIS ISSUE ✓

Tracked: "Locked door in Living Room"
Recent history shows: "The door is still locked" or no mention
→ DO NOT CLOSE (still unresolved)

Tracked: "Sword on table in Living Room"
Recent history shows: "You take the sword"
→ CLOSE THIS ISSUE ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLOSING STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BE AGGRESSIVE: If there's clear evidence an issue is resolved, CLOSE IT.
- Better to close and reopen later than keep stale issues
- If an item was taken, close it
- If a lock was opened, close it
- If a puzzle was solved, close it

ONLY KEEP OPEN:
- Issues with NO evidence of resolution in recent history
- Ongoing obstacles still blocking progress
- Partially-solved puzzles (e.g., found key but haven't used it yet)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return JSON with:
- closed_issue_ids: List of memory IDs (integers) for issues that should be closed
- reasoning: Brief explanation of why these issues were closed

CRITICAL: Return the ID number from each tracked issue.
Example: If tracked issue is "- [ID:5, Importance:405/1000] Climbable cliff above Rocky Ledge"
Return in closed_issue_ids: 5

Example output:
{{
  "closed_issue_ids": [5, 12],
  "reasoning": "Issue ID 5 (mailbox) was opened and leaflet taken. Issue ID 12 (grating) was unlocked and opened."
}}

If no issues should be closed:
{{
  "closed_issue_ids": [],
  "reasoning": "No tracked issues have been resolved in recent history."
}}

Analyze the tracked issues against recent history and identify which to close:
"""

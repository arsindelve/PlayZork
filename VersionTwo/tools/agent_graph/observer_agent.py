"""
ObserverAgent - Identifies new strategic issues from game responses.

This agent analyzes the game's response after a command is executed
and identifies any new puzzles, obstacles, or items that should be
tracked for future turns.

Responsibility: Single-purpose observer that ONLY identifies what's new.
Does NOT make decisions about what command to execute.
"""
from typing import Optional, List
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from .observer_response import ObserverResponse
from tools.memory import MemoryToolkit
import logging


class ObserverAgent:
    """
    Analyzes game responses to identify new strategic issues.

    This agent is called AFTER:
    - Decision Agent chooses a command
    - Command is executed
    - Game response is received

    Its job: Identify anything NEW in the game response that should be tracked.
    """

    def __init__(self):
        """Initialize the ObserverAgent"""
        self.logger = logging.getLogger(__name__)
        self.remember = None
        self.rememberImportance = None
        self.item = None

    def observe(
        self,
        game_response: str,
        location: str,
        score: int,
        moves: int,
        decision_llm: BaseChatModel,
        research_agent,
        history_tools: List[BaseTool],
        memory_toolkit: MemoryToolkit
    ) -> ObserverResponse:
        """
        Analyze the game response and identify new strategic issues.

        Args:
            game_response: The game's response after command execution
            location: Current location name
            score: Current game score
            moves: Current move count
            decision_llm: The LLM to use for analysis
            research_agent: Research agent with access to history tools
            history_tools: List of history tools for context gathering
            memory_toolkit: MemoryToolkit for accessing already-tracked issues

        Returns:
            ObserverResponse with remember, rememberImportance, item
        """
        self.logger.info(f"[ObserverAgent] Analyzing game response at {location}")

        # Phase 0: Get already-tracked issues to avoid duplicates
        self.logger.info(f"[ObserverAgent] Phase 0: Retrieving tracked issues...")
        tracked_issues = memory_toolkit.state.get_top_memories(limit=20)  # Get top 20 tracked issues
        tracked_issues_text = "\n".join([
            f"- [{mem.importance}/1000] {mem.content}"
            for mem in tracked_issues
        ]) if tracked_issues else "No issues tracked yet."

        self.logger.info(f"[ObserverAgent] Found {len(tracked_issues)} tracked issues")

        # Phase 1: Gather historical context using research agent
        self.logger.info(f"[ObserverAgent] Phase 1: Gathering historical context...")
        research_input = {
            "input": "Use get_full_summary to see what has been discovered so far. Use get_recent_turns to see recent history.",
            "score": score,
            "locationName": location,
            "moves": moves,
            "game_response": game_response
        }

        from config import invoke_with_retry
        research_response = invoke_with_retry(
            research_agent.with_config(
                run_name=f"Observer Research: {location}"
            ),
            research_input,
            operation_name="Observer Research"
        )

        # Execute tool calls to get historical context
        historical_context = ""
        if hasattr(research_response, 'tool_calls') and research_response.tool_calls:
            tool_results = []
            tools_map = {tool.name: tool for tool in history_tools}

            self.logger.info(f"[ObserverAgent] Made {len(research_response.tool_calls)} tool calls:")

            for tool_call in research_response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call.get('args', {})

                self.logger.info(f"  -> {tool_name}({tool_args})")

                if tool_name in tools_map:
                    tool_result = tools_map[tool_name].invoke(tool_args)
                    self.logger.info(f"     Result: {str(tool_result)[:150]}...")
                    tool_results.append(f"{tool_name}: {tool_result}")

            historical_context = "\n\n".join(tool_results) if tool_results else "No historical context available."
        else:
            self.logger.info(f"[ObserverAgent] No tool calls made")
            historical_context = "No historical context retrieved."

        self.logger.info(f"[ObserverAgent] Historical context length: {len(historical_context)} chars")

        # Phase 2: Analyze game response with historical context and tracked issues
        self.logger.info(f"[ObserverAgent] Phase 2: Analyzing for new issues...")

        # Create prompt for observation with full context
        prompt = self._create_observation_prompt(game_response, location, historical_context, tracked_issues_text)

        # Use structured output to get ObserverResponse
        observation_chain = decision_llm.with_structured_output(ObserverResponse)

        # Invoke with timeout and retry
        response = invoke_with_retry(
            observation_chain.with_config(
                run_name=f"Observer Agent: {location}"
            ),
            prompt,
            operation_name="Observer Agent Analysis"
        )

        # Store findings
        self.remember = response.remember
        self.rememberImportance = response.rememberImportance
        self.item = response.item

        self.logger.info(f"[ObserverAgent] Observation complete:")
        self.logger.info(f"  remember: '{response.remember}'")
        self.logger.info(f"  importance: {response.rememberImportance}")
        self.logger.info(f"  item: '{response.item}'")

        return response

    def _create_observation_prompt(self, game_response: str, location: str, historical_context: str, tracked_issues: str) -> str:
        """Create the prompt for game response observation"""
        return f"""You are the Observer Agent in a Zork-playing AI system.

YOUR SINGLE RESPONSIBILITY:
Analyze the game response and identify ANY new strategic issues, puzzles, or obstacles to track.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALREADY TRACKED ISSUES (DO NOT DUPLICATE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

These issues are ALREADY being tracked. Do NOT add duplicates:

{tracked_issues}

CRITICAL: If the game response mentions something already in this list, leave 'remember' EMPTY.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HISTORICAL CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{historical_context}

Use this context to understand what has been seen before.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO IDENTIFY (ONLY IF NEW)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ NEW puzzles or obstacles:
  - Locked doors, gates, gratings
  - Blocking entities (trolls, guards, etc.)
  - Environmental hazards (darkness, chasms, etc.)

✓ NEW items or objects:
  - Items mentioned in the description
  - Objects that can be interacted with
  - Tools or keys that might solve puzzles

✓ NEW opportunities:
  - Mechanisms or switches discovered
  - Clues about puzzle solutions

✗ DO NOT TRACK (ExplorerAgent handles these):
  - Blocked paths ("You can't go that way")
  - New directions or exits mentioned
  - Movement confirmations
  - Simple location descriptions

EXAMPLES:

Game: "There is a small mailbox here."
Tracked issues: (empty)
→ remember: "Small mailbox at West Of House"
→ rememberImportance: 700
→ item: "mailbox"

Game: "There is a small mailbox here."
Tracked issues: "Small mailbox at West Of House"
→ remember: ""  ← EMPTY because already tracked!
→ rememberImportance: None
→ item: ""

Game: "In disturbing the pile of leaves, a grating is revealed."
Tracked issues: "Pile of leaves in Clearing"
→ remember: "Grating revealed in pile of leaves at Clearing"  ← NEW discovery
→ rememberImportance: 800
→ item: "grating"

Game: "The grating is locked."
Tracked issues: "Grating revealed in pile of leaves at Clearing"
→ remember: "Locked grating at Clearing - need key"  ← NEW information about existing issue
→ rememberImportance: 900
→ item: ""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANCE SCORING (1-1000)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 900-1000: Critical obstacles (locked gates blocking main path, trolls)
- 700-800: Important items or doors (keys, treasures, entry points)
- 500-600: Interesting objects to investigate (piles, chests, mechanisms)
- 300-400: Minor items or flavor objects

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Check TRACKED ISSUES first - if already listed, leave 'remember' EMPTY
2. Only track TRULY NEW discoveries that require puzzle-solving
3. DO NOT track blocked paths, new exits, or directions (ExplorerAgent handles exploration)
4. DO NOT track general location descriptions or movement confirmations
5. ONLY track items, obstacles, or puzzles that need solving
6. Include location name in 'remember' field for context

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Location: {location}
Game Response: {game_response}

Analyze this response and output JSON with:
- remember: NEW strategic issue (or empty string if already tracked or nothing new)
- rememberImportance: Importance score 1-1000 (or null if remember empty)
- item: Any item mentioned (or empty string)
"""

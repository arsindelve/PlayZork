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
from adventurer.prompt_library import PromptLibrary
from .tool_execution import invoke_tool_safely
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

    async def observe(
        self,
        game_response: str,
        location: str,
        score: int,
        moves: int,
        decision_llm: BaseChatModel,
        memory_toolkit: MemoryToolkit,
        context=None,
    ) -> ObserverResponse:
        """
        Analyze the game response and identify new strategic issues.

        Args:
            game_response: The game's response after command execution
            location: Current location name
            score: Current game score
            moves: Current move count
            decision_llm: The LLM to use for analysis
            history_tools: List of history tools for context gathering
            memory_toolkit: MemoryToolkit for accessing already-tracked issues

        Returns:
            ObserverResponse with remember, rememberImportance, item
        """
        self.logger.info(f"[ObserverAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.logger.info(f"[ObserverAgent] AGENT: ObserverAgent")
        self.logger.info(f"[ObserverAgent] PURPOSE: Identify new strategic issues from game responses")
        self.logger.info(f"[ObserverAgent] LOCATION: {location}")
        self.logger.info(f"[ObserverAgent] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
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

        # Historical context comes from the TurnContext, assembled in code.
        # This was a full LLM round-trip that asked the model to call
        # get_full_summary / get_recent_turns, then executed whatever came
        # back against a map holding only 2 of the 8 bound tools (#5) — so
        # most of what it asked for was silently dropped and the Observer
        # decided what to persist to long-term memory with
        # "No historical context available." (#25).
        historical_context = (
            context.research_context_for() if context is not None
            else "No historical context retrieved."
        )

        self.logger.info(f"[ObserverAgent] Historical context length: {len(historical_context)} chars")

        # Phase 2: Analyze game response with historical context and tracked issues
        self.logger.info(f"[ObserverAgent] Phase 2: Analyzing for new issues...")

        # Create prompt for observation with full context
        prompt = self._create_observation_prompt(game_response, location, historical_context, tracked_issues_text)

        # Use structured output to get ObserverResponse
        # Function-local import: tests monkeypatch llm_utils.invoke_with_retry
        # Async: a timeout must cancel the request rather than leak the
        # thread and retry alongside it (#26, #27).
        from llm_utils import ainvoke_with_retry
        observation_chain = decision_llm.with_structured_output(ObserverResponse)

        # Invoke with timeout and retry
        response = await ainvoke_with_retry(
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
        return PromptLibrary.get_observer_observation_prompt(
            game_response, location, historical_context, tracked_issues
        )

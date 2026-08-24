"""The experiment's control arm: one LLM call, full history in context.

The project's original motivation was external memory for small-context
models. That motivation is obsolete — a whole playthrough now fits in context —
so the standing research question is instead:

    Can multi-agent deliberation let a much weaker model solve long-horizon
    tasks that it cannot solve by direct inference?

Answering that needs a control, and this is it: the *same model*, given the
*same information*, asked for a command in a single inference. No advocacy
agents, no arbiter, no per-issue reasoning. If this plays as well as the
architecture, the architecture is not earning its cost.

Deliberately generous to the baseline. It receives the full long-running
summary, recent turns, inventory, the map and known exits — everything the
multi-agent arm's TurnContext assembles. A weak baseline would make the
comparison meaningless, and the honest experiment is the one where the control
is given every advantage the treatment has.

What it does NOT get is the deliberation itself: no competing proposals, no
expected-value arbitration, no per-issue pathfinding. That difference is the
independent variable.

Implements the same interface as AdventurerService so GameSession can run
either arm without knowing which, and every downstream consumer — reports,
display, persistence, token accounting — keeps working unchanged.
"""
import asyncio
from typing import List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from config import TURN_BUDGET_SECONDS, get_expensive_llm
from game_logger import GameLogger
from tools.agent_graph.turn_context import build_turn_context
from zork.zork_api_response import ZorkApiResponse

from .adventurer_response import AdventurerResponse
from .prompt_library import PromptLibrary


class SingleShotService:
    """One inference per turn, with everything in context."""

    def __init__(self, history_toolkit, memory_toolkit, mapper_toolkit, inventory_toolkit):
        self.history_toolkit = history_toolkit
        self.memory_toolkit = memory_toolkit
        self.mapper_toolkit = mapper_toolkit
        self.inventory_toolkit = inventory_toolkit
        self.logger = GameLogger.get_instance()

        # The SAME model tier the multi-agent arm uses for its decision, so the
        # comparison isolates architecture rather than model quality.
        self.llm = get_expensive_llm(temperature=0)
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", PromptLibrary.get_single_shot_system_prompt()),
                ("human", PromptLibrary.get_single_shot_human_prompt()),
            ])
            | self.llm.with_structured_output(AdventurerResponse)
        )

    async def handle_user_input(
        self,
        last_game_response: ZorkApiResponse,
        turn_number: int,
        player_command: str,
    ) -> Tuple:
        """Choose the next command in a single inference.

        Returns the same shape as AdventurerService.handle_user_input, with
        the agent slots empty — there are no agents in this arm.
        """
        from llm_utils import ainvoke_with_retry

        context = build_turn_context(
            game_response=last_game_response,
            history_toolkit=self.history_toolkit,
            mapper_toolkit=self.mapper_toolkit,
            inventory_toolkit=self.inventory_toolkit,
            issue_locations=None,
        )

        tracked = ""
        try:
            memories = self.memory_toolkit.state.get_top_memories(
                limit=5, current_turn=turn_number
            )
            tracked = "\n".join(
                f"- [{m.importance}/1000] {m.content}" for m in memories
            ) or "None tracked yet."
        except Exception as e:
            self.logger.logger.warning(f"[SingleShot] tracked issues unavailable: {e}")
            tracked = "Unavailable."

        try:
            known_map = self.mapper_toolkit.state.get_all_transitions()
            map_text = "\n".join(
                f"  {t.from_location} --[{t.direction}]--> {t.to_location}"
                for t in known_map
            ) or "Nothing mapped yet."
        except Exception:
            map_text = "Unavailable."

        inputs = {
            "locationName": context.location,
            "score": context.score,
            "moves": context.moves,
            "game_response": context.game_text,
            "inventory": context.inventory_summary,
            "exits": context.exits_summary,
            "already_tried": context.unproductive_summary,
            "tracked_issues": tracked,
            "known_map": map_text,
            "recent_turns": context.recent_turns or "This is the first turn.",
            "full_summary": context.full_summary or "Nothing yet.",
            "long_summary": context.long_summary or "Nothing yet.",
        }

        self.logger.logger.info(
            "\n" + "=" * 80
            + f"\nSINGLE-SHOT BASELINE - one inference, full context (turn {turn_number})\n"
            + "=" * 80
        )

        decision = await asyncio.wait_for(
            ainvoke_with_retry(
                self.chain.with_config(run_name=f"Single-Shot Baseline: Turn {turn_number}"),
                inputs,
                operation_name=f"Single-Shot Baseline: Turn {turn_number}",
            ),
            timeout=TURN_BUDGET_SECONDS,
        )

        self.logger.log_decision(decision.command, decision.reason)

        prompt_for_report = (
            f"[SINGLE-SHOT BASELINE]\n"
            + "\n".join(f"{k}: {v}" for k, v in inputs.items())
        )

        # No agents, no issue closing, no observer: this arm is deliberately
        # just the one call. Empty slots keep every downstream consumer working.
        return (decision, [], None, None, None, None, None, prompt_for_report, [], [])

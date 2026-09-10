"""EXPERIMENT (2026-09-10) — a PASSIVE shadow agent. It changes nothing about play.

Each turn it decides ONE thing: does what it now sees warrant creating a NEW goal
(not already tracked), and if so, what CLOSING CONDITION marks it achieved? It logs
the decision and does nothing else — it never contributes a proposal to the arbiter.

Purpose: watch, turn by turn, what goals qwen proposes under the real game, and
whether it shows restraint (declines) given the goals it has already proposed. This
probes the admission / dedup question from docs/GOAL_AGENTS_PROPOSAL.md ("when to make
a goal, not which"). It is deliberately NOT wired into gameplay and enforces no budget.

Output: logs/goal_experiment_<session>.jsonl (one record per turn) + readable log
lines prefixed [GoalExperiment]. The JSONL is also the agent's memory of what it has
already proposed, so "new" is judged against the running list.
"""
import json
import logging
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from config import GAME_OBJECTIVE, SESSION_ID

logger = logging.getLogger(__name__)
_LOG_DIR = Path("logs")


class GoalDecision(BaseModel):
    """The shadow agent's per-turn judgement."""
    new_goal: bool = Field(
        description="True ONLY if this turn warrants a NEW goal not already tracked.")
    goal: str = Field(
        "", description="If new_goal: the target STATE to reach (a condition that will "
        "become true), NOT a command. Empty otherwise.")
    closing_condition: str = Field(
        "", description="If new_goal: how you will know it is achieved — a checkable "
        "condition on game state (location/score/inventory), NOT an action. Empty otherwise.")
    reason: str = Field(
        "", description="One sentence: why this is, or is not, a new goal worth tracking.")


_INSTRUCTIONS = (
    "You are a GOAL-TRACKING observer for the text adventure Zork. You do NOT choose "
    "the game's moves. Each turn you decide ONE thing: does what you now see warrant "
    "creating a NEW goal to track, and if so, how will you know it is achieved?\n\n"
    "A GOAL is a target STATE the player wants to reach — a condition that will become "
    "true — NOT a command. Good: 'I am inside the white house'. Bad: 'open the window'.\n"
    "A CLOSING CONDITION is how you will know the goal is achieved: a concrete, "
    "checkable condition on the game state (location, score, inventory), NOT an action. "
    "e.g. 'the player is in a room inside the house', 'the score has increased', "
    "'the jeweled egg is in the inventory'.\n\n"
    "ALTITUDE: a goal must be a MEANINGFUL MILESTONE — a place reached, an item "
    "obtained, an obstacle removed — never the mere operation of one object, and never "
    "the entire objective.\n\n"
    "BE SPARING. Most turns do NOT warrant a new goal. Do NOT create a goal that:\n"
    "  - duplicates, or is already covered by, one in the tracked list;\n"
    "  - is mere scenery or flavour with no bearing on the objective;\n"
    "  - is just the next physical action of a goal you already track.\n"
    "Only create a goal for a genuine NEW obstacle, opportunity, or objective-relevant "
    "milestone. If nothing here warrants a new goal, set new_goal=false and say why.\n\n"
    "Give: new_goal (true/false); if true, goal + closing_condition; and a one-sentence "
    "reason either way."
)


class GoalExperimentAgent:
    def __init__(self, decision_llm, session_id: str = SESSION_ID):
        self.llm = decision_llm
        self.path = _LOG_DIR / f"goal_experiment_{session_id}.jsonl"

    def _prior_goals(self) -> List[dict]:
        """Goals this agent has already proposed, from its own JSONL log."""
        if not self.path.exists():
            return []
        goals = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("new_goal"):
                goals.append({"goal": rec.get("goal", ""),
                              "closing": rec.get("closing_condition", "")})
        return goals

    def _build_prompt(self, context, prior_goals: List[dict]) -> str:
        if prior_goals:
            prior_text = "\n".join(
                f"  - GOAL: {g['goal']}  (closes when: {g['closing']})"
                for g in prior_goals)
        else:
            prior_text = "  (none yet)"
        return (
            f"{_INSTRUCTIONS}\n\n"
            f"OBJECTIVE: {GAME_OBJECTIVE}\n\n"
            f"CURRENT ROOM: {context.location}\n"
            f"DESCRIPTION: {context.game_text.strip()}\n"
            f"EXITS THE GAME REPORTS: {context.game_exits or 'unknown'}\n"
            f"COMMANDS THE GAME ACCEPTS HERE:\n{context.available_actions_summary}\n"
            f"INVENTORY: {context.inventory_summary}\n"
            f"SCORE: {context.score}\n\n"
            f"GOALS ALREADY BEING TRACKED:\n{prior_text}"
        )

    async def run(self, context) -> None:
        """One passive judgement. Never raises — a failed turn just logs nothing."""
        try:
            prior = self._prior_goals()
            prompt = self._build_prompt(context, prior)
            from llm_utils import ainvoke_with_retry
            chain = self.llm.with_structured_output(GoalDecision)
            result = await ainvoke_with_retry(
                chain.with_config(run_name=f"GoalExperiment: {context.location}"),
                prompt,
                operation_name="Goal Experiment",
            )
            self._log(context, result, len(prior))
        except Exception as e:
            logger.warning(f"[GoalExperiment] failed this turn: {e}")

    def _log(self, context, result: GoalDecision, prior_count: int) -> None:
        rec = {
            "moves": context.moves,
            "location": context.location,
            "score": context.score,
            "new_goal": bool(result.new_goal),
            "goal": result.goal or "",
            "closing_condition": result.closing_condition or "",
            "reason": result.reason or "",
            "goals_tracked_before": prior_count,
        }
        _LOG_DIR.mkdir(exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if result.new_goal:
            logger.info(
                f"[GoalExperiment] +NEW GOAL @ {context.location} (move {context.moves}): "
                f"'{result.goal}' | closes when: '{result.closing_condition}' | {result.reason}")
        else:
            logger.info(
                f"[GoalExperiment] no new goal @ {context.location} (move {context.moves}) "
                f"| {result.reason}")

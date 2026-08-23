from langchain_core.language_models import BaseChatModel
from typing import List

from .history_state import HistoryState, GameTurn
from .history_summarizer import HistorySummarizer
from .history_tools import initialize_history_tools, get_history_tools
from tools.database import DatabaseManager


class HistoryToolkit:
    """
    Facade for the history tool system.
    Manages state, summarization, and provides tools for LangChain agents.
    Now with SQLite persistence.
    """

    def __init__(self, summarizer_llm: BaseChatModel, session_id: str, db: DatabaseManager):
        """
        Initialize the history toolkit with database backend

        Args:
            summarizer_llm: LLM to use for generating summaries (should be cheap)
            session_id: Unique identifier for this game session
            db: DatabaseManager instance for persistence
        """
        self.state = HistoryState(session_id=session_id, db=db)
        # Created lazily: HistoryToolkit is built before the event loop exists.
        self._summary_lock = None
        self.summarizer = HistorySummarizer(summarizer_llm)

        # Initialize the module-level state for tools
        initialize_history_tools(self.state)

    def record_turn(self,
                    game_response: str,
                    player_command: str,
                    location: str,
                    score: int,
                    moves: int):
        """Persist the turn. Fast, synchronous, and MUST stay on the critical
        path — `get_recent_turns` reads this, and the agents research against
        it during the very turn it is written.

        Split out from summarization (GitHub issue #24 option 2): the turn
        record is what this turn's agents actually need; folding the newest
        turn into the prose summaries buys almost nothing for the decision
        being made right now, and cost 65s+ at the head of every turn.
        """
        return self.state.add_turn(
            game_response=game_response,
            player_command=player_command,
            location=location,
            score=score,
            moves=moves,
        )

    async def refresh_summaries(self, turn) -> None:
        """Regenerate both summaries for `turn` and commit them together.

        Coalesced under a lock: turns can overlap once the engine gets faster,
        and two summarizers racing would let an older result overwrite a newer
        one through `save_both_summaries`.
        """
        import asyncio
        import logging
        import time
        logger = logging.getLogger(__name__)

        # getattr, not attribute access: the lock is created lazily (the
        # toolkit is built before an event loop exists) and callers may
        # construct the toolkit without running __init__.
        if getattr(self, "_summary_lock", None) is None:
            self._summary_lock = asyncio.Lock()

        async with self._summary_lock:
            try:
                logger.info(
                    f"Generating recent + long-running summaries CONCURRENTLY "
                    f"(turn {turn.turn_number}, off the critical path)..."
                )
                started = time.monotonic()

                new_recent_summary, new_long_summary = await asyncio.gather(
                    self.summarizer.agenerate_summary(self.state, turn),
                    self.summarizer.agenerate_long_running_summary(self.state, turn),
                    return_exceptions=True,
                )
                elapsed = time.monotonic() - started

                for label, result in (
                    ("recent", new_recent_summary),
                    ("long-running", new_long_summary),
                ):
                    if isinstance(result, BaseException):
                        logger.error(
                            f"{label} summary failed after {elapsed:.1f}s: {result}",
                            exc_info=result,
                        )

                if isinstance(new_recent_summary, BaseException) or isinstance(
                    new_long_summary, BaseException
                ):
                    logger.error("Summaries NOT saved this turn (see errors above)")
                    return

                logger.info(f"Both summaries generated in {elapsed:.1f}s (concurrent)")
                self.state.save_both_summaries(new_recent_summary, new_long_summary)
                logger.info("Both summaries saved to database")

            except Exception as e:
                logger.error(f"ERROR in refresh_summaries: {e}", exc_info=True)

    async def update_after_turn(self,
                                game_response: str,
                                player_command: str,
                                location: str,
                                score: int,
                                moves: int) -> None:
        """
        Update history after a game turn completes.
        This should be called by the game loop after each turn.

        The two summaries are generated CONCURRENTLY (GitHub issue #24). They
        are independent — each reads its own previous summary from the DB and
        uses its own prompt, and they only meet at `save_both_summaries` — but
        they used to run serially at the head of every turn, blocking all agent
        work behind them (86s measured on turn 1, 113s for the recent summary
        alone by turn 2).

        Args:
            game_response: Text response from the game
            player_command: Command issued by the player/agent
            location: Current location name
            score: Current game score
            moves: Current move count
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            turn = self.record_turn(
                game_response=game_response,
                player_command=player_command,
                location=location,
                score=score,
                moves=moves,
            )
            logger.info(f"Added turn {turn.turn_number}: {player_command}")
            await self.refresh_summaries(turn)
        except Exception as e:
            logger.error(f"ERROR in update_after_turn: {e}", exc_info=True)

    def get_tools(self) -> List:
        """
        Get the list of history tools for use with LangChain agents

        Returns:
            List of tool functions that agents can call
        """
        return get_history_tools()


# Export public API
__all__ = ['HistoryToolkit', 'HistoryState', 'GameTurn']

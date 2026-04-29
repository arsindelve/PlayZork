import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Any

from adventurer.adventurer_service import AdventurerService
from zork.zork_service import ZorkService
from tools.history import HistoryToolkit
from tools.memory import MemoryToolkit
from tools.mapping import MapperToolkit
from tools.database import DatabaseManager
from display_manager import DisplayManager
from game_logger import GameLogger
from config import get_cheap_llm, get_expensive_llm


@dataclass
class PendingDecision:
    """Stores all decision data from the previous turn for report synchronization.

    The HTML report should show the agents/decision that LED to the current command,
    not the agents/decision planning the NEXT command. This dataclass holds all that
    state so it can be used when writing the report for the NEXT turn.
    """
    reasoning: Optional[str] = None
    issue_agents: List[Any] = field(default_factory=list)
    explorer_agent: Any = None
    loop_detection_agent: Any = None
    interaction_agent: Any = None
    decision_prompt: str = ""
    decision: Any = None
    research_tool_calls: List[dict] = field(default_factory=list)
    decision_tool_calls: List[dict] = field(default_factory=list)


class GameSession:

    def __init__(self, session_id: str):
        """
        Initialize the game session with a session ID
        and the required services.
        """
        self.session_id = session_id

        # Initialize logger (will reset log file for this session)
        self.logger = GameLogger.get_instance(session_id)
        self.logger.logger.info(f"Initializing GameSession with ID: {session_id}")

        # Initialize database for persistent storage
        self.db = DatabaseManager()
        self.db.create_session(session_id)
        self.logger.logger.info(f"Created database session: {session_id}")

        self.zork_service = ZorkService(session_id=session_id)

        # Create cheap LLM for de-duplication
        cheap_llm = get_cheap_llm(temperature=0)

        # Create expensive LLM for history summarization (needs quality)
        expensive_llm = get_expensive_llm(temperature=0)

        # Create history toolkit with expensive LLM for better summarization
        self.history_toolkit = HistoryToolkit(expensive_llm, session_id, self.db)

        # Create memory toolkit with cheap LLM for de-duplication (write-only strategic issue storage)
        self.memory_toolkit = MemoryToolkit(session_id, self.db, cheap_llm)

        # Create mapper toolkit for tracking location transitions
        self.mapper_toolkit = MapperToolkit(session_id, self.db)
        self.logger.logger.info("Mapper toolkit initialized")

        # Create inventory toolkit for tracking items
        from tools.inventory import InventoryToolkit
        self.inventory_toolkit = InventoryToolkit(session_id, self.db)
        self.logger.logger.info("Inventory toolkit initialized")

        # Initialize analysis tools so agents can access big picture analysis
        from tools.analysis import initialize_analysis_tools
        initialize_analysis_tools(session_id, self.db)
        self.logger.logger.info("Analysis tools initialized")

        # Pass toolkits to adventurer service
        self.adventurer_service = AdventurerService(
            self.history_toolkit,
            self.memory_toolkit,
            self.mapper_toolkit,
            self.inventory_toolkit
        )

        # Resume turn numbering from where this session left off
        last_turn = self.db.get_latest_turn_number(session_id)
        self.turn_number = last_turn if last_turn is not None else 0
        self.logger.logger.info(f"Resuming from turn {self.turn_number}")

        # Track decision data from previous turn (to display with correct command in report)
        # The report should show agents/decision that LED to the current command
        self.pending_decision = PendingDecision()

        # Post-decision analyzers + report writer run as background tasks so the
        # next turn doesn't wait on them. Tracked here so we can drain them at
        # shutdown and surface any failures.
        self._background_tasks: List[asyncio.Task] = []

    async def play(self):
        """
        Main gameplay loop. Runs indefinitely until interrupted.
        Press Ctrl+C to quit.
        """
        # Create display manager with Rich
        display = DisplayManager()
        try:
            # Initialize the game state.
            await self.zork_service.play_turn("verbose")

            # Bootstrap inventory from game INVENTORY command
            await self._bootstrap_inventory()

            adventurer_response = await self.__play_turn("look", display)

            # Run indefinitely until user interrupts
            while True:
                adventurer_response = await self.__play_turn(adventurer_response, display)

        except KeyboardInterrupt:
            # Clean exit on Ctrl+C - let main.py handle the message
            raise
        except Exception as e:
            print(f"\nAn error occurred during gameplay: {e}")
        finally:
            # Drain pending post-turn analyzers/reports so we don't lose the
            # final turns' output. Failures are logged but do not block exit.
            await self._drain_background_tasks()
            # Always stop the display cleanly
            display.stop()

    async def __play_turn(self, input_text: str, display: DisplayManager) -> str:
        """
        Play a single turn of the game.
        :param input_text: The player's input command (e.g., "open door").
        :param display: The DisplayManager instance for updating the UI
        :return: The next input to be processed.
        """
        try:
            self.turn_number += 1
            self.logger.log_turn_start(self.turn_number, input_text)

            # Step 1: Send input to the Zork service and get the response
            zork_response = await self.zork_service.play_turn(input_text=input_text)
            self.logger.log_game_response(zork_response.Response)

            # Step 2: Update history BEFORE decision (so research sees current turn)
            self.history_toolkit.update_after_turn(
                game_response=zork_response.Response,
                player_command=input_text,  # The command that was just executed
                location=zork_response.LocationName,
                score=zork_response.Score,
                moves=zork_response.Moves
            )

            # Step 2b: Update mapper to track location transitions
            self.mapper_toolkit.update_after_turn(
                current_location=zork_response.LocationName,
                player_command=input_text,
                turn_number=self.turn_number
            )

            # Step 3: Process through LangGraph (Research → Decide → CloseIssues → Observe → Persist)
            # The graph handles: research, decision, issue closing, observation, and memory persistence
            (player_response, issue_agents, explorer_agent, loop_detection_agent, interaction_agent,
             issue_closed_response, observer_response, decision_prompt,
             research_tool_calls, decision_tool_calls) = await self.adventurer_service.handle_user_input(
                zork_response,
                self.turn_number
            )

            # Extract closed issues and new issue from responses
            closed_issues = issue_closed_response.closed_issue_contents if issue_closed_response else []
            new_issue = observer_response.remember if observer_response and observer_response.remember else None

            # Step 4: Update display with the turn
            # Use pending_decision from PREVIOUS turn (agents/reasoning that led to this command)
            # Capture it before updating so we can use it in the report later
            decision_for_this_command = self.pending_decision
            display.add_turn(
                location=zork_response.LocationName,
                game_text=zork_response.Response,
                command=input_text,  # The command that was just executed
                score=zork_response.Score,
                moves=zork_response.Moves,
                reasoning=decision_for_this_command.reasoning,  # Reasoning for THIS command (from previous turn)
                closed_issues=closed_issues,  # Issues that were resolved this turn
                new_issue=new_issue  # New issue identified this turn
            )

            # Store decision data for the NEXT turn's report
            self.pending_decision = PendingDecision(
                reasoning=player_response.reason,
                issue_agents=issue_agents,
                explorer_agent=explorer_agent,
                loop_detection_agent=loop_detection_agent,
                interaction_agent=interaction_agent,
                decision_prompt=decision_prompt,
                decision=player_response,
                research_tool_calls=research_tool_calls or [],
                decision_tool_calls=decision_tool_calls or []
            )

            # Step 5: Update display with current summaries
            recent_summary = self.history_toolkit.state.get_full_summary()
            long_summary = self.history_toolkit.state.get_long_running_summary()
            display.update_summary(recent_summary, long_summary)
            self.logger.log_summary_update(recent_summary)

            # Step 6: Update display with agents (formatting handled by DisplayManager)
            display.update_agents(issue_agents, explorer_agent, loop_detection_agent, interaction_agent)

            # Step 7: Update display with map (formatting handled by DisplayManager)
            transitions = self.mapper_toolkit.state.get_all_transitions()
            display.update_map_from_transitions(transitions)

            # Snapshot current inventory once for the background task to use.
            current_inventory = self.inventory_toolkit.state.get_items()

            # Step 7b–8: Big-picture analysis, death analysis, turn report, and
            # session index are all downstream of the decision and read by no
            # later turn (except strategic_analysis, which tolerates a one-turn
            # lag through its existing "latest" lookup). Dispatch them as a
            # background task so the next turn doesn't wait on this LLM + I/O.
            self._dispatch_post_turn_io(
                turn_number=self.turn_number,
                input_text=input_text,
                zork_response=zork_response,
                decision_for_this_command=decision_for_this_command,
                recent_summary=recent_summary,
                long_summary=long_summary,
                current_inventory=current_inventory,
                transitions=transitions,
            )

            return player_response.command

        except Exception as e:
            self.logger.log_error(str(e))
            print(f"\nAn error occurred while processing turn: {e}")
            raise  # Re-raise to be caught by play() method

    def _dispatch_post_turn_io(
        self,
        *,
        turn_number: int,
        input_text: str,
        zork_response,
        decision_for_this_command: PendingDecision,
        recent_summary: str,
        long_summary: str,
        current_inventory: list,
        transitions,
    ) -> None:
        """Schedule the post-decision analyzers + report writer as a background task.

        Inputs are captured by value so the next turn can mutate state freely
        while this task runs. Sync work is offloaded onto a worker thread via
        asyncio.to_thread so it doesn't block the event loop.
        """
        task = asyncio.create_task(
            asyncio.to_thread(
                self._run_post_turn_io,
                turn_number,
                input_text,
                zork_response,
                decision_for_this_command,
                recent_summary,
                long_summary,
                current_inventory,
                transitions,
            )
        )
        task.add_done_callback(self._on_background_task_done)
        self._background_tasks.append(task)

    def _run_post_turn_io(
        self,
        turn_number: int,
        input_text: str,
        zork_response,
        decision_for_this_command: PendingDecision,
        recent_summary: str,
        long_summary: str,
        current_inventory: list,
        transitions,
    ) -> None:
        """Synchronous post-turn work: big-picture analysis, death analysis,
        turn report, session index. Runs on a worker thread."""
        from tools.analysis import BigPictureAnalyzer, DeathAnalyzer
        from tools.reporting import TurnReportWriter

        big_picture_analyzer = BigPictureAnalyzer(
            self.history_toolkit,
            self.session_id,
            self.db,
            current_inventory=current_inventory,
            current_location=zork_response.LocationName,
        )
        big_picture_analysis = big_picture_analyzer.analyze(turn_number)

        death_analyzer = DeathAnalyzer(
            self.history_toolkit,
            self.session_id,
            self.db,
        )
        death_analyzer.analyze_turn(
            turn_number=turn_number,
            game_response=zork_response.Response,
            player_command=input_text,
            location=zork_response.LocationName,
            score=zork_response.Score,
            moves=zork_response.Moves,
        )
        all_deaths = death_analyzer.get_all_deaths()

        report_writer = TurnReportWriter()
        report_writer.write_turn_report(
            session_id=self.session_id,
            turn_number=turn_number,
            location=zork_response.LocationName,
            score=zork_response.Score,
            moves=zork_response.Moves,
            game_response=zork_response.Response,
            player_command=input_text,
            player_reasoning=decision_for_this_command.reasoning,
            issue_agents=decision_for_this_command.issue_agents,
            explorer_agent=decision_for_this_command.explorer_agent,
            loop_detection_agent=decision_for_this_command.loop_detection_agent,
            interaction_agent=decision_for_this_command.interaction_agent,
            decision_prompt=decision_for_this_command.decision_prompt,
            decision=decision_for_this_command.decision,
            recent_history=recent_summary,
            complete_history=long_summary,
            current_inventory=current_inventory,
            big_picture_analysis=big_picture_analysis,
            research_tool_calls=decision_for_this_command.research_tool_calls,
            decision_tool_calls=decision_for_this_command.decision_tool_calls,
            all_deaths=all_deaths,
            map_transitions=transitions,
        )
        report_writer.update_session_index(
            session_id=self.session_id,
            turn_number=turn_number,
            location=zork_response.LocationName,
            score=zork_response.Score,
            moves=zork_response.Moves,
            player_command=input_text,
            game_response=zork_response.Response,
        )

    def _on_background_task_done(self, task: asyncio.Task) -> None:
        """Surface background-task failures via the logger and drop the task
        from the tracking list so it can be GC'd."""
        try:
            self._background_tasks.remove(task)
        except ValueError:
            pass
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.logger.logger.error(
                "Post-turn background task failed",
                exc_info=exc,
            )

    async def _drain_background_tasks(self) -> None:
        """Wait for all scheduled post-turn tasks to finish."""
        if not self._background_tasks:
            return
        pending = list(self._background_tasks)
        self.logger.logger.info(
            f"Draining {len(pending)} pending post-turn task(s) before shutdown..."
        )
        await asyncio.gather(*pending, return_exceptions=True)

    async def _bootstrap_inventory(self):
        """
        Send INVENTORY command to sync our tracking with game state.
        This doesn't count as a game turn.
        """
        self.logger.logger.info("Bootstrapping inventory from game...")

        # Send INVENTORY command to game
        inventory_response = await self.zork_service.play_turn("INVENTORY")
        game_inventory_text = inventory_response.Response

        self.logger.logger.info(f"Game inventory response: {game_inventory_text}")

        # Parse inventory from game response using LLM
        items = self._parse_inventory_response(game_inventory_text)

        self.logger.logger.info(f"Parsed items: {items}")

        # Sync with our tracking (turn 0 = bootstrap)
        self.inventory_toolkit.state.sync_with_game(items, turn_number=0)

        self.logger.logger.info("Inventory bootstrap complete")

    def _parse_inventory_response(self, response: str) -> list:
        """
        Parse game's INVENTORY response into list of items using LLM.

        Args:
            response: Raw INVENTORY command response from game

        Returns:
            List of item names
        """
        from config import get_cheap_llm
        import json

        llm = get_cheap_llm(temperature=0)

        prompt = f"""Parse this Zork game inventory response into a simple list of items.

INVENTORY RESPONSE:
{response}

Return ONLY a JSON array of item names (lowercase, full names as game provides them).
Examples:
- "You are carrying: a brass lantern and a leaflet." → ["brass lantern", "leaflet"]
- "You are empty-handed." → []
- "Your inventory is empty." → []

Output ONLY the JSON array, nothing else."""

        try:
            result = llm.invoke(prompt)
            items = json.loads(result.content)
            return items if isinstance(items, list) else []
        except Exception as e:
            self.logger.logger.error(f"Failed to parse inventory: {e}")
            return []

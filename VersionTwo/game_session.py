from adventurer.adventurer_service import AdventurerService
from zork.zork_service import ZorkService
from tools.history import HistoryToolkit
from tools.memory import MemoryToolkit
from tools.mapping import MapperToolkit
from tools.database import DatabaseManager
from display_manager import DisplayManager
from game_logger import GameLogger
from config import get_cheap_llm


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

        # Create cheap LLM for summarization and de-duplication
        cheap_llm = get_cheap_llm(temperature=0)

        # Create history toolkit with cheap LLM for summarization
        self.history_toolkit = HistoryToolkit(cheap_llm, session_id, self.db)

        # Create memory toolkit with cheap LLM for de-duplication (write-only strategic issue storage)
        self.memory_toolkit = MemoryToolkit(session_id, self.db, cheap_llm)

        # Create mapper toolkit for tracking location transitions
        self.mapper_toolkit = MapperToolkit(session_id, self.db)
        self.logger.logger.info("Mapper toolkit initialized")

        # Create inventory toolkit for tracking items
        from tools.inventory import InventoryToolkit
        self.inventory_toolkit = InventoryToolkit(session_id, self.db)
        self.logger.logger.info("Inventory toolkit initialized")

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

        # Track reasoning from previous turn (to display with correct command)
        self.pending_reasoning = None

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
            player_response, issue_agents, explorer_agent, loop_detection_agent, interaction_agent, issue_closed_response, observer_response, decision_prompt = self.adventurer_service.handle_user_input(
                zork_response,
                self.turn_number
            )

            # Extract closed issues and new issue from responses
            closed_issues = issue_closed_response.closed_issue_contents if issue_closed_response else []
            new_issue = observer_response.remember if observer_response and observer_response.remember else None

            # Step 4: Update display with the turn
            # Use pending_reasoning from PREVIOUS turn (reasoning for the command that just executed)
            display.add_turn(
                location=zork_response.LocationName,
                game_text=zork_response.Response,
                command=input_text,  # The command that was just executed
                score=zork_response.Score,
                moves=zork_response.Moves,
                reasoning=self.pending_reasoning,  # Reasoning for THIS command (from previous turn)
                closed_issues=closed_issues,  # Issues that were resolved this turn
                new_issue=new_issue  # New issue identified this turn
            )

            # Store reasoning for the NEXT turn
            self.pending_reasoning = player_response.reason

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

            # Step 8: Write turn report for analysis and debugging
            from tools.reporting import TurnReportWriter
            report_writer = TurnReportWriter()

            # Get current inventory for report
            current_inventory = self.inventory_toolkit.state.get_items()

            report_writer.write_turn_report(
                session_id=self.session_id,
                turn_number=self.turn_number,
                location=zork_response.LocationName,
                score=zork_response.Score,
                moves=zork_response.Moves,
                game_response=zork_response.Response,
                player_command=player_response.command,
                player_reasoning=player_response.reason,
                issue_agents=issue_agents,
                explorer_agent=explorer_agent,
                loop_detection_agent=loop_detection_agent,
                interaction_agent=interaction_agent,
                decision_prompt=decision_prompt,
                decision=player_response,
                recent_history=recent_summary,
                complete_history=long_summary,
                current_inventory=current_inventory
            )

            return player_response.command

        except Exception as e:
            self.logger.log_error(str(e))
            print(f"\nAn error occurred while processing turn: {e}")
            raise  # Re-raise to be caught by play() method

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

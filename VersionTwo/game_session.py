from adventurer.adventurer_service import AdventurerService
from zork.zork_service import ZorkService
from tools.history import HistoryToolkit
from tools.memory import MemoryToolkit
from tools.database import DatabaseManager
from langchain_openai import ChatOpenAI
from display_manager import DisplayManager
from game_logger import GameLogger


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

        # Create history toolkit with GPT-5-nano for summarization (cheap)
        cheap_llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", temperature=0)
        self.history_toolkit = HistoryToolkit(cheap_llm, session_id, self.db)

        # Create memory toolkit (write-only strategic issue storage)
        self.memory_toolkit = MemoryToolkit(session_id, self.db)

        # Pass both toolkits to adventurer service
        self.adventurer_service = AdventurerService(self.history_toolkit, self.memory_toolkit)

        self.turn_number = 0

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

            # Step 3: Process through LangGraph (Research → Decide → Persist)
            # The graph handles: research, decision, and memory persistence
            player_response = self.adventurer_service.handle_user_input(
                zork_response,
                self.turn_number
            )

            # Step 4: Update display with the turn
            display.add_turn(
                location=zork_response.LocationName,
                game_text=zork_response.Response,
                command=input_text,  # The command that was just executed
                score=zork_response.Score,
                moves=zork_response.Moves
            )

            # Step 5: Update display with current summaries
            recent_summary = self.history_toolkit.state.get_full_summary()
            long_summary = self.history_toolkit.state.get_long_running_summary()
            display.update_summary(recent_summary, long_summary)
            self.logger.log_summary_update(recent_summary)

            # Step 6: Update display with memories
            memories = self.memory_toolkit.state.get_top_memories(5)
            if memories:
                memories_text = ""
                for i, mem in enumerate(memories, 1):
                    memories_text += f"{i}. [{mem.importance}/1000] {mem.content}\n"
                    memories_text += f"   (Turn {mem.turn_number} @ {mem.location})\n\n"
                display.update_memories(memories_text.strip())
            else:
                display.update_memories("No memories recorded yet.")

            return player_response.command

        except Exception as e:
            self.logger.log_error(str(e))
            print(f"\nAn error occurred while processing turn: {e}")
            raise  # Re-raise to be caught by play() method

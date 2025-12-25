from adventurer.adventurer_service import AdventurerService
from zork.zork_service import ZorkService
from tools.history import HistoryToolkit
from langchain_openai import ChatOpenAI
from display_manager import DisplayManager


class GameSession:

    def __init__(self, session_id: str):
        """
        Initialize the game session with a session ID
        and the required services.
        """
        self.zork_service = ZorkService(session_id=session_id)

        # Create history toolkit with cheap LLM for summarization
        cheap_llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", temperature=0)
        self.history_toolkit = HistoryToolkit(cheap_llm)

        # Pass toolkit to adventurer service
        self.adventurer_service = AdventurerService(self.history_toolkit)

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
            # Step 1: Send input to the Zork service and get the response
            zork_response = await self.zork_service.play_turn(input_text=input_text)

            # Step 2: Process the response through the AdventurerService
            player_response = self.adventurer_service.handle_user_input(zork_response)

            # Step 3: Update history after turn completes
            self.history_toolkit.update_after_turn(
                game_response=zork_response.Response,
                player_command=input_text,  # The command that was just executed
                location=zork_response.LocationName,
                score=zork_response.Score,
                moves=zork_response.Moves
            )

            # Step 4: Update display with the turn
            display.add_turn(
                location=zork_response.LocationName,
                game_text=zork_response.Response,
                command=input_text,  # The command that was just executed
                score=zork_response.Score,
                moves=zork_response.Moves
            )

            # Step 5: Update display with current summary
            display.update_summary(self.history_toolkit.state.get_full_summary())

            return player_response.command

        except Exception as e:
            print(f"\nAn error occurred while processing turn: {e}")
            raise  # Re-raise to be caught by play() method

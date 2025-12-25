from adventurer.adventurer_service import AdventurerService
from zork.zork_service import ZorkService
from tools.history import HistoryToolkit
from langchain_openai import ChatOpenAI


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
        Main gameplay loop. Initializes the game state and processes turns.
        """
        try:
            # Initialize the game state.
            await self.zork_service.play_turn("verbose")
            adventurer_response = await self.__play_turn("look")

            for count in range(1, 25):
                print(f"Turn {count}:")
                print(f"Adventurer Command: {adventurer_response}")
                adventurer_response = await self.__play_turn(adventurer_response)

        except Exception as e:
            print(f"An error occurred during gameplay: {e}")

    async def __play_turn(self, input_text: str) -> str:
        """
        Play a single turn of the game.
        :param input_text: The player's input command (e.g., "open door").
        :return: The next input to be processed.
        """
        try:
            # Step 1: Send input to the Zork service and get the response
            zork_response = await self.zork_service.play_turn(input_text=input_text)

            # Step 2: Display the Zork service's response
            self.zork_service.display_response(zork_response)

            # Step 3: Process the response through the AdventurerService
            player_response = self.adventurer_service.handle_user_input(zork_response)

            # Step 4: Update history after turn completes
            self.history_toolkit.update_after_turn(
                game_response=zork_response.Response,
                player_command=player_response.command,
                location=zork_response.LocationName,
                score=zork_response.Score,
                moves=zork_response.Moves
            )

            return player_response.command

        except Exception as e:
            print(f"An error occurred while processing turn: {e}")
            return ""

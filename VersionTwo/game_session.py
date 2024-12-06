
from adventurer.adventurer_service import AdventurerService
from zork.zork_service import ZorkService


class GameSession:
    def __init__(self, session_id: str):
        """
        Initialize the game session with a session ID and the required services.
        """
        self.zork_service = ZorkService(session_id=session_id)
        self.adventurer_service = AdventurerService()

    async def play_turn(self, input_text: str):
        """
        Play a single turn of the game.
        :param input_text: The player's input command (e.g., "open door").
        """
        try:
            # Step 1: Send input to the Zork service and get the response
            zork_response = await self.zork_service.play_turn(input_text=input_text)

            # Step 2: Display the Zork service's response
            self.zork_service.display_response(zork_response)

            # Step 3: Process the response through the AdventurerService
            player_response = self.adventurer_service.handle_user_input(zork_response)

            # Step 4: Display the player's response
            print(player_response)

        except Exception as e:
            print(f"An error occurred: {e}")
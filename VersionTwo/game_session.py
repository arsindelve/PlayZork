from adventurer.adventurer_service import AdventurerService
from zork.zork_service import ZorkService


class GameSession:

    def __init__(self, session_id: str):
        """
        Initialize the game session with a session ID and the required services.
        """
        self.zork_service = ZorkService(session_id=session_id)
        self.adventurer_service = AdventurerService()

    async def play(self):
        await self.zork_service.play_turn("verbose")
        adventurer_response = await self.__play_turn("look")

        for count in range(1, 51):  
            print(adventurer_response)
            adventurer_response = await self.__play_turn(adventurer_response.command)
            

    async def __play_turn(self, input_text: str) -> str:
        """
        Play a single turn of the game.
        :param input_text: The player's input command (e.g., "open door").
        """
        try:
            # Step 1: Send input to the Zork service and get the response
            zork_response = await self.zork_service.play_turn(
                input_text=input_text)

            print(zork_response.Response)
            
            # Step 2: Display the Zork service's response
            self.zork_service.display_response(zork_response)

            # Step 3: Process the response through the AdventurerService
            player_response = self.adventurer_service.handle_user_input(
                zork_response)
            
            return player_response

        except Exception as e:
            print(f"An error occurred: {e}")
            return ""

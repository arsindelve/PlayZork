from .zork_api_client import ZorkApiClient
from .zork_api_request import ZorkApiRequest
from .zork_api_response import ZorkApiResponse


class ZorkService:

    def __init__(self, session_id: str):
        self.client = ZorkApiClient()
        self.session_id = session_id  # Manage session ID within the service

    async def play_turn(self, input_text: str) -> ZorkApiResponse:
        # Create the request object
        request_data = ZorkApiRequest(Input=input_text,
                                      SessionId=self.session_id)

        # Call the API
        response = await self.client.get_async(request_data)

        # Return the response or raise an error
        if not response:
            raise Exception("Failed to get a response from the API.")
        return response

    @staticmethod
    def display_response(response: ZorkApiResponse):
        print(f"Response: {response.Response}")
        print(f"Location: {response.LocationName}")
        print(f"Moves: {response.Moves}, Score: {response.Score}")
        # if response.PreviousLocationName:
        #    print(f"Previous Location: {response.PreviousLocationName}")
        # if response.LastMovementDirection:
        #    print(f"Last Movement Direction: {response.LastMovementDirection}")

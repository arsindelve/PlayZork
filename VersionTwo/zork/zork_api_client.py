from typing import Optional
import httpx

from .zork_api_request import ZorkApiRequest
from .zork_api_response import ZorkApiResponse
from config import get_game_config


# Define the API client
class ZorkApiClient:

    def __init__(self, timeout: int = 30):
        # Get the game configuration
        game_config = get_game_config()

        self.client = httpx.AsyncClient(
            base_url=game_config["base_url"],
            timeout=httpx.Timeout(timeout)  # Set a custom timeout
        )
        self.endpoint = game_config["endpoint"]

    async def get_async(self,
                        resource: ZorkApiRequest) -> Optional[ZorkApiResponse]:
        # Convert the request model to a dictionary
        request_json = resource.model_dump(by_alias=True)

        # Make the POST request
        try:
            response = await self.client.post(
                self.endpoint,
                json=request_json  # Correctly format the body as JSON
            )
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Deserialize the response JSON into a ZorkApiResponse
            return ZorkApiResponse(**response.json())
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

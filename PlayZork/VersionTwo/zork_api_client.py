from typing import Optional
import httpx
from pydantic import BaseModel, Field

# Define the API endpoint
BASE_URL = "https://bxqzfka0hc.execute-api.us-east-1.amazonaws.com"


# Define the request and response models
class ZorkApiRequest(BaseModel):
    Input: str  # Uppercase field names to match the API
    SessionId: str


class ZorkApiResponse(BaseModel):
    Response: Optional[str] = Field(None, alias="response")
    LocationName: Optional[str] = Field(None, alias="locationName")
    Moves: int = Field(0, alias="moves")
    Score: int = Field(0, alias="score")
    PreviousLocationName: Optional[str] = Field(None,
                                                alias="previousLocationName")
    LastMovementDirection: Optional[str] = Field(None,
                                                 alias="lastMovementDirection")


# Define the API client
class ZorkApiClient:

    def __init__(self):
        self.client = httpx.AsyncClient(base_url=BASE_URL)

    async def get_async(self,
                        resource: ZorkApiRequest) -> Optional[ZorkApiResponse]:
        # Convert the request model to a dictionary
        request_json = resource.model_dump(by_alias=True)

        # Make the POST request
        try:
            response = await self.client.post(
                "/Prod/ZorkOne",
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

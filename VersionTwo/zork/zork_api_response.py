from typing import Optional

from pydantic import BaseModel, Field


class ZorkApiResponse(BaseModel):
    Response: Optional[str] = Field(None, alias="response")
    LocationName: Optional[str] = Field(None, alias="locationName")
    Moves: int = Field(0, alias="moves")
    Score: int = Field(0, alias="score")
    PreviousLocationName: Optional[str] = Field(None,
                                                alias="previousLocationName")
    LastMovementDirection: Optional[str] = Field(None,
                                                 alias="lastMovementDirection")

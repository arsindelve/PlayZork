from pydantic import BaseModel, Field
from typing import Optional


class ObserverResponse(BaseModel):
    """
    Response from the ObserverAgent after analyzing the game response.

    The ObserverAgent identifies new strategic issues, puzzles, or obstacles
    that should be tracked for future turns.
    """
    remember: Optional[str] = Field(default="", description="New strategic issue to track (or empty if nothing new)")
    rememberImportance: Optional[int] = Field(default=None, description="Importance score 1-1000 (or None if remember is empty)")
    item: Optional[str] = Field(default="", description="Any item mentioned in game response")

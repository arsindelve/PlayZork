from pydantic import BaseModel, Field
from typing import Optional, List


class InteractionResponse(BaseModel):
    """Response from InteractionAgent analyzing local interactions."""

    proposed_action: str = Field(
        description="Interaction command to execute (or 'nothing' if no interactions)"
    )
    reason: str = Field(
        default="",
        description="Explanation of why this interaction was chosen"
    )
    confidence: int = Field(
        description="Confidence score 0-100 (0 if no interactions, 70-95 if clear interaction)"
    )
    detected_objects: Optional[List[str]] = Field(
        default=None,
        description="List of interactive objects detected in current location"
    )
    inventory_items: Optional[List[str]] = Field(
        default=None,
        description="Relevant inventory items for this interaction"
    )

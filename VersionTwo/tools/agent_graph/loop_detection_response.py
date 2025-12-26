from pydantic import BaseModel, Field
from typing import Optional


class LoopDetectionResponse(BaseModel):
    """Response from LoopDetectionAgent analyzing for stuck/oscillating patterns."""

    loop_detected: bool = Field(
        description="True if a loop pattern was detected in recent history"
    )
    loop_type: str = Field(
        default="",
        description="Type of loop: 'stuck_location', 'oscillating', or empty if no loop"
    )
    proposed_action: str = Field(
        description="Command to break the loop (or 'nothing' if no loop detected)"
    )
    reason: str = Field(
        default="",
        description="Explanation of the loop detected and why this action breaks it"
    )
    confidence: int = Field(
        description="Confidence score 0-100 (90-100 if loop detected, 0 if no loop)"
    )

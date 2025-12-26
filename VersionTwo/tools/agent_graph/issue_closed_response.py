from pydantic import BaseModel, Field
from typing import List


class IssueClosedResponse(BaseModel):
    """Response from the IssueClosedAgent after analyzing recent history."""
    closed_issues: List[str] = Field(
        default_factory=list,
        description="List of issue content strings that have been resolved and should be closed"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why these issues were closed"
    )

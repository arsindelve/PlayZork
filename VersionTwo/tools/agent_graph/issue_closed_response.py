from pydantic import BaseModel, Field
from typing import List


class IssueClosedResponse(BaseModel):
    """Response from the IssueClosedAgent after analyzing recent history."""
    closed_issue_ids: List[int] = Field(
        default_factory=list,
        description="List of memory IDs for issues that have been resolved and should be closed"
    )
    closed_issue_contents: List[str] = Field(
        default_factory=list,
        description="List of issue content strings for display purposes (populated after removal)"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why these issues were closed"
    )

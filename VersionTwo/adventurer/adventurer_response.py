from pydantic import BaseModel, Field
from typing import Optional


class AdventurerResponse(BaseModel):
    """
    Response from the Decision Agent (arbiter).

    The Decision Agent ONLY chooses the best action from specialist agent proposals.
    It does NOT identify new issues - that's handled by the Observer Agent.
    """
    command: str = Field(description="The command to execute (chosen from agent proposals)")
    reason: Optional[str] = Field(default="", description="Which agent was chosen and why")
    moved: Optional[str] = Field(default="", description="Direction if movement command")

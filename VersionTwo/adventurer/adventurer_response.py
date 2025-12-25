from pydantic import BaseModel, Field
from typing import Optional


class AdventurerResponse(BaseModel):
    command: str = Field()
    reason: Optional[str] = Field(default="")
    remember: Optional[str] = Field(default="")
    rememberImportance: Optional[int] = Field(default=0)
    item: Optional[str] = Field(default="")
    moved: Optional[str] = Field(default="")

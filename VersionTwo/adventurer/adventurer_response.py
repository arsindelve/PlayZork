from pydantic import BaseModel, Field
from typing import Optional


class AdventurerResponse(BaseModel):
    command: str = Field()
    reason: Optional[str] = Field(default="")
    remember: Optional[str] = Field(default="")
    rememberImportance: Optional[str] = Field(default="")
    item: Optional[str] = Field(default="")
    moved: Optional[str] = Field(default="")

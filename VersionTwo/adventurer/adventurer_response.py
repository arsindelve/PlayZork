from pydantic import BaseModel, Field

class AdventurerResponse(BaseModel):
    command: str
    reason: str
    remember: str
    rememberImportance: str
    item: str
    moved: str
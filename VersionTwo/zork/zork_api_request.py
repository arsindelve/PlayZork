from pydantic import BaseModel


class ZorkApiRequest(BaseModel):
    Input: str  # Uppercase field names to match the API
    SessionId: str

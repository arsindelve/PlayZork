from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ZorkApiResponse(BaseModel):
    """One turn's result from the game backend.

    Several of these fields were declared but never read, and four more were
    not declared at all, so the agents inferred with an LLM what the server
    already knew (GitHub issue #30). Verified populated on BOTH hosted
    backends (Zork I and Planetfall) by direct probe, 2026-08-22.
    """

    Response: Optional[str] = Field(None, alias="response")
    LocationName: Optional[str] = Field(None, alias="locationName")
    Moves: int = Field(0, alias="moves")
    Score: int = Field(0, alias="score")

    # The room we were in BEFORE this turn. Equal to LocationName when the turn
    # did not move us, so `PreviousLocationName != LocationName` is an
    # authoritative "did we move?" test — no text parsing required.
    PreviousLocationName: Optional[str] = Field(None,
                                                alias="previousLocationName")

    # The direction of the most recent movement ATTEMPT: "N", "Up", "In",
    # "Out", "Down". Note it is STICKY — it keeps its previous value on turns
    # that attempt no movement — so it must never be read as "this turn moved".
    # Pair it with the PreviousLocationName test above.
    LastMovementDirection: Optional[str] = Field(None,
                                                 alias="lastMovementDirection")

    # Exits from the current room, as an int enum: N=0, S=1, E=2, W=3,
    # Up=10, Down=11 (other values unverified). NOT a walkable-exit oracle —
    # North of House reports 7 while both NW and SW are refused — but it is a
    # usable room FINGERPRINT: two rooms both named "Forest" differ here
    # ([3,2,1] vs [3,0,1]), which is the signal #15 needs for room identity.
    Exits: Optional[List[int]] = Field(None, alias="exits")

    # The game's OWN inventory listing. Authoritative: verified that a failed
    # TAKE leaves it unchanged and a successful one updates it. This is ground
    # truth that the LLM InventoryAnalyzer was previously guessing at.
    Inventory: Optional[List[str]] = Field(None, alias="inventory")

    # Object -> list of commands the game will accept for it, e.g.
    # {"window": ["open window", "close window", "examine window"]}.
    # A deterministic source for the InteractionAgent (relevant to #25).
    ActionsAvailableFromLocation: Optional[Dict[str, Any]] = Field(
        None, alias="actionsAvailableFromLocation")
    ActionsAvailableFromInventory: Optional[Dict[str, Any]] = Field(
        None, alias="actionsAvailableFromInventory")

    Time: Optional[int] = Field(None, alias="time")

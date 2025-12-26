"""Mapper tools - LangChain @tool definitions for map queries"""
from typing import Optional
from langchain_core.tools import tool

from .mapper_state import MapperState


# Module-level reference to shared mapper state
_mapper_state: Optional[MapperState] = None


def initialize_mapper_tools(mapper_state: MapperState) -> None:
    """
    Initialize the mapper tools with a reference to the mapper state.
    Must be called before tools can be used.

    Args:
        mapper_state: The MapperState instance to use for tool queries
    """
    global _mapper_state
    _mapper_state = mapper_state


@tool
def get_map() -> str:
    """Get a complete map of all known location transitions discovered so far.

    Shows every movement you've made between locations, formatted as:
    "From [Location A], going [DIRECTION] leads to [Location B]"

    This is CRITICAL for:
    - Understanding the world's geography
    - Finding unexplored exits from known locations
    - Planning efficient routes to objectives
    - Avoiding getting lost or backtracking unnecessarily

    Returns:
        Complete list of all known transitions, chronologically
    """
    if _mapper_state is None:
        return "Error: Mapper tools not initialized."

    transitions = _mapper_state.get_all_transitions()

    if not transitions:
        return "No map data yet. Explore to build the map!"

    result = "COMPLETE MAP OF KNOWN TRANSITIONS:\n\n"
    for trans in transitions:
        result += f"From '{trans.from_location}', going {trans.direction} leads to '{trans.to_location}'\n"
        result += f"  (discovered on turn {trans.turn_discovered})\n\n"

    result += f"\nTotal known transitions: {len(transitions)}"

    return result.strip()


@tool
def get_exits_from_location(location: str) -> str:
    """Get all known exits from a specific location.

    Use this to see what directions you can go from a given location.
    Essential for:
    - Finding unexplored directions from your current location
    - Planning your next move
    - Avoiding dead ends

    Args:
        location: The location name to query (e.g., "West Of House", "Behind House")

    Returns:
        List of known exits and where they lead
    """
    if _mapper_state is None:
        return "Error: Mapper tools not initialized."

    exits = _mapper_state.get_exits_from(location)

    if not exits:
        return f"No known exits from '{location}'. Either you haven't been there yet, or you haven't tried moving from there."

    result = f"KNOWN EXITS FROM '{location}':\n\n"
    for direction, destination in exits:
        result += f"  {direction} -> '{destination}'\n"

    result += f"\nTotal exits: {len(exits)}"

    return result.strip()


def get_mapper_tools():
    """
    Get the list of all mapper tools for use with LangChain agents

    Returns:
        List of tool functions decorated with @tool
    """
    return [get_map, get_exits_from_location]

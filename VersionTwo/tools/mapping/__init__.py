"""MapperToolkit - Facade for the mapping system"""
from typing import List, Optional
from tools.database import DatabaseManager

from .mapper_state import MapperState, LocationTransition
from .mapper_tools import initialize_mapper_tools, get_mapper_tools
from .pathfinder import PathFinder


class MapperToolkit:
    """
    Facade for the mapper system.
    Manages map state and provides tools for LangChain agents to query the map.
    """

    def __init__(self, session_id: str, db: DatabaseManager):
        """
        Initialize the mapper toolkit with database backend

        Args:
            session_id: Unique identifier for this game session
            db: DatabaseManager instance for persistence
        """
        self.state = MapperState(session_id=session_id, db=db)

        # Initialize the module-level state for tools
        initialize_mapper_tools(self.state)

    def update_after_turn(
        self,
        current_location: str,
        player_command: str,
        turn_number: int
    ) -> None:
        """
        Update map state after a game turn completes.
        This should be called by the game loop after each turn.

        Args:
            current_location: Current location name
            player_command: Command that was executed
            turn_number: Current turn number
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            self.state.update_from_turn(
                current_location=current_location,
                player_command=player_command,
                turn_number=turn_number
            )
        except Exception as e:
            logger.error(f"ERROR in mapper update_after_turn: {e}", exc_info=True)

    def get_tools(self) -> List:
        """
        Get the list of mapper tools for use with LangChain agents

        Returns:
            List of tool functions that agents can call
        """
        return get_mapper_tools()

    def find_path(
        self,
        from_location: str,
        to_location: str
    ) -> Optional[List[str]]:
        """
        Find shortest path between two locations (Python API).

        Args:
            from_location: Starting location name
            to_location: Destination location name

        Returns:
            List of directions or None if no path exists
        """
        return self.state.pathfinder.find_path(from_location, to_location)

    def get_path_string(
        self,
        from_location: str,
        to_location: str
    ) -> str:
        """
        Find path and return as formatted string (Python API).

        Args:
            from_location: Starting location name
            to_location: Destination location name

        Returns:
            Comma-separated directions or error message
        """
        return self.state.pathfinder.get_path_string(from_location, to_location)


# Export public API
__all__ = ['MapperToolkit', 'MapperState', 'LocationTransition', 'PathFinder']

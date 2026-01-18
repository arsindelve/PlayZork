"""MapperState - Tracks location transitions and builds a map"""
from typing import Optional, List, Tuple, TYPE_CHECKING
from pydantic import BaseModel
from tools.database import DatabaseManager

if TYPE_CHECKING:
    from .pathfinder import PathFinder


class LocationTransition(BaseModel):
    """Represents a movement from one location to another"""
    from_location: str
    to_location: str
    direction: str
    turn_discovered: int


class MapperState:
    """
    Manages the game world map by tracking location transitions.
    Persists to database for long-term memory.
    """

    def __init__(self, session_id: str, db: DatabaseManager):
        """
        Initialize mapper state with database backend.

        Args:
            session_id: Unique identifier for this game session
            db: DatabaseManager instance for persistence
        """
        self.session_id = session_id
        self.db = db
        self.previous_location: Optional[str] = None
        self._pathfinder: Optional['PathFinder'] = None

    @property
    def pathfinder(self) -> 'PathFinder':
        """
        Get pathfinder instance (lazy initialization).

        Returns:
            PathFinder instance for this mapper state
        """
        if self._pathfinder is None:
            from .pathfinder import PathFinder
            self._pathfinder = PathFinder(self)
        return self._pathfinder

    def record_movement(
        self,
        from_location: str,
        to_location: str,
        direction: str,
        turn_number: int
    ) -> bool:
        """
        Record a movement between locations.

        Args:
            from_location: Starting location name
            to_location: Destination location name
            direction: Direction moved (NORTH, SOUTH, etc.)
            turn_number: Turn when this movement occurred

        Returns:
            True if this is a new transition, False if already known
        """
        import logging
        logger = logging.getLogger(__name__)

        # Normalize direction to uppercase
        direction = direction.upper()

        # Add to database
        is_new = self.db.add_map_transition(
            session_id=self.session_id,
            from_location=from_location,
            to_location=to_location,
            direction=direction,
            turn_number=turn_number
        )

        if is_new:
            logger.info(f"[MAPPER] NEW TRANSITION: {from_location} --[{direction}]--> {to_location}")
        else:
            logger.debug(f"[MAPPER] Known transition: {from_location} --[{direction}]--> {to_location}")

        return is_new

    def update_from_turn(
        self,
        current_location: str,
        player_command: str,
        turn_number: int
    ) -> None:
        """
        Update the map based on the current turn.
        Detects movement commands and records transitions.

        Args:
            current_location: Current location name
            player_command: Command that was executed
            turn_number: Current turn number
        """
        import logging
        logger = logging.getLogger(__name__)

        # Try to extract direction from command FIRST
        direction = self._extract_direction(player_command)

        if self.previous_location:
            if self.previous_location != current_location:
                # Location CHANGED - successful movement
                if direction:
                    self.record_movement(
                        from_location=self.previous_location,
                        to_location=current_location,
                        direction=direction,
                        turn_number=turn_number
                    )
                else:
                    logger.debug(f"[MAPPER] Location changed but no direction detected: '{player_command}'")
            elif direction:
                # Location SAME but direction command was issued - BLOCKED direction
                logger.info(f"[MAPPER] BLOCKED: {self.previous_location} --[{direction}]--> (failed)")
                # Record as transition to "BLOCKED" so ExplorerAgent knows not to try it
                self.record_movement(
                    from_location=self.previous_location,
                    to_location="BLOCKED",
                    direction=direction,
                    turn_number=turn_number
                )

        # Update previous location for next turn
        self.previous_location = current_location

    def _extract_direction(self, command: str) -> Optional[str]:
        """
        Extract movement direction from a command.

        Args:
            command: Player command

        Returns:
            Direction string or None if not a movement command
        """
        command_upper = command.upper().strip()

        # Direct direction commands
        # IMPORTANT: Order matters! Longer compound directions must come BEFORE simple directions
        # to avoid substring matching (e.g., "SOUTH" would match in "SOUTHEAST" if checked first)
        directions = [
            # Compound directions FIRST
            "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST",
            # Simple directions SECOND
            "NORTH", "SOUTH", "EAST", "WEST",
            "UP", "DOWN",
            # Abbreviations LAST (after checking full names)
            "NE", "NW", "SE", "SW",  # Compound abbreviations first
            "N", "S", "E", "W", "U", "D"  # Simple abbreviations last
        ]

        for direction in directions:
            if command_upper == direction:
                return direction
            if command_upper.startswith("GO ") and direction in command_upper:
                return direction
            if command_upper.startswith("MOVE ") and direction in command_upper:
                return direction
            if command_upper.startswith("WALK ") and direction in command_upper:
                return direction

        return None

    def get_all_transitions(self) -> List[LocationTransition]:
        """
        Get all known map transitions.

        Returns:
            List of LocationTransition objects
        """
        db_transitions = self.db.get_all_transitions(self.session_id)
        return [
            LocationTransition(
                from_location=from_loc,
                to_location=to_loc,
                direction=direction,
                turn_discovered=turn
            )
            for from_loc, to_loc, direction, turn in db_transitions
        ]

    def get_exits_from(self, location: str) -> List[Tuple[str, str]]:
        """
        Get all known exits from a specific location.

        Args:
            location: Location name

        Returns:
            List of (direction, destination) tuples
        """
        return self.db.get_transitions_from_location(self.session_id, location)

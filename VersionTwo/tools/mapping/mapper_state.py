"""MapperState - Tracks location transitions and builds a map"""
from typing import Optional, List, Tuple, TYPE_CHECKING
from pydantic import BaseModel
from tools.database import DatabaseManager
from .directions import (
    extract_direction,
    is_probable_movement_command,
    normalize_direction,
    normalize_movement_command,
)
from .locations import is_known_location, normalize_location
from .room_identity import RoomRegistry
from .response_signals import is_movement_refusal, looks_like_death

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
        # Distinguishes rooms that share a display name (#15). Rebuilt per
        # process; it only ever ADDS labels, so a resumed session degrades to
        # the bare name rather than mislabelling.
        self._rooms = RoomRegistry()
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
            True if the stored map changed (new passage or a correction)
        """
        import logging
        logger = logging.getLogger(__name__)

        # Normalize to the canonical full name (N -> NORTH, SE -> SOUTHEAST).
        # map_transitions is UNIQUE(session_id, from_location, direction), so
        # storing both "N" and "NORTH" would create two rows for one passage
        # and leave the explored-check believing NORTH is unexplored (#9).
        direction = normalize_direction(direction)

        # Add to database
        changed = self.db.add_map_transition(
            session_id=self.session_id,
            from_location=from_location,
            to_location=to_location,
            direction=direction,
            turn_number=turn_number
        )

        if changed:
            logger.info(f"[MAPPER] RECORDED: {from_location} --[{direction}]--> {to_location}")
        else:
            logger.debug(f"[MAPPER] No map change: {from_location} --[{direction}]--> {to_location}")

        return changed

    def update_from_turn(
        self,
        current_location: str,
        player_command: str,
        turn_number: int,
        game_response: Optional[str] = None,
        api_direction: Optional[str] = None,
        exits: Optional[list] = None
    ) -> None:
        """
        Update the map based on the current turn.
        Detects movement commands and records transitions.

        Args:
            current_location: Current location name
            player_command: Command that was executed
            turn_number: Current turn number
            game_response: This turn's game text. Optional so existing callers
                keep working; without it the death gate cannot fire.
            api_direction: The backend's own LastMovementDirection for this
                turn. The server already knows which way we went, so this is
                preferred over re-deriving it from the command string (#30).
            exits: The backend's exits array, used to tell apart two rooms
                that report the same display name (#15).
        """
        import logging
        logger = logging.getLogger(__name__)

        # Resolve which room this actually is. Zork has several rooms called
        # "Forest" and a maze where every room reports one name; without this
        # they collapse into a single node with all their exits merged (#15).
        # Falls back to the bare name when the backend gives no exits array.
        current_location = self._rooms.resolve(current_location, exits) or current_location

        # Try to extract direction from command FIRST
        direction = self._extract_direction(player_command)

        # With no room name there is nothing to connect the edge to. The DB
        # column is NOT NULL, so the insert was doomed and its IntegrityError
        # was swallowed as "already known" — a silent no-op that also polluted
        # previous_location for the next turn (#7).
        if not is_known_location(current_location):
            logger.debug(
                f"[MAPPER] No location name this turn; skipping transition for "
                f"'{player_command}'"
            )
            # Drop the chain rather than guessing: carrying an unnamed room
            # forward would let the NEXT turn record an edge out of nowhere.
            self.previous_location = None
            return

        # Death is not movement. The game kills the player and TELEPORTS them
        # (Zork respawns you in the Forest), yet reports the respawn room as
        # this turn's LocationName — so the "name changed => passage" rule
        # fabricates an edge from where we died to where we respawned, and BFS
        # then routes future journeys through the fatal move (#12).
        #
        # Record nothing. A missing edge is self-healing: the next survivable
        # move records it. A fabricated one is not — since #11 a real
        # destination overwrites what is stored, so an ungated death turn
        # actively DESTROYS a correct `Cellar --NORTH--> The Troll Room`.
        #
        # This gates the BLOCKED branch too: dying in a room whose name matches
        # the respawn room's would otherwise write a wall that is not there.
        if looks_like_death(game_response):
            logger.info(
                f"[MAPPER] Death detected; not mapping '{player_command}' as a "
                f"passage into '{current_location}'"
            )
            # Keep the chain, unlike the unnamed-room guard above: the player
            # really is standing in the respawn room, so the NEXT turn's move
            # out of it is a genuine edge worth recording.
            self.previous_location = current_location
            return

        if self.previous_location and is_known_location(self.previous_location):
            if self.previous_location != current_location:
                # Location CHANGED - successful movement
                if not direction and api_direction and is_probable_movement_command(player_command):
                    # The backend names the direction it actually moved us:
                    # "climb tree" -> "Up", "enter window" -> "In" (#30). This
                    # beats both the command tokenizer and #14's raw-command
                    # label, because it is canonical AND executable, so the
                    # explorer correctly sees UP as explored rather than
                    # hunting it separately.
                    #
                    # Gated on is_probable_movement_command because
                    # LastMovementDirection is STICKY: on a turn where a timed
                    # event relocated us during "TAKE LAMP", the field still
                    # holds the previous move's direction.
                    direction = normalize_direction(api_direction)

                if not direction and is_probable_movement_command(player_command):
                    # Zork moves the player with plain commands too: CLIMB
                    # TREE, ENTER HOUSE, IN, OUT, CROSS BRIDGE, TOUCH MIRROR.
                    # Dropping those left the destination reachable-FROM but
                    # never reachable-TO — an orphan node BFS can leave and
                    # never plan a route into (#14). The raw command is the
                    # edge label, which makes what the pathfinder hands back
                    # directly executable by the agent.
                    direction = normalize_movement_command(player_command)

                if direction:
                    self.record_movement(
                        from_location=self.previous_location,
                        to_location=current_location,
                        direction=direction,
                        turn_number=turn_number
                    )
                else:
                    logger.debug(f"[MAPPER] Location changed but no direction detected: '{player_command}'")
            elif direction and is_movement_refusal(game_response):
                # Cardinals ONLY: `direction` here is always an extracted
                # compass point, never a raw command. Every EXAMINE/TAKE/READ
                # also leaves the location unchanged, so recording raw commands
                # as BLOCKED would add one junk row per non-movement command,
                # and get_map dumps every row into the LLM prompt (#14).
                # Location SAME *and* the game explicitly refused the move.
                # Inferring BLOCKED from the room name alone wrote permanent
                # walls between Zork's several identically-named "Forest"
                # rooms, where the move actually SUCCEEDED (#10, #15).
                logger.info(f"[MAPPER] BLOCKED: {self.previous_location} --[{direction}]--> (refused)")
                # Record as transition to "BLOCKED" so ExplorerAgent knows not to try it
                self.record_movement(
                    from_location=self.previous_location,
                    to_location="BLOCKED",
                    direction=direction,
                    turn_number=turn_number
                )
            elif direction:
                # Same room name, but nothing said the move failed. It may have
                # SUCCEEDED between two same-named rooms. Record nothing: an
                # incomplete map beats a false wall, which the explorer would
                # treat as explored and never retry (#10, #15).
                logger.debug(
                    f"[MAPPER] '{player_command}' left the room name unchanged and "
                    f"the game did not refuse it; recording nothing"
                )

        # Update previous location for next turn
        self.previous_location = current_location

    def _extract_direction(self, command: str) -> Optional[str]:
        """
        Extract movement direction from a command.

        Delegates to `directions.extract_direction`, which matches whole
        TOKENS. The previous implementation asked `if direction in
        command_upper` under a `startswith("MOVE ")` guard, so the "E" inside
        the verb MOVE itself matched: every `MOVE <noun>` reported EAST-ish
        movement, and since object manipulation leaves the room unchanged the
        mapper then wrote a false `--[E]--> BLOCKED` edge (#10).

        Args:
            command: Player command

        Returns:
            Canonical direction string, or None if this is not a movement command
        """
        return extract_direction(command)

    def resolve_location(self, name: str) -> str:
        """Return the map's own spelling of `name`, else `name` unchanged.

        Lookups are case-insensitive, but OUTPUT keeps the backend's casing so
        prompts and the HTML map never read "WEST OF HOUSE" (#13).
        """
        target = normalize_location(name)
        for from_loc, to_loc, _, _ in self.db.get_all_transitions(self.session_id):
            for stored in (from_loc, to_loc):
                if normalize_location(stored) == target:
                    return stored
        return name

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
            for from_loc, to_loc, direction, turn in self._collapse(db_transitions)
        ]

    def get_exits_from(self, location: str) -> List[Tuple[str, str]]:
        """
        Get all known exits from a specific location.

        Args:
            location: Location name

        Returns:
            List of (direction, destination) tuples
        """
        rows = [
            (location, destination, direction, 0)
            for direction, destination in
            self.db.get_transitions_from_location(self.session_id, location)
        ]
        return [(direction, destination) for _, destination, direction, _ in self._collapse(rows)]

    def _collapse(self, rows):
        """Canonicalize legacy direction tokens and merge rows that collide.

        Sessions recorded before #9 can hold both "N" and "NORTH" from the same
        location — the UNIQUE constraint accepted both because the strings
        differ. Normalizing on read makes a resumed session self-heal without a
        SQL migration.

        Collision rule: a real passage always beats a stale BLOCKED row. Keeping
        the BLOCKED one would hide a known passage from the pathfinder.
        """
        best = {}
        order = []
        for from_loc, to_loc, direction, turn in rows:
            direction = normalize_direction(direction)
            key = (normalize_location(from_loc), direction)
            if key not in best:
                best[key] = (from_loc, to_loc, direction, turn)
                order.append(key)
            elif best[key][1] == "BLOCKED" and to_loc != "BLOCKED":
                best[key] = (from_loc, to_loc, direction, turn)
        return [best[key] for key in order]

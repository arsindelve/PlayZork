"""PathFinder - BFS pathfinding for the game world map"""
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING
from collections import deque

from .locations import normalize_location

# Inverse of each canonical direction, for provisional reverse edges. Kept here
# rather than imported from turn_context so the mapping package stays
# dependency-free (see directions.py).
_REVERSE_DIRECTIONS = {
    "NORTH": "SOUTH", "SOUTH": "NORTH",
    "EAST": "WEST", "WEST": "EAST",
    "NORTHEAST": "SOUTHWEST", "SOUTHWEST": "NORTHEAST",
    "NORTHWEST": "SOUTHEAST", "SOUTHEAST": "NORTHWEST",
    "UP": "DOWN", "DOWN": "UP",
    "N": "SOUTH", "S": "NORTH", "E": "WEST", "W": "EAST",
    "NE": "SOUTHWEST", "SW": "NORTHEAST", "NW": "SOUTHEAST", "SE": "NORTHWEST",
    "U": "DOWN", "D": "UP",
}

if TYPE_CHECKING:
    from .mapper_state import MapperState


class PathFinder:
    """
    Provides pathfinding capabilities for the game world.
    Uses BFS to find shortest paths between locations.
    Pure Python implementation with no LLM usage.
    """

    def __init__(self, mapper_state: 'MapperState'):
        """
        Initialize pathfinder with reference to mapper state.

        Args:
            mapper_state: MapperState instance for accessing transitions
        """
        self.mapper_state = mapper_state

    def _build_graph(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Build adjacency list from all known transitions.
        Filters out BLOCKED transitions.

        Returns:
            Dict mapping location -> [(direction, destination), ...]
        """
        graph: Dict[str, List[Tuple[str, str]]] = {}
        provisional: Dict[str, List[Tuple[str, str]]] = {}
        blocked_directions: Dict[str, set] = {}
        transitions = self.mapper_state.get_all_transitions()

        for trans in transitions:
            # Skip BLOCKED transitions (failed movement attempts), but REMEMBER
            # them: a direction the game has already refused must never be
            # re-offered as a provisional reverse.
            if normalize_location(trans.to_location) == normalize_location("BLOCKED"):
                blocked_directions.setdefault(
                    normalize_location(trans.from_location), set()
                ).add((trans.direction or "").strip().upper())
                continue

            # Graph keys are case-folded so a room entered under one casing and
            # left under another is ONE node (#13). Only directions are ever
            # returned, so nothing the LLM or the HTML map displays changes.
            from_key = normalize_location(trans.from_location)
            if from_key not in graph:
                graph[from_key] = []

            to_key = normalize_location(trans.to_location)
            graph[from_key].append((trans.direction, to_key))

            # PROVISIONAL REVERSE EDGE — for routing only, never stored.
            #
            # `map_transitions` is directed, so walking A->B taught us nothing
            # about getting back, and the agent could not PLAN a return to
            # anywhere it had not already walked back from. In pf4-20260824 the
            # escape pod was at Deck Nine and the agent, one room away, was
            # told "NO PATH"; it only got back by inventing "RETURN TO DECK
            # NINE", which Planetfall's parser happened to accept and Zork's
            # would not.
            #
            # This is a ROUTING hint, not a recorded fact. Nothing is written
            # to the database and no reverse edge appears on the map, so the
            # stored world model stays exactly as conservative as before. One-
            # way passages (Zork's slide, the chimney) therefore never gain a
            # false edge in the map — the guess costs at most one refused move,
            # which `is_movement_refusal` then records as a real BLOCKED edge,
            # and the route corrects itself. That is the direction the standing
            # invariant says to err in: a wrong "yes" costs a turn and the game
            # re-teaches it, a wrong "no" silently removes something real.
            #
            # A real edge always wins: reverses are appended after every
            # recorded transition is in place, and never for a direction that
            # already has one.
            reverse = _REVERSE_DIRECTIONS.get((trans.direction or "").strip().upper())
            if reverse:
                provisional.setdefault(to_key, []).append((reverse, from_key))

        # Merge provisional reverses, letting recorded edges take precedence.
        for node, edges in provisional.items():
            known = graph.setdefault(node, [])
            recorded = {d.strip().upper() for d, _ in known}
            blocked = blocked_directions.get(node, set())
            for direction, destination in edges:
                if direction in recorded or direction in blocked:
                    continue
                recorded.add(direction)
                known.append((direction, destination))

        return graph

    def find_path(
        self,
        from_location: str,
        to_location: str
    ) -> Optional[List[str]]:
        """
        Find shortest path between two locations using BFS.

        Args:
            from_location: Starting location name
            to_location: Destination location name

        Returns:
            List of directions to follow (e.g., ["NORTH", "EAST"])
            or None if no path exists
        """
        # These arrive straight from an LLM tool call, so their casing is not
        # necessarily the backend's (#13).
        from_location = normalize_location(from_location)
        to_location = normalize_location(to_location)

        # Handle same location case
        if from_location == to_location:
            return []

        # Build graph from transitions
        graph = self._build_graph()

        # Check if start location exists in graph
        if from_location not in graph:
            return None

        # BFS with parent tracking for path reconstruction
        queue = deque([from_location])
        visited = {from_location}
        # parent maps: location -> (direction_to_reach_it, previous_location)
        parent: Dict[str, Tuple[Optional[str], Optional[str]]] = {
            from_location: (None, None)
        }

        while queue:
            current = queue.popleft()

            # Check if we reached destination
            if current == to_location:
                # Reconstruct path by backtracking through parent pointers
                path = []
                node = to_location
                while parent[node][0] is not None:
                    direction, prev_node = parent[node]
                    path.append(direction)
                    node = prev_node
                # Reverse to get forward path (we built it backwards)
                path.reverse()
                return path

            # Explore neighbors
            if current in graph:
                for direction, neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = (direction, current)
                        queue.append(neighbor)

        # No path found - destination unreachable
        return None

    def get_path_string(
        self,
        from_location: str,
        to_location: str
    ) -> str:
        """
        Get path as formatted string (user-friendly).

        Args:
            from_location: Starting location name
            to_location: Destination location name

        Returns:
            Comma-separated directions (e.g., "NORTH, EAST, SOUTH")
            or "cannot determine how to get there" if no path exists
        """
        path = self.find_path(from_location, to_location)

        if path is None:
            return "cannot determine how to get there"

        if len(path) == 0:
            return ""  # Already at destination

        return ", ".join(path)

    def get_abbreviated_path(
        self,
        from_location: str,
        to_location: str
    ) -> str:
        """
        Get path with abbreviated directions (e.g., N, S, E, W).

        Args:
            from_location: Starting location name
            to_location: Destination location name

        Returns:
            Comma-separated abbreviated directions or error message
        """
        path = self.find_path(from_location, to_location)

        if path is None:
            return "cannot determine how to get there"

        if len(path) == 0:
            return ""

        # Abbreviation mapping
        abbrev_map = {
            "NORTH": "N",
            "SOUTH": "S",
            "EAST": "E",
            "WEST": "W",
            "NORTHEAST": "NE",
            "NORTHWEST": "NW",
            "SOUTHEAST": "SE",
            "SOUTHWEST": "SW",
            "UP": "U",
            "DOWN": "D"
        }

        abbreviated = [abbrev_map.get(d, d) for d in path]
        return ", ".join(abbreviated)

    def get_next_step(
        self,
        from_location: str,
        to_location: str
    ) -> Optional[str]:
        """
        Get just the first direction to take toward a destination.

        Args:
            from_location: Starting location name
            to_location: Destination location name

        Returns:
            First direction string (e.g., "NORTH") if path exists,
            None if no path exists or already at destination
        """
        path = self.find_path(from_location, to_location)

        if path is None or len(path) == 0:
            return None

        return path[0]

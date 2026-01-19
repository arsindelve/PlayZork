"""PathFinder - BFS pathfinding for the game world map"""
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING
from collections import deque

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
        transitions = self.mapper_state.get_all_transitions()

        for trans in transitions:
            # Skip BLOCKED transitions (failed movement attempts)
            if trans.to_location == "BLOCKED":
                continue

            if trans.from_location not in graph:
                graph[trans.from_location] = []

            graph[trans.from_location].append(
                (trans.direction, trans.to_location)
            )

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

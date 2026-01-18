"""
Comprehensive unit test suite for PathFinder functionality.

Tests cover:
- Basic pathfinding (single hop, multi-hop)
- Edge cases (same location, unreachable, unknown)
- BLOCKED transitions filtering
- Cyclic graphs
- Multiple paths (BFS finds shortest)
- Integration with MapperState and MapperToolkit
- LangChain tool functionality
"""
import pytest
from typing import List, Tuple

from tools.mapping.pathfinder import PathFinder
from tools.mapping.mapper_state import MapperState, LocationTransition
from tools.mapping import MapperToolkit
from tools.mapping.mapper_tools import initialize_mapper_tools, find_path_between_locations


class MockDatabase:
    """Mock database for testing without actual SQLite dependency"""

    def __init__(self):
        self.transitions: List[Tuple[str, str, str, int]] = []

    def add_map_transition(
        self,
        session_id: str,
        from_location: str,
        to_location: str,
        direction: str,
        turn_number: int
    ) -> bool:
        """Add a transition (always returns True for new in mock)"""
        self.transitions.append((from_location, to_location, direction, turn_number))
        return True

    def get_all_transitions(self, session_id: str) -> List[Tuple[str, str, str, int]]:
        """Get all transitions as tuples"""
        return self.transitions

    def get_transitions_from_location(
        self,
        session_id: str,
        location: str
    ) -> List[Tuple[str, str]]:
        """Get exits from a specific location"""
        return [
            (direction, to_loc)
            for from_loc, to_loc, direction, turn in self.transitions
            if from_loc == location
        ]


@pytest.fixture
def mock_db():
    """Fixture providing a clean mock database"""
    return MockDatabase()


@pytest.fixture
def mapper_state(mock_db):
    """Fixture providing a MapperState with mock database"""
    return MapperState(session_id="test_session", db=mock_db)


@pytest.fixture
def pathfinder(mapper_state):
    """Fixture providing a PathFinder instance"""
    return PathFinder(mapper_state)


class TestPathFinderBasics:
    """Test basic PathFinder initialization and simple cases"""

    def test_pathfinder_initialization(self, mapper_state):
        """Test PathFinder can be initialized with MapperState"""
        pf = PathFinder(mapper_state)
        assert pf.mapper_state is mapper_state

    def test_empty_map_no_path(self, pathfinder):
        """Test pathfinding on empty map returns None"""
        result = pathfinder.find_path("Location A", "Location B")
        assert result is None

    def test_empty_map_string_response(self, pathfinder):
        """Test get_path_string on empty map returns error message"""
        result = pathfinder.get_path_string("Location A", "Location B")
        assert result == "cannot determine how to get there"


class TestSingleHopPaths:
    """Test pathfinding with single-hop (direct) connections"""

    def test_single_hop_path(self, mapper_state, pathfinder):
        """Test finding a direct one-hop path"""
        mapper_state.record_movement("Start", "End", "NORTH", 1)

        path = pathfinder.find_path("Start", "End")
        assert path == ["NORTH"]

    def test_single_hop_path_string(self, mapper_state, pathfinder):
        """Test string format for single hop"""
        mapper_state.record_movement("West Of House", "Behind House", "EAST", 1)

        result = pathfinder.get_path_string("West Of House", "Behind House")
        assert result == "EAST"

    def test_single_hop_abbreviated(self, mapper_state, pathfinder):
        """Test abbreviated format for single hop"""
        mapper_state.record_movement("Room A", "Room B", "SOUTH", 1)

        result = pathfinder.get_abbreviated_path("Room A", "Room B")
        assert result == "S"

    def test_multiple_single_hops_from_same_location(self, mapper_state, pathfinder):
        """Test location with multiple exits"""
        mapper_state.record_movement("Center", "North Room", "NORTH", 1)
        mapper_state.record_movement("Center", "East Room", "EAST", 2)
        mapper_state.record_movement("Center", "South Room", "SOUTH", 3)

        path_north = pathfinder.find_path("Center", "North Room")
        path_east = pathfinder.find_path("Center", "East Room")
        path_south = pathfinder.find_path("Center", "South Room")

        assert path_north == ["NORTH"]
        assert path_east == ["EAST"]
        assert path_south == ["SOUTH"]


class TestMultiHopPaths:
    """Test pathfinding with multi-hop paths"""

    def test_two_hop_path(self, mapper_state, pathfinder):
        """Test finding a two-hop path"""
        mapper_state.record_movement("A", "B", "NORTH", 1)
        mapper_state.record_movement("B", "C", "EAST", 2)

        path = pathfinder.find_path("A", "C")
        assert path == ["NORTH", "EAST"]

    def test_three_hop_path(self, mapper_state, pathfinder):
        """Test finding a three-hop path"""
        mapper_state.record_movement("Start", "Room1", "NORTH", 1)
        mapper_state.record_movement("Room1", "Room2", "EAST", 2)
        mapper_state.record_movement("Room2", "End", "SOUTH", 3)

        path = pathfinder.find_path("Start", "End")
        assert path == ["NORTH", "EAST", "SOUTH"]

    def test_long_path(self, mapper_state, pathfinder):
        """Test finding a longer path (5+ hops)"""
        # Build a linear path: A -> B -> C -> D -> E -> F
        mapper_state.record_movement("A", "B", "NORTH", 1)
        mapper_state.record_movement("B", "C", "EAST", 2)
        mapper_state.record_movement("C", "D", "SOUTH", 3)
        mapper_state.record_movement("D", "E", "WEST", 4)
        mapper_state.record_movement("E", "F", "UP", 5)

        path = pathfinder.find_path("A", "F")
        assert path == ["NORTH", "EAST", "SOUTH", "WEST", "UP"]

    def test_multi_hop_path_string_format(self, mapper_state, pathfinder):
        """Test comma-separated format for multi-hop path"""
        mapper_state.record_movement("West Of House", "Behind House", "EAST", 1)
        mapper_state.record_movement("Behind House", "Kitchen", "NORTH", 2)
        mapper_state.record_movement("Kitchen", "Living Room", "EAST", 3)

        result = pathfinder.get_path_string("West Of House", "Living Room")
        assert result == "EAST, NORTH, EAST"


class TestEdgeCases:
    """Test edge cases and special scenarios"""

    def test_same_location_returns_empty_list(self, pathfinder):
        """Test from==to returns empty list, not None"""
        path = pathfinder.find_path("Kitchen", "Kitchen")
        assert path == []
        assert path is not None

    def test_same_location_returns_empty_string(self, pathfinder):
        """Test from==to returns empty string for get_path_string"""
        result = pathfinder.get_path_string("Kitchen", "Kitchen")
        assert result == ""

    def test_unreachable_location_returns_none(self, mapper_state, pathfinder):
        """Test disconnected locations return None"""
        mapper_state.record_movement("A", "B", "NORTH", 1)
        mapper_state.record_movement("C", "D", "SOUTH", 2)  # Separate component

        path = pathfinder.find_path("A", "D")
        assert path is None

    def test_unreachable_location_error_message(self, mapper_state, pathfinder):
        """Test unreachable location returns proper error message"""
        mapper_state.record_movement("A", "B", "NORTH", 1)

        result = pathfinder.get_path_string("A", "Unknown")
        assert result == "cannot determine how to get there"

    def test_unknown_start_location(self, mapper_state, pathfinder):
        """Test starting from unknown location returns None"""
        mapper_state.record_movement("A", "B", "NORTH", 1)

        path = pathfinder.find_path("Unknown", "B")
        assert path is None

    def test_unknown_destination_location(self, mapper_state, pathfinder):
        """Test going to unknown location returns None"""
        mapper_state.record_movement("A", "B", "NORTH", 1)

        path = pathfinder.find_path("A", "Unknown")
        assert path is None

    def test_both_locations_unknown(self, pathfinder):
        """Test both locations unknown returns None"""
        path = pathfinder.find_path("Unknown1", "Unknown2")
        assert path is None


class TestBlockedTransitions:
    """Test handling of BLOCKED transitions"""

    def test_blocked_transition_filtered_out(self, mapper_state, pathfinder):
        """Test BLOCKED transitions are not used in pathfinding"""
        mapper_state.record_movement("A", "B", "EAST", 1)
        mapper_state.record_movement("A", "BLOCKED", "NORTH", 2)  # Failed attempt

        # Should find path EAST, not try NORTH
        path = pathfinder.find_path("A", "B")
        assert path == ["EAST"]

        # NORTH direction should not create a path
        path_blocked = pathfinder.find_path("A", "BLOCKED")
        assert path_blocked is None

    def test_only_blocked_path_available(self, mapper_state, pathfinder):
        """Test when only path is BLOCKED, returns None"""
        mapper_state.record_movement("A", "BLOCKED", "NORTH", 1)

        path = pathfinder.find_path("A", "B")
        assert path is None

    def test_route_around_blocked(self, mapper_state, pathfinder):
        """Test pathfinding routes around BLOCKED to find alternative"""
        # Build map with blocked shortcut but working long route
        # A --[NORTH]--> BLOCKED (shortcut doesn't work)
        # A --[EAST]--> B --[NORTH]--> C (working route)
        mapper_state.record_movement("A", "BLOCKED", "NORTH", 1)
        mapper_state.record_movement("A", "B", "EAST", 2)
        mapper_state.record_movement("B", "C", "NORTH", 3)

        path = pathfinder.find_path("A", "C")
        assert path == ["EAST", "NORTH"]


class TestGraphTopologies:
    """Test pathfinding with different graph structures"""

    def test_cyclic_graph_no_infinite_loop(self, mapper_state, pathfinder):
        """Test cycles in graph don't cause infinite loops"""
        # Create a cycle: A -> B -> C -> A
        mapper_state.record_movement("A", "B", "NORTH", 1)
        mapper_state.record_movement("B", "C", "EAST", 2)
        mapper_state.record_movement("C", "A", "SOUTH", 3)

        # Should still find shortest path
        path = pathfinder.find_path("A", "C")
        assert path == ["NORTH", "EAST"]

    def test_multiple_paths_returns_shortest(self, mapper_state, pathfinder):
        """Test BFS returns shortest path when multiple exist"""
        # Create diamond shape:
        #     B
        #    / \
        #   A   D
        #    \ /
        #     C
        # Paths A->D: A->B->D (2 hops) or A->C->D (2 hops)
        # Both are equally short, BFS should return one of them

        mapper_state.record_movement("A", "B", "NORTH", 1)
        mapper_state.record_movement("A", "C", "SOUTH", 2)
        mapper_state.record_movement("B", "D", "EAST", 3)
        mapper_state.record_movement("C", "D", "EAST", 4)

        path = pathfinder.find_path("A", "D")
        # Should be length 2 (shortest)
        assert len(path) == 2
        assert path[-1] == "EAST"  # Must end with EAST to reach D

    def test_tree_structure(self, mapper_state, pathfinder):
        """Test pathfinding in tree-like structure"""
        # Root -> Left, Root -> Right, Left -> LeftLeaf, Right -> RightLeaf
        mapper_state.record_movement("Root", "Left", "WEST", 1)
        mapper_state.record_movement("Root", "Right", "EAST", 2)
        mapper_state.record_movement("Left", "LeftLeaf", "DOWN", 3)
        mapper_state.record_movement("Right", "RightLeaf", "DOWN", 4)

        path = pathfinder.find_path("Root", "LeftLeaf")
        assert path == ["WEST", "DOWN"]

        path = pathfinder.find_path("Root", "RightLeaf")
        assert path == ["EAST", "DOWN"]

    def test_disconnected_components(self, mapper_state, pathfinder):
        """Test multiple disconnected graph components"""
        # Component 1: A -> B
        mapper_state.record_movement("A", "B", "NORTH", 1)

        # Component 2: X -> Y -> Z
        mapper_state.record_movement("X", "Y", "EAST", 2)
        mapper_state.record_movement("Y", "Z", "SOUTH", 3)

        # Within component should work
        assert pathfinder.find_path("A", "B") == ["NORTH"]
        assert pathfinder.find_path("X", "Z") == ["EAST", "SOUTH"]

        # Between components should fail
        assert pathfinder.find_path("A", "Z") is None
        assert pathfinder.find_path("X", "B") is None


class TestAllDirections:
    """Test pathfinding with all supported directions"""

    def test_cardinal_directions(self, mapper_state, pathfinder):
        """Test NORTH, SOUTH, EAST, WEST"""
        mapper_state.record_movement("Center", "North", "NORTH", 1)
        mapper_state.record_movement("Center", "South", "SOUTH", 2)
        mapper_state.record_movement("Center", "East", "EAST", 3)
        mapper_state.record_movement("Center", "West", "WEST", 4)

        assert pathfinder.find_path("Center", "North") == ["NORTH"]
        assert pathfinder.find_path("Center", "South") == ["SOUTH"]
        assert pathfinder.find_path("Center", "East") == ["EAST"]
        assert pathfinder.find_path("Center", "West") == ["WEST"]

    def test_vertical_directions(self, mapper_state, pathfinder):
        """Test UP and DOWN"""
        mapper_state.record_movement("Ground", "Sky", "UP", 1)
        mapper_state.record_movement("Ground", "Underground", "DOWN", 2)

        assert pathfinder.find_path("Ground", "Sky") == ["UP"]
        assert pathfinder.find_path("Ground", "Underground") == ["DOWN"]

    def test_compound_directions(self, mapper_state, pathfinder):
        """Test NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST"""
        mapper_state.record_movement("Center", "NE", "NORTHEAST", 1)
        mapper_state.record_movement("Center", "NW", "NORTHWEST", 2)
        mapper_state.record_movement("Center", "SE", "SOUTHEAST", 3)
        mapper_state.record_movement("Center", "SW", "SOUTHWEST", 4)

        assert pathfinder.find_path("Center", "NE") == ["NORTHEAST"]
        assert pathfinder.find_path("Center", "NW") == ["NORTHWEST"]
        assert pathfinder.find_path("Center", "SE") == ["SOUTHEAST"]
        assert pathfinder.find_path("Center", "SW") == ["SOUTHWEST"]

    def test_mixed_directions_in_path(self, mapper_state, pathfinder):
        """Test path using various direction types"""
        mapper_state.record_movement("A", "B", "NORTH", 1)
        mapper_state.record_movement("B", "C", "NORTHEAST", 2)
        mapper_state.record_movement("C", "D", "UP", 3)
        mapper_state.record_movement("D", "E", "WEST", 4)

        path = pathfinder.find_path("A", "E")
        assert path == ["NORTH", "NORTHEAST", "UP", "WEST"]


class TestAbbreviatedDirections:
    """Test abbreviated direction output"""

    def test_abbreviate_cardinal_directions(self, mapper_state, pathfinder):
        """Test N, S, E, W abbreviations"""
        mapper_state.record_movement("A", "B", "NORTH", 1)
        mapper_state.record_movement("B", "C", "SOUTH", 2)
        mapper_state.record_movement("C", "D", "EAST", 3)
        mapper_state.record_movement("D", "E", "WEST", 4)

        result = pathfinder.get_abbreviated_path("A", "E")
        assert result == "N, S, E, W"

    def test_abbreviate_vertical_directions(self, mapper_state, pathfinder):
        """Test U, D abbreviations"""
        mapper_state.record_movement("A", "B", "UP", 1)
        mapper_state.record_movement("B", "C", "DOWN", 2)

        result = pathfinder.get_abbreviated_path("A", "C")
        assert result == "U, D"

    def test_abbreviate_compound_directions(self, mapper_state, pathfinder):
        """Test NE, NW, SE, SW abbreviations"""
        mapper_state.record_movement("A", "B", "NORTHEAST", 1)
        mapper_state.record_movement("B", "C", "NORTHWEST", 2)
        mapper_state.record_movement("C", "D", "SOUTHEAST", 3)
        mapper_state.record_movement("D", "E", "SOUTHWEST", 4)

        result = pathfinder.get_abbreviated_path("A", "E")
        assert result == "NE, NW, SE, SW"

    def test_abbreviated_unreachable(self, pathfinder):
        """Test abbreviated format for unreachable locations"""
        result = pathfinder.get_abbreviated_path("A", "B")
        assert result == "cannot determine how to get there"


class TestMapperStateIntegration:
    """Test PathFinder integration with MapperState"""

    def test_pathfinder_property_lazy_initialization(self, mapper_state):
        """Test pathfinder property is lazily initialized"""
        # Should not be initialized yet
        assert mapper_state._pathfinder is None

        # Access property triggers initialization
        pf = mapper_state.pathfinder
        assert pf is not None
        assert isinstance(pf, PathFinder)

        # Second access returns same instance
        pf2 = mapper_state.pathfinder
        assert pf is pf2

    def test_pathfinder_uses_mapper_state_data(self, mapper_state):
        """Test pathfinder correctly reads from mapper state"""
        mapper_state.record_movement("A", "B", "NORTH", 1)
        mapper_state.record_movement("B", "C", "EAST", 2)

        # Pathfinder should see these transitions
        path = mapper_state.pathfinder.find_path("A", "C")
        assert path == ["NORTH", "EAST"]

    def test_pathfinder_updates_with_new_transitions(self, mapper_state):
        """Test pathfinder sees new transitions added to mapper"""
        # Initially only A->B exists
        mapper_state.record_movement("A", "B", "NORTH", 1)
        path1 = mapper_state.pathfinder.find_path("A", "C")
        assert path1 is None  # Can't reach C yet

        # Add B->C
        mapper_state.record_movement("B", "C", "EAST", 2)
        path2 = mapper_state.pathfinder.find_path("A", "C")
        assert path2 == ["NORTH", "EAST"]  # Now can reach C


class TestMapperToolkitIntegration:
    """Test PathFinder integration with MapperToolkit facade"""

    def test_mapper_toolkit_find_path(self, mock_db):
        """Test MapperToolkit.find_path() convenience method"""
        toolkit = MapperToolkit(session_id="test", db=mock_db)
        toolkit.state.record_movement("A", "B", "NORTH", 1)

        path = toolkit.find_path("A", "B")
        assert path == ["NORTH"]

    def test_mapper_toolkit_get_path_string(self, mock_db):
        """Test MapperToolkit.get_path_string() convenience method"""
        toolkit = MapperToolkit(session_id="test", db=mock_db)
        toolkit.state.record_movement("A", "B", "NORTH", 1)
        toolkit.state.record_movement("B", "C", "EAST", 2)

        result = toolkit.get_path_string("A", "C")
        assert result == "NORTH, EAST"

    def test_mapper_toolkit_unreachable(self, mock_db):
        """Test MapperToolkit methods handle unreachable locations"""
        toolkit = MapperToolkit(session_id="test", db=mock_db)

        path = toolkit.find_path("A", "B")
        assert path is None

        path_str = toolkit.get_path_string("A", "B")
        assert path_str == "cannot determine how to get there"


class TestLangChainToolIntegration:
    """Test LangChain @tool integration"""

    def test_langchain_tool_initialization(self, mapper_state):
        """Test LangChain tool can be initialized with mapper state"""
        initialize_mapper_tools(mapper_state)
        # Should not raise any errors

    def test_langchain_tool_find_path(self, mapper_state):
        """Test find_path_between_locations tool function"""
        initialize_mapper_tools(mapper_state)
        mapper_state.record_movement("West Of House", "Behind House", "EAST", 1)
        mapper_state.record_movement("Behind House", "Kitchen", "NORTH", 2)

        # LangChain @tool creates a StructuredTool, use .invoke() method
        result = find_path_between_locations.invoke({
            "from_location": "West Of House",
            "to_location": "Kitchen"
        })
        assert result == "EAST, NORTH"

    def test_langchain_tool_unreachable(self, mapper_state):
        """Test LangChain tool handles unreachable locations"""
        initialize_mapper_tools(mapper_state)
        mapper_state.record_movement("A", "B", "NORTH", 1)

        # LangChain @tool creates a StructuredTool, use .invoke() method
        result = find_path_between_locations.invoke({
            "from_location": "A",
            "to_location": "Unknown"
        })
        assert result == "cannot determine how to get there"

    def test_langchain_tool_not_initialized(self):
        """Test LangChain tool returns error if not initialized"""
        # Reset the module-level state
        from tools.mapping import mapper_tools
        mapper_tools._mapper_state = None

        # LangChain @tool creates a StructuredTool, use .invoke() method
        result = find_path_between_locations.invoke({
            "from_location": "A",
            "to_location": "B"
        })
        assert result == "Error: Mapper tools not initialized."


class TestRealWorldScenarios:
    """Test realistic game scenarios"""

    def test_zork_west_of_house_scenario(self, mapper_state, pathfinder):
        """Test a realistic Zork-like map scenario"""
        # Build a small section of a Zork-like map
        mapper_state.record_movement("West Of House", "North Of House", "NORTH", 1)
        mapper_state.record_movement("West Of House", "South Of House", "SOUTH", 2)
        mapper_state.record_movement("West Of House", "Behind House", "EAST", 3)
        mapper_state.record_movement("North Of House", "Behind House", "EAST", 4)
        mapper_state.record_movement("Behind House", "Kitchen", "NORTH", 5)
        mapper_state.record_movement("Kitchen", "Living Room", "EAST", 6)

        # Test various paths
        path1 = pathfinder.get_path_string("West Of House", "Kitchen")
        assert path1 in ["EAST, NORTH", "NORTH, EAST, NORTH"]  # Multiple valid paths

        path2 = pathfinder.get_path_string("West Of House", "Living Room")
        # Should find shortest path (3 hops)
        assert len(pathfinder.find_path("West Of House", "Living Room")) == 3

    def test_maze_with_dead_ends(self, mapper_state, pathfinder):
        """Test navigating a maze with dead ends (BLOCKED)"""
        # Maze structure with some dead ends
        mapper_state.record_movement("Start", "Path1", "NORTH", 1)
        mapper_state.record_movement("Path1", "BLOCKED", "EAST", 2)  # Dead end
        mapper_state.record_movement("Path1", "Path2", "NORTH", 3)
        mapper_state.record_movement("Path2", "BLOCKED", "WEST", 4)  # Dead end
        mapper_state.record_movement("Path2", "Goal", "NORTH", 5)

        path = pathfinder.find_path("Start", "Goal")
        assert path == ["NORTH", "NORTH", "NORTH"]
        # Should not try to use BLOCKED paths

    def test_return_journey(self, mapper_state, pathfinder):
        """Test finding path back to starting location"""
        # Build a path and see if we can return
        mapper_state.record_movement("Home", "Forest", "NORTH", 1)
        mapper_state.record_movement("Forest", "Cave", "EAST", 2)
        mapper_state.record_movement("Cave", "Treasure", "DOWN", 3)

        # Forward journey
        path_there = pathfinder.find_path("Home", "Treasure")
        assert path_there == ["NORTH", "EAST", "DOWN"]

        # Note: Return journey won't work without reverse transitions!
        # This is correct behavior - PathFinder doesn't assume bidirectional edges
        path_back = pathfinder.find_path("Treasure", "Home")
        assert path_back is None  # No return path recorded


class TestPerformance:
    """Test pathfinding performance with larger graphs"""

    def test_large_linear_graph(self, mapper_state, pathfinder):
        """Test performance with a long linear path (50 nodes)"""
        # Build chain: A -> B -> C -> ... -> 50 nodes
        for i in range(50):
            from_loc = f"Node{i}"
            to_loc = f"Node{i+1}"
            mapper_state.record_movement(from_loc, to_loc, "NORTH", i)

        path = pathfinder.find_path("Node0", "Node50")
        assert path is not None
        assert len(path) == 50
        # All should be NORTH
        assert all(direction == "NORTH" for direction in path)

    def test_large_grid_graph(self, mapper_state, pathfinder):
        """Test pathfinding in a grid structure (10x10)"""
        # Build a 10x10 grid where you can move EAST or SOUTH
        for row in range(10):
            for col in range(10):
                current = f"R{row}C{col}"
                if col < 9:  # Can move EAST
                    east = f"R{row}C{col+1}"
                    mapper_state.record_movement(current, east, "EAST", row * 10 + col)
                if row < 9:  # Can move SOUTH
                    south = f"R{row+1}C{col}"
                    mapper_state.record_movement(current, south, "SOUTH", row * 10 + col + 100)

        # Find path from top-left to bottom-right
        path = pathfinder.find_path("R0C0", "R9C9")
        assert path is not None
        assert len(path) == 18  # 9 EAST + 9 SOUTH
        assert path.count("EAST") == 9
        assert path.count("SOUTH") == 9


# Pytest markers for running specific test groups
pytestmark = pytest.mark.pathfinder

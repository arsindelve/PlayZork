# PlayZork Test Suite

This directory contains unit tests for the PlayZork VersionTwo implementation.

## Running Tests

### From Command Line

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_pathfinder.py
```

Run with verbose output:
```bash
pytest -v
```

Run specific test class:
```bash
pytest tests/test_pathfinder.py::TestPathFinderBasics
```

Run specific test:
```bash
pytest tests/test_pathfinder.py::TestPathFinderBasics::test_pathfinder_initialization
```

Run tests with specific marker:
```bash
pytest -m pathfinder
```

### From PyCharm

1. **Run All Tests in File:**
   - Right-click on `test_pathfinder.py` in the Project view
   - Select "Run 'pytest in test_pathfinder.py'"

2. **Run Specific Test Class:**
   - Open `test_pathfinder.py`
   - Click the green arrow icon next to the class definition (e.g., `class TestPathFinderBasics:`)
   - Select "Run 'pytest in test_pathfinder.py::TestPathFinderBasics'"

3. **Run Individual Test:**
   - Open `test_pathfinder.py`
   - Click the green arrow icon next to the test method (e.g., `def test_pathfinder_initialization`)
   - Select "Run 'pytest in test_pathfinder.py::TestPathFinderBasics::test_pathfinder_initialization'"

4. **Debug Tests:**
   - Right-click on any test/class/file
   - Select "Debug 'pytest in...'" instead of "Run"
   - Set breakpoints by clicking in the gutter next to line numbers

5. **Configure PyCharm Test Runner:**
   - Go to: Preferences > Tools > Python Integrated Tools
   - Set "Default test runner" to "pytest"
   - This enables pytest as the default test framework

## Test Organization

### test_pathfinder.py (48 tests)

Comprehensive test suite for the PathFinder pathfinding functionality.

**Test Classes:**

1. **TestPathFinderBasics** (3 tests)
   - Basic initialization
   - Empty map handling

2. **TestSingleHopPaths** (4 tests)
   - Direct single-hop pathfinding
   - Multiple exits from same location

3. **TestMultiHopPaths** (4 tests)
   - Two-hop, three-hop, and longer paths
   - Path string formatting

4. **TestEdgeCases** (7 tests)
   - Same location (from == to)
   - Unreachable locations
   - Unknown locations

5. **TestBlockedTransitions** (3 tests)
   - BLOCKED transition filtering
   - Routing around blocked paths

6. **TestGraphTopologies** (4 tests)
   - Cyclic graphs
   - Multiple paths (shortest path selection)
   - Tree structures
   - Disconnected components

7. **TestAllDirections** (4 tests)
   - Cardinal directions (N, S, E, W)
   - Vertical directions (UP, DOWN)
   - Compound directions (NE, NW, SE, SW)
   - Mixed directions in paths

8. **TestAbbreviatedDirections** (4 tests)
   - Direction abbreviation (NORTH -> N)
   - All direction types
   - Error messages

9. **TestMapperStateIntegration** (3 tests)
   - Lazy initialization
   - Integration with MapperState
   - Dynamic updates

10. **TestMapperToolkitIntegration** (3 tests)
    - MapperToolkit facade methods
    - Convenience APIs

11. **TestLangChainToolIntegration** (4 tests)
    - LangChain @tool integration
    - Tool initialization
    - Error handling

12. **TestRealWorldScenarios** (3 tests)
    - Zork-like map scenarios
    - Mazes with dead ends
    - Return journeys

13. **TestPerformance** (2 tests)
    - Large linear graphs (50 nodes)
    - Large grid graphs (10x10)

## Coverage

The test suite covers:
- ✅ Basic pathfinding (single & multi-hop)
- ✅ Edge cases (same location, unreachable, unknown)
- ✅ BLOCKED transitions filtering
- ✅ All direction types (cardinal, vertical, compound)
- ✅ Graph topologies (cycles, trees, disconnected)
- ✅ Direction abbreviations
- ✅ Integration with MapperState
- ✅ Integration with MapperToolkit
- ✅ LangChain tool functionality
- ✅ Real-world game scenarios
- ✅ Performance with large graphs

## Test Fixtures

The test suite uses pytest fixtures for clean test setup:

- `mock_db`: Provides a MockDatabase instance (no SQLite dependency)
- `mapper_state`: Provides a MapperState with mock database
- `pathfinder`: Provides a PathFinder instance

## Mock Objects

**MockDatabase**: Simulates database operations without requiring actual SQLite database. This makes tests:
- Faster (no disk I/O)
- More reliable (no database state pollution)
- Easier to debug (all state in memory)

## Dependencies

The test suite requires:
- `pytest` (installed as dev dependency via `uv add --dev pytest`)

No additional test dependencies are needed - all tests use mock objects.

## Continuous Integration

To run tests in CI/CD pipelines:

```bash
# Install dependencies
uv sync

# Run tests with coverage (if pytest-cov is installed)
pytest --cov=tools/mapping/pathfinder --cov-report=term-missing

# Run tests with JUnit XML output (for CI systems)
pytest --junitxml=test-results.xml
```

## Test Results

Current status: **48/48 tests passing (100%)**

```
========== 48 passed, 1 warning in 0.17s ==========
```

The warning is about the custom `pathfinder` marker and is non-critical.

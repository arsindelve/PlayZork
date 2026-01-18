"""
pytest configuration file (conftest.py)

This file configures pytest to properly resolve imports when running from PyCharm.
It ensures the VersionTwo directory is in the Python path.
"""
import sys
from pathlib import Path

# Get the VersionTwo directory (where this conftest.py is located)
version_two_dir = Path(__file__).parent

# Add VersionTwo to Python path if not already there
if str(version_two_dir) not in sys.path:
    sys.path.insert(0, str(version_two_dir))

# Optional: Print path for debugging (remove after confirming it works)
# print(f"[conftest.py] Added to sys.path: {version_two_dir}")
# print(f"[conftest.py] sys.path = {sys.path[:3]}")

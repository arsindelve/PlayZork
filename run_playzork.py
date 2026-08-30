"""PyCharm play-button entry point for PlayZork VersionTwo."""

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VERSION_TWO_DIR = PROJECT_ROOT / "VersionTwo"


def main() -> None:
    """Launch VersionTwo with the same paths as the documented CLI command."""
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(VERSION_TWO_DIR))
    runpy.run_path(str(VERSION_TWO_DIR / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()

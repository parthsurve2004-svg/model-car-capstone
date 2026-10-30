"""Launch the isolated traffic-sign runtime from any working directory."""
from pathlib import Path
import sys

SNAPSHOT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SNAPSHOT_ROOT))

from car.main import main

if __name__ == "__main__":
    main()

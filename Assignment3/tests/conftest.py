"""Allow tests to run from the submission folder or its parent."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Allow tests from an extracted source folder before an editable install."""
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The two submissions intentionally use the same CLI filename. Give this
# runner a distinct test import name so collecting both suites is order-safe.
_spec = importlib.util.spec_from_file_location("functionaltests2_runner", ROOT / "run_functional_tests.py")
_runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _runner
_spec.loader.exec_module(_runner)

"""Build the panels for one export (see common.py): python3 -I build.py old|new"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

common.build(sys.argv[1])

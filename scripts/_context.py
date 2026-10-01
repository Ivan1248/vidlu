import sys
from pathlib import Path

_root = (Path(__file__).parent.parent).resolve()
sys.path.insert(0, str(_root))

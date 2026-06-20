import sys
from pathlib import Path

_root = (Path(__file__).parent.parent).resolve()
sys.path.insert(0, str(_root))

# irap_data uses a src-layout (irap-data/irap_data/); add its project dir so
# it's importable without installation, matching how vidlu itself is found.
_irap_data_project = _root / "irap-data"
if _irap_data_project.exists() and str(_irap_data_project) not in sys.path:
    sys.path.insert(1, str(_irap_data_project))

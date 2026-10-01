"""Make ``erd_recording`` importable from a plain ``pytest`` run (no colcon).

``colcon test`` installs the package properly; this is only for fast
iteration (``pytest src/ros2/erd_recording/test`` from ``.venv-erd``).
"""

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

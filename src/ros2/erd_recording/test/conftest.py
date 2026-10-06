"""Make ``erd_recording`` importable from a plain ``pytest`` run (no colcon).

``colcon test`` installs the package properly; this is only for fast
iteration (``pytest src/ros2/erd_recording/test`` from ``.venv-erd``).
"""

import os
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# The checked-in lab configs name machine-dependent paths through the
# variables workspace_setup.sh exports. Give a plain `pytest` run (or a
# `colcon test` from a shell that never sourced it) the same defaults.
_REPO_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("ERD_REPO_ROOT", str(_REPO_ROOT))
os.environ.setdefault("ERD_DATA_ROOT", str(_REPO_ROOT / "data" / "real_robot"))
os.environ.setdefault("ERD_CONSUMER_REPO", str(_REPO_ROOT.parent / "dynamic_model_nn"))
os.environ.setdefault("ERD_CONSUMER_PYTHON",
                      str(Path(os.environ["ERD_CONSUMER_REPO"]) / ".venv" / "bin" / "python"))

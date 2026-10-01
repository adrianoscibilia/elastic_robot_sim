# Copyright 2026 elastic_robot_sim contributors
# SPDX-License-Identifier: Apache-2.0
"""RR_08: exercise the installed script, including colcon's interpreter."""
import sys
from pathlib import Path

from ament_index_python.packages import get_package_prefix


def test_installed_rtde_logger_uses_active_venv():
    prefix = Path(get_package_prefix("erd_ur10"))
    assert not (prefix / "bin" / "rtde_logger").exists()
    script = prefix / "lib" / "erd_ur10" / "rtde_logger"
    interpreter = script.read_text().splitlines()[0].removeprefix("#!")
    # Resolving symlinks would hide an incorrect system-Python shebang.
    assert Path(interpreter).parent == Path(sys.executable).parent
    assert Path(interpreter).parent.parent == Path(sys.prefix)

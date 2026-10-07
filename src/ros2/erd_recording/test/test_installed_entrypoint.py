"""RR_10 item 1: the installed record_* scripts run .venv-erd's interpreter,
not whatever `python3` is first on PATH (mirrors erd_ur10's test)."""

import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ["record_iiwa", "record_ur10"])
def test_installed_script_shebang_is_the_workspace_venv(script):
    ws = os.environ.get("ERD_WS")
    if not ws:
        pytest.skip("ERD_WS not set (run through workspace_setup.sh --test)")
    path = Path(ws) / "install" / "erd_recording" / "lib" / "erd_recording" / script
    interpreter = path.read_text(encoding="utf-8").splitlines()[0].removeprefix("#!")
    # Resolving symlinks would hide a system-Python shebang.
    assert Path(interpreter).parent.parent == Path(ws) / ".venv-erd"

"""Environment assertions every entry point runs first (RR_01 S3.2).

Sourcing ROS Jazzy's ``setup.bash`` (needed for ``ros2 run``/``rclpy``) sets
``PYTHONPATH`` to include ``/opt/ros/jazzy/lib/python3.12/site-packages``.
``PYTHONPATH`` entries are inserted ahead of a venv's own ``site-packages`` in
``sys.path``, so without this fix ``import pinocchio`` silently resolves to
ROS's system build (compiled against NumPy 1.x) instead of ``.venv-erd``'s
`pin==4.1.0` (compiled against NumPy 2.x) -- and segfaults or raises deep
inside the C extension instead of at an ``import`` line, which is why this
module fixes it before anything else runs (found during T1.0: confirmed the
crash is exactly this PYTHONPATH ordering, not a broken install).

Every ``erd_recording`` console entry point does::

    from erd_recording.env_guard import assert_environment
    assert_environment()

before importing anything from ``elastic_sim`` or ``pinocchio``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


class EnvironmentError_(RuntimeError):
    """Raised with a one-line fix, per RR_01 S3.2."""


def _reorder_ros_paths() -> None:
    """Move ``/opt/ros/*/site-packages`` entries to the end of ``sys.path``.

    Only reorders; never removes. ``rclpy`` and friends stay importable (they
    exist nowhere else), but ``.venv-erd``'s own packages -- most importantly
    ``pinocchio`` -- are found first.
    """
    ros_paths = [p for p in sys.path if "/opt/ros/" in p and "site-packages" in p]
    if not ros_paths:
        return
    for path in ros_paths:
        sys.path.remove(path)
    sys.path.extend(ros_paths)


def clean_ros_subprocess_env() -> dict[str, str]:
    """A copy of ``os.environ`` with ``/opt/ros/*`` stripped from
    ``LD_LIBRARY_PATH``/``PYTHONPATH`` (see module docstring: the *second*,
    deeper conflict found at T1.0/T1.6).

    Sourcing ROS's ``setup.bash`` also prepends ``/opt/ros/jazzy/lib*`` to
    ``LD_LIBRARY_PATH``. Unlike the ``sys.path`` ordering issue above, this
    one is not fixable by reordering *inside* an already-running process: the
    dynamic linker resolves a compiled extension's shared-library
    dependencies (here, ``pinocchio``'s bundled ``libeigenpy.so``) using
    ``LD_LIBRARY_PATH`` at ``dlopen`` time, and once ``/opt/ros/jazzy/lib``
    is searched first, a same-named or ABI-incompatible library already
    resolves before this venv's own copy ever gets a chance -- confirmed by
    reproducing `import pinocchio` succeeding with a clean env and failing
    (``undefined symbol: EIGENPY_ARRAY_API...``) with ROS's sourced.
    Every pinocchio-needing stage (``plan``, and the offline pieces of
    ``convert``) is therefore run as a **subprocess** with this cleaned
    environment, from ``erd_recording.plan_cli``/``erd_recording.convert_cli``,
    never in the same process as ``rclpy``.
    """
    env = dict(os.environ)
    for key in ("LD_LIBRARY_PATH", "PYTHONPATH"):
        if key in env:
            kept = [p for p in env[key].split(":") if p and not p.startswith("/opt/ros/")]
            if kept:
                env[key] = ":".join(kept)
            else:
                del env[key]
    return env


def assert_environment(*, require_ros: bool = False, require_pinocchio: bool = True) -> None:
    """Reorder ``sys.path``, then assert the venv and ``elastic_sim`` are sane.

    ``require_pinocchio`` defaults to true for the offline (``plan``/
    ``convert``/``identify``/``validate``) code paths. The ROS orchestration
    module (:mod:`erd_recording.pipeline`) passes ``require_pinocchio=False``:
    it never imports ``pinocchio`` itself (it delegates to a subprocess, see
    :func:`clean_ros_subprocess_env`), and asserting it here would fail for
    the unrelated LD_LIBRARY_PATH reason documented there even though nothing
    in the ROS process actually needs it.
    """
    _reorder_ros_paths()

    venv = Path(sys.prefix)
    if venv.name != ".venv-erd":
        raise EnvironmentError_(
            f"not running inside .venv-erd (sys.prefix={sys.prefix}). Fix: "
            "`source <repo>/workspace_setup.sh` before `ros2 run erd_recording ...` (it activates "
            "<repo>/ros2_ws/.venv-erd), or (from a launch file) set the node's executable to "
            "`<repo>/ros2_ws/.venv-erd/bin/python3 -m erd_recording.cli`."
        )
    try:
        import elastic_sim  # noqa: F401
    except ImportError as exc:
        raise EnvironmentError_(
            "`import elastic_sim` failed. Fix: "
            "`source <repo>/workspace_setup.sh` (it installs the repo editable into .venv-erd)."
        ) from exc
    if require_pinocchio:
        try:
            import pinocchio  # noqa: F401
        except ImportError as exc:
            raise EnvironmentError_(
                "`import pinocchio` failed. Fix: `pip install pin==4.1.0` inside .venv-erd. If ROS's "
                "setup.bash is sourced in this shell, this is the LD_LIBRARY_PATH conflict documented in "
                "`clean_ros_subprocess_env` -- run this command without sourcing it, or through the "
                "*_cli subprocess wrappers that strip it."
            ) from exc
        if "/opt/ros/" in (pinocchio.__file__ or ""):
            raise EnvironmentError_(
                f"pinocchio resolved to {pinocchio.__file__!r} (ROS's system build, numpy-ABI-incompatible "
                "with this venv's numpy). Fix: this should be impossible after `_reorder_ros_paths()`; if "
                "you see this, some other code re-inserted /opt/ros ahead of sys.path[1:] after "
                "`assert_environment()` ran -- call it again immediately before the first `import pinocchio`."
            )
    if require_ros:
        try:
            import rclpy  # noqa: F401
        except ImportError as exc:
            raise EnvironmentError_(
                "`import rclpy` failed. Fix: `source /opt/ros/jazzy/setup.bash` before running this command."
            ) from exc

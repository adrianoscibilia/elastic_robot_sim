"""RR_16 Q-4(i), RR_18 R-4: the code a run was recorded and analysed with.

``code_digest`` (method ``erd-src-digest/2``) hashes everything that runs:

    { find src/ros2 -type f -not -name "*.md" -not -path "*/__pycache__/*"; \\
      echo workspace_setup.sh; \\
      find src/elastic_sim -type f -name "*.py" -not -path "*/__pycache__/*"; } \\
      | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16

run from the repository root. ``/1`` (RR_17 D-11) hashed only ``*.py
*.yaml *.cpp *.hpp *.xml *.sh`` under ``src/ros2``; it left out
``CMakeLists.txt``, ``ErdFri.java``, ``*.msg``, ``erd.repos``, the
requirements, the reference ``*.json``, ``workspace_setup.sh`` and the
``elastic_sim`` loader the converter imports (RR_18 F-17).

The installed workspace runs *copies* and *builds* of the sources, so a
source digest says nothing about a workspace that was pulled but not
rebuilt. Two checks close that:

* the **build stamp**: ``workspace_setup.sh --build`` writes the digest it
  built from to ``ros2_ws/install/.erd_build_digest``
  (``python3 code_digest.py --write-stamp``); a missing stamp or one that
  differs from the source digest refuses. It covers the C++ libraries and
  the installed launch/config files, which no per-file check sees;
* the **per-module check**: every installed Python module of every package
  under ``src/ros2`` that installs one (``erd_recording``, ``erd_ur10``
  -- both ``ament_python`` -- and ``erd_iiwa`` through
  ``ament_cmake_python``) equals its source.

The block also records the commit checked out for each repository of
``src/ros2/erd.repos`` (RR_18 R-4(c)).

``ros2 run erd_recording erd_code_digest`` prints the block (the laptop's
pre-session comparison, OPERATOR.md) and exits 1 on any refusal.

Standard library only: ``workspace_setup.sh`` runs this file with the
system ``python3`` before anything is built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

METHOD = ("erd-src-digest/2: sha256 of the `sha256sum` listing (\"<hex>  <repo-relative path>\") of every file "
          "under src/ros2 except *.md, plus workspace_setup.sh and src/elastic_sim/**/*.py, __pycache__ excluded, "
          "paths in byte order (LC_ALL=C sort); first 16 hex")
STAMP_NAME = ".erd_build_digest"


def digest_files(repo_root: Path) -> dict[bytes, Path]:
    """The hashed files, keyed by their repo-relative path (bytes, for byte order)."""
    repo_root = Path(repo_root)
    files: dict[bytes, Path] = {}

    def add(path: Path) -> None:
        if path.is_file() and "__pycache__" not in path.parts:
            files[path.relative_to(repo_root).as_posix().encode()] = path

    for path in (repo_root / "src" / "ros2").rglob("*"):
        if path.suffix != ".md":
            add(path)
    add(repo_root / "workspace_setup.sh")
    for path in (repo_root / "src" / "elastic_sim").rglob("*.py"):
        add(path)
    return files


def source_digest(repo_root: Path) -> str:
    """:data:`METHOD` over the repository at ``repo_root``."""
    files = digest_files(repo_root)
    listing = b"".join(hashlib.sha256(files[name].read_bytes()).hexdigest().encode() + b"  " + name + b"\n"
                       for name in sorted(files))
    return hashlib.sha256(listing).hexdigest()[:16]


def _is_repo_root(path: Path) -> bool:
    return (path / "workspace_setup.sh").is_file() and \
        (path / "src" / "ros2" / "erd_recording" / "erd_recording" / "code_digest.py").is_file()


def find_repo_root() -> Path | None:
    """The repository this installation was built from: through the
    workspace's ``src/erd`` symlink (``ros2_ws/install/erd_recording/lib/
    python3.X/site-packages/erd_recording/`` -> ``ros2_ws/src/erd`` ->
    ``src/ros2``), else ``$ERD_REPO_ROOT``, else the source tree this module
    is imported from."""
    here = Path(__file__).resolve()
    candidates = []
    if len(here.parents) > 6:
        candidates.append((here.parents[6] / "src" / "erd").resolve().parent.parent)
    if os.environ.get("ERD_REPO_ROOT"):
        candidates.append(Path(os.environ["ERD_REPO_ROOT"]))
    if len(here.parents) > 4:
        candidates.append(here.parents[4])  # <repo>/src/ros2/erd_recording/erd_recording/code_digest.py
    for candidate in candidates:
        if _is_repo_root(candidate):
            return candidate.resolve()
    return None


def find_install_root(repo_root: Path | None) -> Path | None:
    """``ros2_ws/install``: the one this module is installed in, else
    ``$ERD_WS/install``, else ``<repo>/ros2_ws/install``."""
    here = Path(__file__).resolve()
    if len(here.parents) > 5 and here.parents[1].name == "site-packages" and (here.parents[5] / "setup.bash").is_file():
        return here.parents[5]
    for candidate in ([Path(os.environ["ERD_WS"]) / "install"] if os.environ.get("ERD_WS") else []) + \
            ([repo_root / "ros2_ws" / "install"] if repo_root is not None else []):
        if candidate.is_dir():
            return candidate
    return None


def python_packages(src_ros2: Path) -> list[str]:
    """Packages under ``src/ros2`` that install a Python package of their name."""
    return sorted(path.parent.parent.name for path in Path(src_ros2).glob("*/*/__init__.py")
                  if path.parent.name == path.parent.parent.name)


def _modules(package_dir: Path) -> dict[str, Path]:
    return {path.relative_to(package_dir).as_posix(): path for path in package_dir.rglob("*.py")
            if "__pycache__" not in path.parts}


def install_check(src_ros2: Path, install_root: Path | None) -> dict[str, Any]:
    """Does every installed Python module of every package equal its source?
    Per package: the modules that differ or exist on one side only."""
    if install_root is None:
        return {"ok": False, "note": "no ros2_ws/install found"}
    packages: dict[str, Any] = {}
    for name in python_packages(src_ros2):
        source = Path(src_ros2) / name / name
        installed_dirs = sorted(Path(install_root).glob(f"{name}/lib/python3*/site-packages/{name}"))
        if not installed_dirs:
            packages[name] = {"ok": False, "note": "not installed"}
            continue
        installed, sources = _modules(installed_dirs[0]), _modules(source)
        differ = sorted(m for m in installed.keys() & sources.keys()
                        if installed[m].read_bytes() != sources[m].read_bytes())
        packages[name] = {"ok": not differ and installed.keys() == sources.keys(), "installed": str(installed_dirs[0]),
                          "differ": differ, "only_installed": sorted(installed.keys() - sources.keys()),
                          "only_source": sorted(sources.keys() - installed.keys())}
    return {"ok": bool(packages) and all(p["ok"] for p in packages.values()), "packages": packages}


def build_stamp(install_root: Path | None, digest: str) -> dict[str, Any]:
    """The digest ``workspace_setup.sh --build`` built from, against ``digest``."""
    path = None if install_root is None else Path(install_root) / STAMP_NAME
    value = path.read_text(encoding="utf-8").strip() if path is not None and path.is_file() else None
    return {"path": None if path is None else str(path), "value": value, "ok": value == digest}


def write_stamp(repo_root: Path, install_root: Path, digest: str | None = None) -> str:
    """Write the build stamp (``workspace_setup.sh`` after a successful
    build, with the digest computed *before* the build)."""
    digest = digest or source_digest(repo_root)
    (Path(install_root) / STAMP_NAME).write_text(digest + "\n", encoding="utf-8")
    return digest


def external_repos(repo_root: Path, workspace: Path | None) -> dict[str, Any]:
    """RR_18 R-4(c): per ``src/ros2/erd.repos`` entry, the pinned version and
    the commit checked out under ``ros2_ws/src/external/<name>`` (``git
    rev-parse HEAD``), and whether tracked files differ from it."""
    repos_file = Path(repo_root) / "src" / "ros2" / "erd.repos"
    if not repos_file.is_file():
        return {}
    import yaml  # not needed for --write-stamp, which must run before the venv exists

    entries = (yaml.safe_load(repos_file.read_text(encoding="utf-8")) or {}).get("repositories") or {}
    result = {}
    for name, spec in entries.items():
        checkout = None if workspace is None else Path(workspace) / "src" / "external" / name
        head = dirty = None
        if checkout is not None and (checkout / ".git").exists():
            run = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True)
            head = (run.stdout.strip() or None) if run.returncode == 0 else None
            status = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True)
            dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        pinned = str(spec.get("version"))
        result[name] = {"url": spec.get("url"), "pinned": pinned, "checked_out": head,
                        "matches_pin": head is not None and head.startswith(pinned), "dirty_tracked_files": dirty,
                        "path": None if checkout is None else str(checkout)}
    return result


def _problems(block: dict[str, Any]) -> list[str]:
    problems = []
    if block["value"] is None:
        problems.append("the repository (workspace_setup.sh, src/ros2) was not found next to this installation")
        return problems
    stamp = block["build_stamp"]
    if not stamp["ok"]:
        problems.append(f"build stamp {stamp['path']} is {stamp['value'] or 'missing'}, the source digest is "
                        f"{block['value']}")
    stale = {name: {k: v for k, v in entry.items() if k != "ok" and v}
             for name, entry in block["install"].get("packages", {}).items() if not entry["ok"]}
    if stale or not block["install"]["ok"]:
        problems.append(f"installed Python modules differ from src/ros2: {stale or block['install']}")
    return problems


def code_digest() -> dict[str, Any]:
    """The block every run's ``manifest.yaml`` and ``validation.json`` carry.
    ``ok`` is false (and ``problems`` says why) when the build stamp is
    missing or stale, or an installed module differs from its source."""
    repo_root = find_repo_root()
    if repo_root is None:
        block = {"value": None, "method": METHOD, "source": None, "install": {"ok": False},
                 "build_stamp": {"ok": False}, "external_repos": {}}
    else:
        install_root = find_install_root(repo_root)
        value = source_digest(repo_root)
        block = {"value": value, "method": METHOD, "source": str(repo_root),
                 "install": install_check(repo_root / "src" / "ros2", install_root),
                 "build_stamp": build_stamp(install_root, value),
                 "external_repos": external_repos(repo_root, None if install_root is None else install_root.parent)}
    block["problems"] = _problems(block)
    block["ok"] = not block["problems"]
    return block


def refusal(block: dict[str, Any]) -> str | None:
    """The refusal message for ``record_*``, ``erd_link_test``,
    ``erd_monitor_test`` and ``erd_code_digest``, or ``None``."""
    if block["ok"]:
        return None
    return "; ".join(block["problems"]) + "; rebuild with `source workspace_setup.sh --build`"


def main(argv: list[str] | None = None) -> int:
    """``erd_code_digest``: print the block; exit 1 on a refusal.

    ``--write-stamp INSTALL_DIR --digest D`` (``workspace_setup.sh``): write
    the build stamp. ``--source-digest``: print the bare source digest."""
    parser = argparse.ArgumentParser(prog="erd_code_digest")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--source-digest", action="store_true")
    parser.add_argument("--write-stamp", metavar="INSTALL_DIR", default=None)
    parser.add_argument("--digest", default=None)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.source_digest or args.write_stamp:
        repo_root = Path(args.repo_root) if args.repo_root else find_repo_root()
        if repo_root is None or not _is_repo_root(repo_root):
            print("erd_code_digest: repository not found", file=sys.stderr)
            return 2
        if args.write_stamp:
            print(write_stamp(repo_root, Path(args.write_stamp), args.digest))
        else:
            print(source_digest(repo_root))
        return 0
    block = code_digest()
    print(json.dumps(block, indent=1))
    stamp = block["build_stamp"].get("value") or "missing"
    print(f"method {block['method'].split(':')[0]}  code_digest {block['value']}  build stamp {stamp} "
          f"({'matches' if block['build_stamp']['ok'] else 'DIFFERS'})  installed modules "
          f"{'match' if block['install']['ok'] else 'DIFFER'}", file=sys.stderr)
    message = refusal(block)
    if message:
        print(f"REFUSED: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

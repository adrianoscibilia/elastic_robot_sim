#!/usr/bin/env python3
"""Compile a ROS 2 robot-description xacro package into a flat URDF.

The output URDF contains absolute mesh paths so that Newton (or any other
loader that does not understand ``package://`` URIs) can find the mesh files
without a ROS workspace.

Typical usage
-------------
::

    python scripts/compile_urdf_from_description.py ^
        "C:\\Users\\adria\\Projects\\Universal_Robots_ROS2_Description" ^
        --output-name ur10 ^
        ur_type:=ur10 name:=ur10

This produces ``urdf/ur10.urdf`` with absolute mesh paths pointing back to the
original description package.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
import tempfile
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# xacro import — must be installed (``pip install xacro``)
# ---------------------------------------------------------------------------
try:
    import xacro
except ImportError:
    sys.exit(
        "The 'xacro' Python package is required.\n"
        "Install it with:  pip install xacro"
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_package_name(pkg_path: Path) -> str:
    """Parse ``package.xml`` and return the ``<name>`` text."""
    package_xml = pkg_path / "package.xml"
    if not package_xml.is_file():
        raise FileNotFoundError(
            f"No package.xml found in {pkg_path}.  "
            "Is this a valid ROS 2 description package?"
        )
    tree = ET.parse(package_xml)  # noqa: S314
    name_elem = tree.getroot().find("name")
    if name_elem is None or not name_elem.text:
        raise ValueError("package.xml does not contain a <name> element.")
    return name_elem.text.strip()


def _auto_detect_entry_xacro(pkg_path: Path) -> str:
    """Return the relative path (inside the package) of the main xacro file.

    Heuristic: look for ``urdf/*.urdf.xacro`` and pick the shortest name that
    does *not* contain ``mocked`` or ``mock``.
    """
    urdf_dir = pkg_path / "urdf"
    if not urdf_dir.is_dir():
        raise FileNotFoundError(
            f"No 'urdf/' directory in {pkg_path}. "
            "Specify --entry-xacro explicitly."
        )
    candidates = sorted(
        urdf_dir.glob("*.urdf.xacro"),
        key=lambda p: len(p.name),
    )
    candidates = [
        c for c in candidates
        if "mock" not in c.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No *.urdf.xacro files found in {urdf_dir}. "
            "Specify --entry-xacro explicitly."
        )
    return candidates[0].relative_to(pkg_path).as_posix()


def _copy_package_light(pkg_path: Path, tmp_dir: Path) -> Path:
    """Copy the package into *tmp_dir*, skipping heavy/unnecessary dirs."""
    skip_dirs = {".git", "__pycache__", "meshes"}
    dest = tmp_dir / pkg_path.name

    def _ignore(directory: str, contents: list[str]) -> list[str]:
        return [c for c in contents if c in skip_dirs]

    shutil.copytree(pkg_path, dest, ignore=_ignore)
    return dest


def _resolve_find_in_xacros(pkg_copy: Path, pkg_name: str, original_pkg_path: Path) -> None:
    """In-place replace ``$(find <pkg_name>)`` in every ``.xacro`` file.

    The replacement points to *pkg_copy* for include / load_yaml resolution
    during xacro processing.
    """
    copy_posix = pkg_copy.as_posix()
    pattern = re.compile(re.escape(f"$(find {pkg_name})"))

    for xacro_file in pkg_copy.rglob("*.xacro"):
        text = xacro_file.read_text(encoding="utf-8")
        new_text = pattern.sub(copy_posix, text)
        if new_text != text:
            xacro_file.write_text(new_text, encoding="utf-8")


def _resolve_package_uris(urdf_xml: str, pkg_name: str, original_pkg_path: Path) -> str:
    """Replace ``package://<pkg_name>/…`` with absolute file paths."""
    original_posix = original_pkg_path.as_posix()
    # package://ur_description/meshes/…  →  C:/Users/…/meshes/…
    pattern = re.compile(re.escape(f"package://{pkg_name}/"))
    return pattern.sub(original_posix + "/", urdf_xml)


def _resolve_file_uris(urdf_xml: str, pkg_copy: Path, original_pkg_path: Path) -> str:
    """Replace leftover ``file://<pkg_copy>/…`` with original absolute paths."""
    copy_posix = pkg_copy.as_posix()
    original_posix = original_pkg_path.as_posix()
    return urdf_xml.replace(f"file://{copy_posix}/", f"{original_posix}/")


def _validate_mesh_paths(urdf_xml: str) -> list[str]:
    """Return a list of mesh filenames referenced in the URDF that do not exist."""
    missing = []
    for m in re.finditer(r'filename\s*=\s*"([^"]+)"', urdf_xml):
        fpath = m.group(1)
        # Skip package:// or http(s):// — those weren't resolved (multi-pkg)
        if fpath.startswith(("package://", "http://", "https://")):
            continue
        # Normalise forward-slash path to OS path
        if not os.path.isfile(fpath.replace("/", os.sep)):
            missing.append(fpath)
    return missing


# ---------------------------------------------------------------------------
# Main compilation
# ---------------------------------------------------------------------------

def compile_urdf(
    pkg_path: Path,
    entry_xacro: str | None,
    output_name: str | None,
    output_dir: Path,
    xacro_mappings: dict[str, str],
) -> Path:
    """Compile a robot-description xacro package into a flat URDF file.

    Returns the path to the written URDF.
    """
    pkg_path = pkg_path.resolve()
    if not pkg_path.is_dir():
        raise NotADirectoryError(f"Package path does not exist: {pkg_path}")

    pkg_name = _read_package_name(pkg_path)
    print(f"Package name: {pkg_name}")

    # --- entry xacro ---
    if entry_xacro is None:
        entry_xacro = _auto_detect_entry_xacro(pkg_path)
    print(f"Entry xacro:  {entry_xacro}")

    # --- output name ---
    if output_name is None:
        # Derive from first mapping value or package name
        output_name = next(iter(xacro_mappings.values()), None) or pkg_name
    output_path = (output_dir / output_name).with_suffix(".urdf")

    # --- Temp copy + $(find) resolution ---
    with tempfile.TemporaryDirectory(prefix="compile_urdf_") as tmp_str:
        tmp_dir = Path(tmp_str)
        pkg_copy = _copy_package_light(pkg_path, tmp_dir)
        _resolve_find_in_xacros(pkg_copy, pkg_name, pkg_path)

        # --- Xacro processing ---
        entry_path = pkg_copy / entry_xacro
        if not entry_path.is_file():
            raise FileNotFoundError(
                f"Entry xacro not found: {entry_path}\n"
                f"  (resolved from --entry-xacro={entry_xacro})"
            )

        print(f"Processing xacro with mappings: {xacro_mappings}")
        doc = xacro.process_file(
            str(entry_path),
            mappings=xacro_mappings,
        )
        urdf_xml = doc.toprettyxml(indent="  ")

    # --- Post-process: resolve package:// and file:// URIs ---
    urdf_xml = _resolve_package_uris(urdf_xml, pkg_name, pkg_path)
    urdf_xml = _resolve_file_uris(urdf_xml, pkg_copy, pkg_path)

    # --- Validate mesh paths ---
    missing = _validate_mesh_paths(urdf_xml)
    if missing:
        print(f"\nWARNING: {len(missing)} mesh file(s) not found on disk:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("All mesh paths validated OK.")

    # --- Write output ---
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(urdf_xml, encoding="utf-8")
    print(f"\nCompiled URDF written to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict[str, str]]:
    parser = argparse.ArgumentParser(
        description="Compile a ROS 2 robot-description xacro package into a flat URDF.",
        epilog=textwrap.dedent("""\
            Xacro mappings are passed as trailing key:=value arguments, e.g.:
              %(prog)s /path/to/pkg ur_type:=ur10 name:=ur10
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pkg_path",
        type=Path,
        help="Path to the robot description package (must contain package.xml).",
    )
    parser.add_argument(
        "--entry-xacro",
        default=None,
        help="Relative path (inside package) to the main xacro file. "
             "Auto-detected from urdf/*.urdf.xacro if omitted.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename stem (e.g. 'ur10'). Defaults to the first "
             "xacro mapping value or the package name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "urdf",
        help="Directory for the output URDF (default: <project>/urdf/).",
    )

    # Split known args from xacro key:=value mappings
    known, remaining = parser.parse_known_args(argv)

    mappings: dict[str, str] = {}
    for arg in remaining:
        if ":=" in arg:
            key, _, value = arg.partition(":=")
            mappings[key] = value
        else:
            parser.error(f"Unrecognised argument: {arg!r}  (xacro mappings use key:=value)")

    return known, mappings


def main(argv: list[str] | None = None) -> None:
    args, mappings = _parse_args(argv)
    compile_urdf(
        pkg_path=args.pkg_path,
        entry_xacro=args.entry_xacro,
        output_name=args.output_name,
        output_dir=args.output_dir,
        xacro_mappings=mappings,
    )


if __name__ == "__main__":
    main()

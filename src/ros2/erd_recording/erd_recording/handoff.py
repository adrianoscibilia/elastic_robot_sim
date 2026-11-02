"""RR_23 C7: the ``real_files.yaml`` entry of one accepted run.

``record_<robot> ... handoff --run-dir <run> --split <test|modal_reference|calibration>``
prints the entry (robot, id, dataset path, split, sha256); ``--write`` appends
it to ``REFACTOR_SPECS/identified_filter/real_files.yaml``. The owner runs it,
only for files the session accepted.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .config import LabConfig

SPLITS = ("test", "modal_reference", "calibration")
ROBOT_LABEL = {"kuka_lbr_iiwa_14_r820": "iiwa", "ur10_cb3": "ur10"}
REAL_FILES = Path("REFACTOR_SPECS") / "identified_filter" / "real_files.yaml"


class HandoffError(ValueError):
    pass


def repo_root() -> Path:
    root = os.environ.get("ERD_REPO_ROOT")
    if not root:
        raise HandoffError("ERD_REPO_ROOT is not set (source workspace_setup.sh)")
    return Path(root)


def _sha256(path: Path) -> str:
    """A file's sha256; for a folder, over its files' relative names and bytes, sorted."""
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode() + b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _calibration_source(run_dir: Path) -> tuple[Path, str] | None:
    """What a ``calibration`` file without a dataset hands off: the run's raw
    frame, the identify-only raw frame (the mount check), or the standstill bag."""
    for candidate, fmt in ((run_dir / "raw.parquet", "raw_parquet"),
                           (run_dir / "identification" / "raw.parquet", "identification_raw_parquet"),
                           (run_dir / "bag_standstill", "standstill_bag")):
        if candidate.exists():
            return candidate, fmt
    return None


def entry_for(config: LabConfig, run_dir: Path, split: str, entry_id: str | None = None) -> dict[str, str]:
    """The entry. A dataset must exist, ``validation.json`` must say ``valid``
    (``synthetic`` off real hardware) and its segments must carry ``split``.
    ``calibration`` runs without a dataset (standstill, mount check) hand off
    their raw frame or standstill bag instead, with ``format`` saying which."""
    if split not in SPLITS:
        raise HandoffError(f"--split must be one of {SPLITS}, got {split!r}")
    run_dir = Path(run_dir).resolve()
    dataset = run_dir / "dataset" / f"{config.robot}.parquet"
    fmt = "dataset"
    if dataset.is_file():
        validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
        expected = "valid" if config.hardware == "real" else "synthetic"
        if validation.get("status") != expected:
            raise HandoffError(f"{run_dir.name}: validation status {validation.get('status')!r}, not {expected!r}")
        contract = json.loads(dataset.with_suffix(".contract.json").read_text(encoding="utf-8"))
        splits = sorted({s.get("split") for s in contract.get("segments", [])})
        if splits != [split]:
            raise HandoffError(f"{run_dir.name}: the dataset's segments are {splits}, not [{split!r}]")
        source = dataset
    elif split == "calibration" and _calibration_source(run_dir) is not None:
        source, fmt = _calibration_source(run_dir)
    else:
        raise HandoffError(f"{run_dir}: no dataset ({dataset.name}); `convert` first. Ladder runs never have one")
    try:
        path = source.relative_to(repo_root().resolve())
    except ValueError:
        path = source
    entry = {"robot": ROBOT_LABEL[config.robot], "id": entry_id or run_dir.name, "path": str(path),
             "split": split, "sha256": _sha256(source)}
    if fmt != "dataset":
        entry["format"] = fmt
    return entry


def render(entry: dict[str, str]) -> str:
    keys = [k for k in ("robot", "id", "path", "split", "sha256", "format") if k in entry]
    return "  - " + "\n    ".join(f"{k}: {entry[k]}" for k in keys) + "\n"


def append(entry: dict[str, str], real_files: Path) -> None:
    """Append under ``files:`` (an empty ``files: []`` becomes a block list);
    an entry whose id is already listed is refused."""
    text = real_files.read_text(encoding="utf-8")
    if f"id: {entry['id']}\n" in text:
        raise HandoffError(f"{real_files}: {entry['id']} is already listed")
    if "\nfiles: []" in text:
        text = text.replace("\nfiles: []", "\nfiles:", 1)
    elif "\nfiles:" not in text:
        raise HandoffError(f"{real_files}: no top-level `files:` key")
    if not text.endswith("\n"):
        text += "\n"
    real_files.write_text(text + render(entry), encoding="utf-8")


def handoff(config: LabConfig, run_dir: Path, *, split: str | None, entry_id: str | None = None,
            write: bool = False, real_files: Path | None = None) -> dict[str, str]:
    if split is None:
        raise HandoffError("handoff needs --split <test|modal_reference|calibration>")
    entry = entry_for(config, run_dir, split, entry_id)
    print(render(entry), end="")
    if write:
        target = real_files or repo_root() / REAL_FILES
        append(entry, target)
        print(f"appended to {target}")
    return entry

"""UR_11 ``--keep-derived-raw``: an offline convert on a copied run folder
reads its raw.parquet/events.json/rtde_alignment.json/recorder_losses*.json
as they are. ``pipeline`` imports rclpy, so this file runs only in a
ROS-sourced shell (skipped in the ROS-free erd_recording suite)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
from erd_recording import pipeline  # noqa: E402


def _paths(tmp_path: Path, *, raw: bool) -> pipeline.RunPaths:
    (tmp_path / "plan").mkdir()
    if raw:
        (tmp_path / "raw.parquet").write_bytes(b"original")
        (tmp_path / "events.json").write_text("[]")
    return pipeline.RunPaths(tmp_path)


def test_keep_derived_raw_converts_without_rederiving_the_raw_files(tmp_path, monkeypatch):
    # UR_11: on a copied run folder raw.parquet & co. are the originals.
    paths = _paths(tmp_path, raw=True)
    calls = []
    monkeypatch.setattr(pipeline, "_prepare_conversion", lambda *a: calls.append("prepare"))
    monkeypatch.setattr(pipeline, "_run_convert_cli", lambda *a: calls.append("convert") or {"status": "valid"})
    config = SimpleNamespace(robot="ur10_cb3")
    assert pipeline.stage_convert(config, "c.yaml", paths, reference_contract=None, keep_derived_raw=True) \
        == {"status": "valid"}
    assert calls == ["convert"]
    assert (tmp_path / "raw.parquet").read_bytes() == b"original"
    calls.clear()
    pipeline.stage_convert(config, "c.yaml", paths, reference_contract=None)  # default: unchanged behaviour
    assert calls == ["prepare", "convert"]


def test_keep_derived_raw_refuses_a_folder_without_raw_files(tmp_path, monkeypatch):
    paths = _paths(tmp_path, raw=False)
    monkeypatch.setattr(pipeline, "_prepare_conversion", lambda *a: pytest.fail("must not re-derive"))
    with pytest.raises(pipeline.PipelineError, match="--keep-derived-raw"):
        pipeline.stage_convert(SimpleNamespace(robot="ur10_cb3"), "c.yaml", paths, reference_contract=None,
                               keep_derived_raw=True)


def test_keep_derived_raw_needs_run_dir():
    config = Path(__file__).resolve().parents[1] / "config" / "lab" / "ur10_mock.yaml"
    with pytest.raises(pipeline.PipelineError, match="--run-dir"):
        pipeline.main_ur10(["--config", str(config), "convert", "--keep-derived-raw"])

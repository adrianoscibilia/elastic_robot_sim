"""RR_16 Q-4(i), RR_18 R-4: the run's code digest (``erd-src-digest/2``),
the build stamp, the per-package install check and the ``erd.repos``
commits."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from erd_recording import code_digest as cd


def _repo(root):
    """A miniature repository with one file of every kind R-4 names."""
    src = root / "src" / "ros2"
    files = {
        "src/ros2/erd_recording/erd_recording/__init__.py": "",
        "src/ros2/erd_recording/erd_recording/pipeline.py": "x = 1\n",
        "src/ros2/erd_recording/erd_recording/code_digest.py": "y = 1\n",
        "src/ros2/erd_recording/erd_recording/__pycache__/skip.py": "ignored\n",
        "src/ros2/erd_ur10/erd_ur10/__init__.py": "",
        "src/ros2/erd_ur10/erd_ur10/rtde_logger.py": "z = 1\n",
        "src/ros2/erd_iiwa/erd_iiwa/__init__.py": "",
        "src/ros2/erd_iiwa/CMakeLists.txt": "project(erd_iiwa)\n",
        "src/ros2/erd_iiwa/src/Fri_system.cpp": "int main() {}\n",
        "src/ros2/erd_iiwa/sunrise/ErdFri.java": "class ErdFri {}\n",
        "src/ros2/erd_msgs/msg/Event.msg": "string kind\n",
        "src/ros2/erd.repos": "repositories: {}\n",
        "src/ros2/README.md": "not hashed\n",
        "src/elastic_sim/loader.py": "LOADER = 1\n",
        "src/elastic_sim/assets/notes.txt": "not hashed (not *.py)\n",
        "workspace_setup.sh": "# setup\n",
        "README.md": "outside the digest\n",
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return src


def _install(root, src):
    """``ros2_ws/install`` with a copy of every Python package, as colcon builds it."""
    install = root / "ros2_ws" / "install"
    for name in cd.python_packages(src):
        target = install / name / "lib" / "python3.12" / "site-packages" / name
        shutil.copytree(src / name / name, target, ignore=shutil.ignore_patterns("__pycache__"))
    return install


@pytest.mark.skipif(shutil.which("sha256sum") is None, reason="needs coreutils")
def test_digest_equals_the_shell_command_in_byte_order(tmp_path):
    _repo(tmp_path)
    shell = subprocess.run(
        '{ find src/ros2 -type f -not -name "*.md" -not -path "*/__pycache__/*"; echo workspace_setup.sh; '
        'find src/elastic_sim -type f -name "*.py" -not -path "*/__pycache__/*"; } '
        '| LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16',
        shell=True, cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()
    assert cd.source_digest(tmp_path) == shell


@pytest.mark.parametrize("edited", ["src/ros2/erd_iiwa/CMakeLists.txt", "src/ros2/erd_msgs/msg/Event.msg",
                                    "src/elastic_sim/loader.py", "src/ros2/erd_iiwa/sunrise/ErdFri.java",
                                    "src/ros2/erd.repos", "workspace_setup.sh"])
def test_an_edit_to_anything_that_runs_changes_the_digest(tmp_path, edited):
    _repo(tmp_path)
    before = cd.source_digest(tmp_path)
    (tmp_path / edited).write_text("changed\n")
    assert cd.source_digest(tmp_path) != before


def test_markdown_caches_and_non_python_elastic_sim_files_are_not_hashed(tmp_path):
    _repo(tmp_path)
    before = cd.source_digest(tmp_path)
    for name in ("src/ros2/README.md", "src/ros2/erd_recording/erd_recording/__pycache__/skip.py",
                 "src/elastic_sim/assets/notes.txt", "README.md"):
        (tmp_path / name).write_text("changed\n")
    assert cd.source_digest(tmp_path) == before


def _block(monkeypatch, root, install):
    monkeypatch.setattr(cd, "find_repo_root", lambda: root)
    monkeypatch.setattr(cd, "find_install_root", lambda repo_root: install)
    return cd.code_digest()


def test_a_source_change_without_rebuild_refuses_naming_the_stamp(tmp_path, monkeypatch):
    src = _repo(tmp_path)
    install = _install(tmp_path, src)
    missing = _block(monkeypatch, tmp_path, install)
    assert not missing["ok"] and ".erd_build_digest is missing" in cd.refusal(missing)
    cd.write_stamp(tmp_path, install)                       # workspace_setup.sh --build
    assert _block(monkeypatch, tmp_path, install)["ok"]
    (tmp_path / "src/ros2/erd_iiwa/CMakeLists.txt").write_text("project(erd_iiwa) # pulled\n")
    stale = _block(monkeypatch, tmp_path, install)
    message = cd.refusal(stale)
    assert not stale["ok"] and ".erd_build_digest is " in message and stale["value"] in message
    assert "workspace_setup.sh --build" in message
    cd.write_stamp(tmp_path, install)                       # the rebuild clears it
    assert cd.refusal(_block(monkeypatch, tmp_path, install)) is None


def test_every_python_package_is_checked_module_by_module(tmp_path, monkeypatch):
    src = _repo(tmp_path)
    assert cd.python_packages(src) == ["erd_iiwa", "erd_recording", "erd_ur10"]
    install = _install(tmp_path, src)
    cd.write_stamp(tmp_path, install)
    (src / "erd_ur10" / "erd_ur10" / "rtde_logger.py").write_text("z = 2\n")  # pulled, not rebuilt ...
    cd.write_stamp(tmp_path, install)                   # ... even with a (forged) current stamp
    block = _block(monkeypatch, tmp_path, install)
    assert not block["ok"] and block["build_stamp"]["ok"]
    assert block["install"]["packages"]["erd_ur10"]["differ"] == ["rtde_logger.py"]
    assert block["install"]["packages"]["erd_recording"]["ok"]
    (src / "erd_iiwa" / "erd_iiwa" / "description.py").write_text("new\n")
    assert cd.install_check(src, install)["packages"]["erd_iiwa"]["only_source"] == ["description.py"]


def test_write_stamp_cli_writes_the_digest_built_from(tmp_path):
    _repo(tmp_path)
    install = tmp_path / "ros2_ws" / "install"
    install.mkdir(parents=True)
    assert cd.main(["--repo-root", str(tmp_path), "--write-stamp", str(install), "--digest", "0123456789abcdef"]) == 0
    assert (install / cd.STAMP_NAME).read_text().strip() == "0123456789abcdef"
    assert cd.main(["--repo-root", str(tmp_path), "--write-stamp", str(install)]) == 0
    assert (install / cd.STAMP_NAME).read_text().strip() == cd.source_digest(tmp_path)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_external_repos_record_the_checked_out_commit(tmp_path):
    _repo(tmp_path)
    checkout = tmp_path / "ros2_ws" / "src" / "external" / "iiwa_ros2"
    checkout.mkdir(parents=True)
    git = ["git", "-C", str(checkout), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + ["init", "-q"], check=True)
    (checkout / "file.txt").write_text("a\n")
    subprocess.run(git + ["add", "file.txt"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "c"], check=True)
    head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "src/ros2/erd.repos").write_text(
        f"repositories:\n  iiwa_ros2:\n    type: git\n    url: https://example.invalid/iiwa_ros2.git\n"
        f"    version: {head}\n  missing_repo:\n    type: git\n    url: u\n    version: abc\n")
    repos = cd.external_repos(tmp_path, tmp_path / "ros2_ws")
    assert repos["iiwa_ros2"]["checked_out"] == head and repos["iiwa_ros2"]["matches_pin"]
    assert repos["iiwa_ros2"]["dirty_tracked_files"] is False
    assert repos["missing_repo"]["checked_out"] is None and not repos["missing_repo"]["matches_pin"]
    (checkout / "file.txt").write_text("b\n")
    assert cd.external_repos(tmp_path, tmp_path / "ros2_ws")["iiwa_ros2"]["dirty_tracked_files"] is True


def test_block_names_its_method():
    block = cd.code_digest()
    assert block["method"].startswith("erd-src-digest/2") and "LC_ALL=C" in block["method"]
    assert len(block["value"]) == 16 and "iiwa_ros2" in block["external_repos"]

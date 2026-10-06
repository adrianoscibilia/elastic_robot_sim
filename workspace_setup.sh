# shellcheck shell=bash
# workspace_setup.sh -- build and source the ROS 2 real-data stack (src/ros2)
# entirely inside this repository.
#
# New machine (Ubuntu 24.04), two lines:
#   git clone https://github.com/adrianoscibilia/elastic_robot_sim.git && cd elastic_robot_sim
#   source workspace_setup.sh --deps
#
# Every new terminal afterwards (fast: builds only if nothing is built yet):
#   source <repo>/workspace_setup.sh
#
# Options (combine freely):
#   --deps    install system packages with sudo: ROS 2 Jazzy if missing, the
#             stack's apt/rosdep dependencies, python3-venv, git, netcat
#   --build   incremental `colcon build` before sourcing
#   --clean   delete ros2_ws/{build,install,log} and rebuild from scratch
#   --test    run both test invocations (ROS-sourced packages, then the
#             ROS-free erd_recording suite) after building
#   --real    ROS_DOMAIN_ID=0 for real robots (default: 87, the domain every
#             mock/emulator/URSim config requires, RR_01 S9)
#   --help    print this header
#
# Everything generated lives in the repo but is gitignored:
#   ros2_ws/                colcon workspace: src/erd -> ../../src/ros2 (symlink),
#                           src/external/ (pinned checkouts from src/ros2/erd.repos),
#                           build/ install/ log/, .venv-erd/ (Python 3.12,
#                           --system-site-packages, src/ros2/requirements-erd.txt)
#   data/real_robot/        recording output root (ERD_DATA_ROOT)
#
# Exported for the lab configs and tools:
#   ERD_REPO_ROOT ERD_WS ERD_DATA_ROOT ERD_LAB ERD_FRI_SDK_ROOT
#   ERD_CONSUMER_REPO ERD_CONSUMER_PYTHON   (default: ../dynamic_model_nn next
#                                            to this repo and its .venv; set
#                                            them before sourcing to override)
#
# KUKA's libFRI is compiled from the pinned iiwa_ros2 checkout under
# ros2_ws/src/external/ and is never copied into this repository.

_erd_log()  { printf '\033[1;34m[erd]\033[0m %s\n' "$*"; }
_erd_warn() { printf '\033[1;33m[erd] warning:\033[0m %s\n' "$*" >&2; }
_erd_err()  { printf '\033[1;31m[erd] error:\033[0m %s\n' "$*" >&2; }

_erd_install_system_deps() {
  local codename
  codename="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")"
  sudo apt-get update || return 1
  sudo apt-get install -y software-properties-common curl git python3-venv python3-pip \
    netcat-openbsd || return 1
  if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
    _erd_log "installing ROS 2 Jazzy (ros-base + ros-dev-tools)"
    sudo add-apt-repository -y universe || return 1
    local version
    version="$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
      | grep -F '"tag_name"' | awk -F'"' '{print $4}')"
    [[ -n "$version" ]] || { _erd_err "could not read the ros-apt-source release"; return 1; }
    curl -L -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${version}/ros2-apt-source_${version}.${codename}_all.deb" \
      || return 1
    sudo dpkg -i /tmp/ros2-apt-source.deb || return 1
    sudo apt-get update || return 1
    sudo apt-get install -y ros-jazzy-ros-base ros-dev-tools || return 1
  fi
  sudo apt-get install -y \
    ros-dev-tools python3-pytest python3-yaml \
    ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
    ros-jazzy-ur-robot-driver ros-jazzy-ur-description ros-jazzy-ur-client-library \
    ros-jazzy-xacro ros-jazzy-pinocchio ros-jazzy-rosbag2-storage-mcap || return 1
  if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    sudo rosdep init || return 1
  fi
  rosdep update --rosdistro jazzy || return 1
}

_erd_fetch_external() {
  # Clone every entry of src/ros2/erd.repos at its pinned version into
  # ros2_ws/src/external/. An existing checkout is never modified; a version
  # mismatch is only reported.
  local repos_file="$ERD_REPO_ROOT/src/ros2/erd.repos"
  local external="$ERD_WS/src/external"
  mkdir -p "$external"
  local name url version
  while read -r name url version; do
    [[ -n "$name" ]] || continue
    if [[ ! -d "$external/$name/.git" ]]; then
      _erd_log "cloning $name @ ${version:0:12}"
      git clone --quiet "$url" "$external/$name" || return 1
      git -C "$external/$name" -c advice.detachedHead=false checkout --quiet "$version" || return 1
    else
      local head
      head="$(git -C "$external/$name" rev-parse HEAD)"
      if [[ "$head" != "$version"* ]]; then
        _erd_warn "$external/$name is at ${head:0:12}, erd.repos pins ${version:0:12} (left as is)"
      fi
    fi
  done < <(/usr/bin/python3 - "$repos_file" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
for name, spec in (data.get("repositories") or {}).items():
    print(name, spec["url"], spec["version"])
PY
  )
  # iiwa_ros2 packages this stack does not build (MoveIt/bringup pull heavy
  # dependencies and are not used, RR_01 S1.1).
  local pkg
  for pkg in iiwa_bringup iiwa_controllers iiwa_description_moveit_config; do
    [[ -d "$external/iiwa_ros2/$pkg" ]] && touch "$external/iiwa_ros2/$pkg/COLCON_IGNORE"
  done
  return 0
}

_erd_setup_venv() {
  local venv="$ERD_WS/.venv-erd"
  local req="$ERD_REPO_ROOT/src/ros2/requirements-erd.txt"
  local stamp="$venv/.erd_requirements.sha256"
  if ! /usr/bin/python3 -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
    _erd_err "/usr/bin/python3 must be 3.12 (the interpreter ROS 2 Jazzy's rclpy is built for)"
    return 1
  fi
  if [[ ! -x "$venv/bin/python" ]]; then
    _erd_log "creating $venv (system Python 3.12, --system-site-packages for rclpy)"
    /usr/bin/python3 -m venv --system-site-packages "$venv" || return 1
  fi
  local want have=""
  want="$(sha256sum "$req" | cut -d' ' -f1)"
  [[ -f "$stamp" ]] && have="$(cat "$stamp")"
  if [[ "$want" != "$have" ]] || ! "$venv/bin/python" -m pip show -q elastic-robot-sim >/dev/null 2>&1; then
    _erd_log "installing Python requirements into .venv-erd"
    "$venv/bin/python" -m pip install --quiet --upgrade pip setuptools wheel || return 1
    "$venv/bin/python" -m pip install --quiet -r "$req" || return 1
    # elastic_sim itself, editable, without the simulation's own dependency
    # set (MuJoCo/Newton/warp live in the repo's uv .venv, not here).
    "$venv/bin/python" -m pip install --quiet --no-deps --no-build-isolation -e "$ERD_REPO_ROOT" || return 1
    echo "$want" > "$stamp"
  fi
}

_erd_build() {
  (
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    # shellcheck disable=SC1091
    source "$ERD_WS/.venv-erd/bin/activate"
    cd "$ERD_WS" || exit 1
    python -m colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo "-DERD_FRI_SDK_ROOT=$ERD_FRI_SDK_ROOT"
  )
}

_erd_test() {
  local rc=0
  (
    cd "$ERD_WS" || exit 1
    python -m colcon test --packages-select erd_iiwa erd_ur10 erd_fri_emulator erd_msgs
    python -m colcon test-result --verbose
  ) || rc=1
  # erd_recording imports pinocchio directly, which conflicts with a sourced
  # ROS environment (README "Building and running"): run it ROS-free.
  env -i HOME="$HOME" USER="$USER" LANG="${LANG:-C.UTF-8}" PATH=/usr/local/bin:/usr/bin:/bin \
    ERD_REPO_ROOT="$ERD_REPO_ROOT" ERD_WS="$ERD_WS" ERD_DATA_ROOT="$ERD_DATA_ROOT" \
    ERD_CONSUMER_REPO="$ERD_CONSUMER_REPO" ERD_CONSUMER_PYTHON="$ERD_CONSUMER_PYTHON" \
    bash -c 'source "$ERD_WS/.venv-erd/bin/activate" && cd "$ERD_WS" \
      && python -m colcon test --packages-select erd_recording \
      && python -m colcon test-result --verbose' || rc=1
  return "$rc"
}

_erd_main() {
  local deps=0 build=0 clean=0 run_tests=0 domain=87 arg
  for arg in "$@"; do
    case "$arg" in
      --deps) deps=1 ;;
      --build) build=1 ;;
      --clean) clean=1; build=1 ;;
      --test) run_tests=1 ;;
      --real) domain=0 ;;
      -h|--help) awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/, ""); print}' "${BASH_SOURCE[0]}"; return 0 ;;
      *) _erd_err "unknown option $arg (see --help)"; return 2 ;;
    esac
  done

  local os_id os_version
  os_id="$(. /etc/os-release && echo "$ID")"
  os_version="$(. /etc/os-release && echo "$VERSION_ID")"
  if [[ "$os_id" != "ubuntu" || "$os_version" != "24.04" ]]; then
    _erd_err "ROS 2 Jazzy needs Ubuntu 24.04; this machine is $os_id $os_version"
    return 1
  fi

  export ERD_REPO_ROOT
  ERD_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  export ERD_WS="$ERD_REPO_ROOT/ros2_ws"
  export ERD_DATA_ROOT="$ERD_REPO_ROOT/data/real_robot"
  export ERD_LAB="$ERD_REPO_ROOT/src/ros2/erd_recording/config/lab"
  export ERD_FRI_SDK_ROOT="$ERD_WS/src/external/iiwa_ros2/iiwa_hardware/external/libFRI"
  export ERD_CONSUMER_REPO="${ERD_CONSUMER_REPO:-$(dirname "$ERD_REPO_ROOT")/dynamic_model_nn}"
  export ERD_CONSUMER_PYTHON="${ERD_CONSUMER_PYTHON:-$ERD_CONSUMER_REPO/.venv/bin/python}"

  # Another venv (e.g. the simulation's uv .venv, Python 3.11) would shadow
  # .venv-erd; an old out-of-repo overlay would shadow this workspace.
  if [[ -n "${VIRTUAL_ENV:-}" && "$VIRTUAL_ENV" != "$ERD_WS/.venv-erd" ]] && declare -F deactivate >/dev/null; then
    _erd_log "deactivating $VIRTUAL_ENV"
    deactivate
  fi
  if [[ ":${AMENT_PREFIX_PATH:-}:" == *"/erd_ws/install"* ]]; then
    _erd_warn "this shell already sourced an old out-of-repo erd_ws overlay; open a fresh terminal"
  fi

  if (( deps )); then
    _erd_install_system_deps || { _erd_err "system dependency install failed"; return 1; }
  fi
  if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
    _erd_err "ROS 2 Jazzy not found in /opt/ros/jazzy; re-run with --deps"
    return 1
  fi

  mkdir -p "$ERD_WS/src" "$ERD_DATA_ROOT"
  ln -sfn ../../src/ros2 "$ERD_WS/src/erd"
  _erd_fetch_external || { _erd_err "fetching src/ros2/erd.repos failed"; return 1; }
  if (( deps )); then
    _erd_log "rosdep install (stack + external sources)"
    # shellcheck disable=SC1091
    ( source /opt/ros/jazzy/setup.bash && rosdep install -y -r --rosdistro jazzy --ignore-src \
        --from-paths "$ERD_REPO_ROOT/src/ros2" "$ERD_WS/src/external" ) \
      || _erd_warn "rosdep reported unresolved keys (see above); the explicit apt list usually covers them"
  fi
  _erd_setup_venv || { _erd_err "creating .venv-erd failed"; return 1; }

  if (( clean )); then
    _erd_log "removing ros2_ws/{build,install,log}"
    rm -rf "$ERD_WS/build" "$ERD_WS/install" "$ERD_WS/log"
  fi
  if (( build )) || [[ ! -f "$ERD_WS/install/setup.bash" ]]; then
    _erd_log "colcon build in $ERD_WS"
    _erd_build || { _erd_err "colcon build failed"; return 1; }
  fi

  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  # shellcheck disable=SC1091
  source "$ERD_WS/install/setup.bash"
  # The venv goes last so its python/pip come first on PATH (env_guard
  # reorders sys.path against ROS's PYTHONPATH, README "Building and running").
  # shellcheck disable=SC1091
  source "$ERD_WS/.venv-erd/bin/activate"
  export ROS_DOMAIN_ID="$domain"

  if (( run_tests )); then
    _erd_test || _erd_warn "some tests failed (see colcon test-result above)"
  fi

  [[ -x "$ERD_CONSUMER_PYTHON" ]] \
    || _erd_warn "consumer env $ERD_CONSUMER_PYTHON missing: datasets are written, the CustomDataset load check is skipped"
  command -v docker >/dev/null 2>&1 \
    || _erd_warn "docker not found: L2 (FRI emulator, URSim) needs it"

  _erd_log "ready: ROS_DOMAIN_ID=$ROS_DOMAIN_ID, workspace $ERD_WS"
  _erd_log "lab configs: \$ERD_LAB  recordings: \$ERD_DATA_ROOT"
}

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "workspace_setup.sh: run it from bash (source workspace_setup.sh)" >&2
elif [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "workspace_setup.sh must be sourced, not executed: source ${0} $*" >&2
  exit 1
else
  _erd_main "$@"
  _erd_rc=$?
  unset -f _erd_log _erd_warn _erd_err _erd_install_system_deps _erd_fetch_external \
    _erd_setup_venv _erd_build _erd_test _erd_main
  eval "unset _erd_rc; return $_erd_rc"
fi

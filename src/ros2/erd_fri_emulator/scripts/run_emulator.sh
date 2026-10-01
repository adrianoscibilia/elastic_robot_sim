#!/usr/bin/env bash
# RR_04 B-6: starts erd_fri_emulator in its own Docker network namespace and
# prints the container IP for `iiwa.launch.py robot_ip:=`.
#
# Why a container at all: the real FRI client (erd_iiwa/FriPositionSystem,
# unmodified KUKA SDK code underneath) binds a UDP wildcard address
# (0.0.0.0:<port>) for its own socket. Two processes on the same host cannot
# both bind a wildcard address to the same port, so the emulator (which would
# otherwise also bind 0.0.0.0:<port>) cannot coexist with the client in the
# same network namespace. A container gets the emulator its own namespace and
# its own bridge IP, so both sides keep the exact same "bind my own port"
# code unmodified (RR_03/RR_01 S3.5, found live in T1.9).
#
# This is exactly what pass 1 did by hand; see src/ros2/README.md's "Running
# L2" section for the full explanation, including why the container's port
# must NOT be published with `-p` (docker-proxy would then bind the host's
# port itself, reintroducing the identical collision).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ERD_WS="${ERD_WS:-$HOME/projects/erd_ws}"
CONTAINER_NAME="${CONTAINER_NAME:-erd_fri_emulator}"
FRI_PORT="${FRI_PORT:-30200}"
URDF_PATH="${URDF_PATH:-/assets/robots/kuka_lbr_iiwa_14_r820/description/kuka_lbr_iiwa_14_r820.urdf}"
EXTRA_ARGS=("$@")

if ! command -v docker >/dev/null 2>&1; then
  echo "run_emulator.sh: docker is required" >&2
  exit 1
fi

HOST_IP="$(ip -4 addr show docker0 | grep -oP 'inet \K[\d.]+' || true)"
if [[ -z "$HOST_IP" ]]; then
  echo "run_emulator.sh: could not determine the docker0 bridge IP -- is Docker running?" >&2
  exit 1
fi

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "run_emulator.sh: removing a previous $CONTAINER_NAME container" >&2
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker run -d --name "$CONTAINER_NAME" \
  -v /opt/ros/jazzy:/opt/ros/jazzy:ro \
  -v "$ERD_WS/install":/erd_ws_install:ro \
  -v "$REPO_ROOT/assets":/assets:ro \
  -v /usr/lib/x86_64-linux-gnu:/host_lib:ro \
  ubuntu:24.04 bash -c "
    export LD_LIBRARY_PATH=/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/lib:/host_lib
    exec /erd_ws_install/erd_fri_emulator/lib/erd_fri_emulator/erd_fri_emulator \
      --urdf '$URDF_PATH' --port '$FRI_PORT' --client-address '$HOST_IP' ${EXTRA_ARGS[*]}
  " >/dev/null
# no -p/port publishing -- see the header comment

CONTAINER_IP="$(docker inspect "$CONTAINER_NAME" --format '{{.NetworkSettings.IPAddress}}')"
echo "erd_fri_emulator running as '$CONTAINER_NAME', container IP: $CONTAINER_IP"
echo "Use: ros2 launch erd_iiwa iiwa.launch.py hardware:=emulator robot_ip:=$CONTAINER_IP fri_port:=$FRI_PORT"
echo "Stop with: docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"

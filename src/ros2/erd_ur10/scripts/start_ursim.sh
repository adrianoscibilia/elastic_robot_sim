#!/usr/bin/env bash
# RR_04 B-6: wraps UR's own `start_ursim.sh -m ur10 -v 3.15.8 -d` (now
# shipped in `ur_client_library`, RR_01 v3.1 S3.5 -- 3.15.7 isn't published).
# That script downloads and mounts the External Control URCap itself
# (`/urcaps`, programs in `/ursim/programs`, S-18), so nothing here needs to.
# Prints the container IP for `ur10.launch.py robot_ip:=`.
#
# At L2, pass `headless_mode:=true` to `ur10.launch.py` so the driver sends
# its own program and trajectory goals execute without a pendant program
# (RR_01 v3.1 S3.5) -- this script only starts URSim, it doesn't launch the
# driver.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-ursim_ur10}"
UR_TYPE="${UR_TYPE:-ur10}"
URSIM_VERSION="${URSIM_VERSION:-3.15.8}"

if ! command -v docker >/dev/null 2>&1; then
  echo "start_ursim.sh: docker is required" >&2
  exit 1
fi

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "start_ursim.sh: removing a previous $CONTAINER_NAME container" >&2
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

ros2 run ur_client_library start_ursim.sh -m "$UR_TYPE" -v "$URSIM_VERSION" -d -n "$CONTAINER_NAME"

# RR_06 P3-1 item 11, live finding: ur_client_library's own start_ursim.sh
# puts the container on its own bridge network (e.g. `ursim_net`), not the
# default `bridge` one -- `.NetworkSettings.IPAddress` (only ever populated
# for the default bridge) and `docker0`'s address (a different bridge
# entirely) were both empty/wrong here. Read the actual network the
# container landed on, and that network's own gateway as the host-side IP.
NETWORK_NAME="$(docker inspect "$CONTAINER_NAME" --format '{{range $net, $v := .NetworkSettings.Networks}}{{$net}}{{end}}')"
CONTAINER_IP="$(docker inspect "$CONTAINER_NAME" --format "{{(index .NetworkSettings.Networks \"$NETWORK_NAME\").IPAddress}}")"
HOST_IP="$(docker network inspect "$NETWORK_NAME" --format '{{(index .IPAM.Config 0).Gateway}}')"
echo "URSim running as '$CONTAINER_NAME' on network '$NETWORK_NAME', container IP: $CONTAINER_IP"
echo "Dashboard/noVNC: http://$CONTAINER_IP:6080/vnc.html (power on + release brakes once before use)"
echo "Use: ros2 launch erd_ur10 ur10.launch.py hardware:=ursim headless_mode:=true \\"
echo "         robot_ip:=$CONTAINER_IP reverse_ip:=$HOST_IP"
echo "Stop with: docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"

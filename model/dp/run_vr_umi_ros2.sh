#!/usr/bin/env bash
# 加载 ROS2 与本地消息工作区，在现有 DP 虚拟环境中启动 Link7 策略推理。
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source /opt/ros/humble/setup.bash
source "$REPO_ROOT/ros2_ws/install/local_setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/infer_vr_umi_ros2.py" "$@"

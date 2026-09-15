#!/usr/bin/env bash
# 加载 RM75 反馈、FastUMI 消息与 DP 环境后启动纯推理入口。
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source /opt/ros/humble/setup.bash
source "$REPO_ROOT/ros2_ws/install/setup.bash"
# RM75、Unitree 与高带宽压缩图像节点统一使用该 Fast DDS transport；
# 避免调用终端遗留的 UDPv4 配置造成同 domain 内端点无法互相发现。
export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA
exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/infer_real.py" "$@"

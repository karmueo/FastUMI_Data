#!/usr/bin/env bash
# 加载工作区环境，一键启动厂商服务端与 ROS 2 控制节点。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
NETWORK_INTERFACE="wlP1p1s0"

usage() {
    echo "用法: $0 [-n 网卡]"
}

fail() {
    echo "error: $*" >&2
    exit 1
}

while getopts ":n:h" option; do
    case "${option}" in
        n) NETWORK_INTERFACE="${OPTARG}" ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
[[ $# -eq 0 ]] || { usage >&2; exit 2; }

[[ "$(uname -m)" == "aarch64" ]] || fail "厂商服务端只支持 Linux aarch64"
[[ -f /opt/ros/humble/setup.bash ]] || fail "未找到 ROS 2 Humble"
[[ -x "${WORKSPACE}/.venv-numpy1/bin/python" ]] \
    || fail "缺少 NumPy 1 环境；请运行 ${SCRIPT_DIR}/setup_gripper_env.sh"
export CYCLONEDDS_HOME="${WORKSPACE}/.deps/unitree_gripper/install/cyclonedds"
[[ -f "${CYCLONEDDS_HOME}/lib/libddsc.so.0.10.2" ]] \
    || fail "缺少匹配的 CycloneDDS；请运行 ${SCRIPT_DIR}/setup_gripper_env.sh"
[[ -f "${WORKSPACE}/install/setup.bash" ]] \
    || fail "工作区尚未构建；请在 NumPy 1 环境中构建 unitree_gripper"
ip link show "${NETWORK_INTERFACE}" >/dev/null 2>&1 \
    || fail "网卡不存在: ${NETWORK_INTERFACE}"

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "${WORKSPACE}/.venv-numpy1/bin/activate"
# shellcheck disable=SC1091
source "${WORKSPACE}/install/setup.bash"
set -u

PACKAGE_PREFIX="$(ros2 pkg prefix unitree_gripper 2>/dev/null)" \
    || fail "未找到已构建的 unitree_gripper；请在 NumPy 1 环境中构建该包"
PACKAGE_SHARE="${PACKAGE_PREFIX}/share/unitree_gripper"
[[ -x "${PACKAGE_SHARE}/vendor/dex1_1_gripper_server" ]] \
    || fail "缺少已安装的服务端；请重新构建 unitree_gripper"
[[ -f "${PACKAGE_SHARE}/vendor/lib/libUnitreeMotorSDK_Arm64.so" ]] \
    || fail "缺少已安装的 Unitree 动态库；请重新构建 unitree_gripper"
[[ -f "${PACKAGE_SHARE}/launch/gripper.launch.py" ]] \
    || fail "缺少 launch 文件；请重新构建 unitree_gripper"
exec ros2 launch unitree_gripper gripper.launch.py \
    "network_interface:=${NETWORK_INTERFACE}"

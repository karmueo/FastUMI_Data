#!/usr/bin/env bash
# 在 Jazzy 中按主机架构启动服务端与 ROS 2 控制节点。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config/gripper.yaml"
NETWORK_INTERFACE=""

usage() {
    echo "用法: $0 [-c 配置文件] [-n 网卡]"
}

fail() {
    echo "error: $*" >&2
    exit 1
}

while getopts ":c:n:h" option; do
    case "${option}" in
        c) CONFIG_FILE="${OPTARG}" ;;
        n)
            [[ -n "${OPTARG}" ]] || fail "-n 指定的网卡不能为空"
            NETWORK_INTERFACE="${OPTARG}"
            ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
[[ $# -eq 0 ]] || { usage >&2; exit 2; }
[[ -f "${CONFIG_FILE}" ]] || fail "夹爪配置文件不存在: ${CONFIG_FILE}"
CONFIG_FILE="$(realpath -- "${CONFIG_FILE}")"

# 厂商服务端只扫描这些本地串口；缺少设备时不启动两个 ROS 进程。
SERIAL_PORT_FOUND=false
INACCESSIBLE_PORTS=()
for serial_port in /dev/ttyUSB* /dev/ttyCH343USB* /dev/ttyACM*; do
    if [[ -c "${serial_port}" ]]; then
        SERIAL_PORT_FOUND=true
        if [[ ! -r "${serial_port}" || ! -w "${serial_port}" ]]; then
            INACCESSIBLE_PORTS+=("${serial_port}")
        fi
    fi
done
[[ "${SERIAL_PORT_FOUND}" == true ]] \
    || fail "未发现夹爪串口（/dev/ttyUSB*、/dev/ttyCH343USB*、/dev/ttyACM*）；请检查夹爪供电和 USB 串口连接"
[[ ${#INACCESSIBLE_PORTS[@]} -eq 0 ]] \
    || fail "当前用户 ${USER:-$(id -un)} 无权读写串口: ${INACCESSIBLE_PORTS[*]}；请将用户加入设备所属组（通常为 dialout），重新登录后再启动"

ARCH="$(uname -m)"
[[ "${ARCH}" == "aarch64" || "${ARCH}" == "x86_64" ]] \
    || fail "不支持的架构: ${ARCH}"
[[ -f /opt/ros/jazzy/setup.bash ]] || fail "未找到 ROS 2 Jazzy"
[[ -x "${WORKSPACE}/.venv-numpy1/bin/python" ]] \
    || fail "缺少 NumPy 1 环境；请运行 ${SCRIPT_DIR}/setup_gripper_env.sh"
export CYCLONEDDS_HOME="${SCRIPT_DIR}/deps/install/cyclonedds"
[[ "${ARCH}" == "aarch64" ]] \
    && export CYCLONEDDS_HOME="${SCRIPT_DIR}/deps/local/aarch64/cyclonedds"
[[ -f "${CYCLONEDDS_HOME}/lib/libddsc.so.0.10.2" ]] \
    || fail "缺少匹配的 CycloneDDS；请运行 ${SCRIPT_DIR}/setup_gripper_env.sh"
[[ -f "${WORKSPACE}/install/setup.bash" ]] \
    || fail "工作区尚未构建；请在 NumPy 1 环境中构建 unitree_gripper"

set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "${WORKSPACE}/.venv-numpy1/bin/activate"
# shellcheck disable=SC1091
source "${WORKSPACE}/install/setup.bash"
set -u

PACKAGE_PREFIX="$(ros2 pkg prefix unitree_gripper 2>/dev/null)" \
    || fail "未找到已构建的 unitree_gripper；请在 NumPy 1 环境中构建该包"
PACKAGE_SHARE="${PACKAGE_PREFIX}/share/unitree_gripper"
if [[ "${ARCH}" == "x86_64" ]]; then
    export UNITREE_GRIPPER_VENDOR_DIR="${SCRIPT_DIR}/deps/vendor/x86_64"
else
    export UNITREE_GRIPPER_VENDOR_DIR="${PACKAGE_SHARE}/vendor"
fi
[[ -x "${UNITREE_GRIPPER_VENDOR_DIR}/dex1_1_gripper_server" ]] \
    || fail "缺少夹爪服务端: ${UNITREE_GRIPPER_VENDOR_DIR}；请运行 ${SCRIPT_DIR}/setup_gripper_env.sh"
MOTOR_LIBRARY="libUnitreeMotorSDK_Arm64.so"
[[ "${ARCH}" == "x86_64" ]] && MOTOR_LIBRARY="libUnitreeMotorSDK_Linux64.so"
[[ -f "${UNITREE_GRIPPER_VENDOR_DIR}/lib/${MOTOR_LIBRARY}" ]] \
    || fail "缺少夹爪电机库: ${UNITREE_GRIPPER_VENDOR_DIR}/lib/${MOTOR_LIBRARY}；请运行 ${SCRIPT_DIR}/setup_gripper_env.sh"
[[ -f "${PACKAGE_SHARE}/launch/gripper.launch.py" ]] \
    || fail "缺少 launch 文件；请重新构建 unitree_gripper"
LAUNCH_ARGUMENTS=("config_file:=${CONFIG_FILE}")
if [[ -n "${NETWORK_INTERFACE}" ]]; then
    LAUNCH_ARGUMENTS+=("network_interface:=${NETWORK_INTERFACE}")
fi
exec ros2 launch unitree_gripper gripper.launch.py "${LAUNCH_ARGUMENTS[@]}"

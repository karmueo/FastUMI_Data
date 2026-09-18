#!/usr/bin/env bash
# 为 Jazzy 准备 CycloneDDS，并按主机架构构建 Dex1-1 服务端。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV="${WORKSPACE}/.venv-numpy1"
DEPS_DIR="${SCRIPT_DIR}/deps"
CYCLONE_SOURCE="${DEPS_DIR}/src/cyclonedds"
CYCLONE_BUILD="${DEPS_DIR}/build/cyclonedds"
CYCLONEDDS_HOME="${DEPS_DIR}/install/cyclonedds"
SERVER_SOURCE="${DEPS_DIR}/src/dex1_1_service"
SDK_SOURCE="${DEPS_DIR}/src/unitree_sdk2"
SERVER_BUILD="${DEPS_DIR}/build/dex1_1_service"
VENDOR_DIR="${DEPS_DIR}/vendor/x86_64"
SERVER_REV="ced3216290bb22dae356213993786beeaa7143ab"
SDK_REV="c753829882fba461ed07ba25aaabee0a25d83663"
CYCLONE_REV="0.10.2"
CYCLONE_COMMIT="9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd"
REBUILD_DEPS=false

fail() {
    echo "error: $*" >&2
    exit 1
}

case "${1:-}" in
    "") [[ $# -eq 0 ]] || fail "用法: $0 [--rebuild]" ;;
    --rebuild)
        [[ $# -eq 1 ]] || fail "用法: $0 [--rebuild]"
        REBUILD_DEPS=true
        ;;
    *) fail "用法: $0 [--rebuild]" ;;
esac

ARCH="$(uname -m)"
[[ "${ARCH}" == "aarch64" || "${ARCH}" == "x86_64" ]] \
    || fail "不支持的架构: ${ARCH}"
[[ "${ARCH}" == "aarch64" ]] \
    && CYCLONEDDS_HOME="${DEPS_DIR}/local/aarch64/cyclonedds"
[[ -f /opt/ros/jazzy/setup.bash ]] || fail "未找到 /opt/ros/jazzy/setup.bash"
for tool in uv ldd file; do
    command -v "${tool}" >/dev/null 2>&1 || fail "缺少构建工具: ${tool}"
done

if [[ ! -x "${VENV}/bin/python" ]]; then
    uv venv --python /usr/bin/python3 --system-site-packages "${VENV}"
    uv pip install --python "${VENV}/bin/python" -r "${WORKSPACE}/requirements-numpy1.txt"
fi

# ROS 2 的设置脚本不兼容 nounset。
set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
set -u

if [[ "${REBUILD_DEPS}" == true || \
      ! -f "${CYCLONEDDS_HOME}/lib/libddsc.so.0.10.2" ]]; then
    for tool in git cmake cc c++; do
        command -v "${tool}" >/dev/null 2>&1 || fail "缺少构建工具: ${tool}"
    done
    mkdir -p "${DEPS_DIR}/src" "${DEPS_DIR}/build" "${DEPS_DIR}/install"
    if [[ ! -d "${CYCLONE_SOURCE}/.git" ]]; then
        git clone --branch "${CYCLONE_REV}" --depth 1 \
            https://github.com/eclipse-cyclonedds/cyclonedds.git "${CYCLONE_SOURCE}"
    fi
    [[ "$(git -C "${CYCLONE_SOURCE}" rev-parse HEAD)" == "${CYCLONE_COMMIT}" ]] \
        || fail "CycloneDDS 源码版本不匹配: ${CYCLONE_SOURCE}"
    rm -rf -- "${CYCLONE_BUILD}"
    cmake -S "${CYCLONE_SOURCE}" -B "${CYCLONE_BUILD}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${CYCLONEDDS_HOME}" \
        -DBUILD_DDSPERF=OFF -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF -DENABLE_SHM=OFF
    cmake --build "${CYCLONE_BUILD}" --parallel "$(nproc)"
    cmake --install "${CYCLONE_BUILD}"
    # 安装树纳入 Git，pkg-config 不应记录当前机器的绝对路径。
    sed -i '1c\prefix=${pcfiledir}/../..' \
        "${CYCLONEDDS_HOME}/lib/pkgconfig/CycloneDDS.pc"
fi

export CYCLONEDDS_HOME
uv pip install --python "${VENV}/bin/python" \
    --reinstall --no-cache --no-binary cyclonedds --no-deps \
    cyclonedds==0.10.2

env LD_LIBRARY_PATH="${CYCLONEDDS_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}" \
    "${VENV}/bin/python" - <<'PY'
from pathlib import Path
import os

import cyclonedds
import rclpy
from cyclonedds.domain import Domain
from unitree_gripper._vendor.unitree_sdk2py.core.channel_config import ChannelConfigHasInterface

runtime = str(Path(os.environ["CYCLONEDDS_HOME"]).resolve())
loaded = Path("/proc/self/maps").read_text()
if runtime not in loaded:
    raise RuntimeError("Python 未加载工作区私有 CycloneDDS 运行库")
Domain(0, ChannelConfigHasInterface.replace("$__IF_NAME__$", "lo"))
print("夹爪环境已就绪:", cyclonedds.__file__)
PY

checkout_pinned() {
    local url="$1" directory="$2" revision="$3"
    if [[ ! -d "${directory}/.git" ]]; then
        git clone --filter=blob:none "${url}" "${directory}"
    fi
    git -C "${directory}" fetch --depth 1 origin "${revision}"
    git -C "${directory}" checkout --detach --force FETCH_HEAD
    [[ "$(git -C "${directory}" rev-parse HEAD)" == "${revision}" ]] \
        || fail "源码版本校验失败: ${directory}"
}

if [[ "${ARCH}" == "x86_64" && \
      ( "${REBUILD_DEPS}" == true || \
        ! -x "${VENDOR_DIR}/dex1_1_gripper_server" || \
        ! -f "${VENDOR_DIR}/lib/libUnitreeMotorSDK_Linux64.so" || \
        ! -f "${VENDOR_DIR}/lib/libddsc.so.0.10.2" || \
        ! -f "${VENDOR_DIR}/lib/libddscxx.so.0" ) ]]; then
    for tool in git cmake gcc c++; do
        command -v "${tool}" >/dev/null 2>&1 || fail "缺少构建工具: ${tool}"
    done
    BOOST_DIR=""
    SYSTEM_CMAKE_DIR="/usr/lib/$(gcc -dumpmachine)/cmake"
    for candidate in "${SYSTEM_CMAKE_DIR}"/Boost-*; do
        [[ -d "${candidate}" ]] && BOOST_DIR="${candidate}" && break
    done
    [[ -n "${BOOST_DIR}" ]] || fail "缺少系统 Boost CMake 配置"
    [[ -d "${SYSTEM_CMAKE_DIR}/fmt" && -d "${SYSTEM_CMAKE_DIR}/yaml-cpp" ]] \
        || fail "缺少系统 fmt 或 yaml-cpp CMake 配置"
    mkdir -p "${DEPS_DIR}/include"
    ln -sfn /usr/include/boost "${DEPS_DIR}/include/boost"
    checkout_pinned https://github.com/unitreerobotics/dex1_1_service.git \
        "${SERVER_SOURCE}" "${SERVER_REV}"
    checkout_pinned https://github.com/unitreerobotics/unitree_sdk2.git \
        "${SDK_SOURCE}" "${SDK_REV}"
    git -C "${SERVER_SOURCE}" apply "${SCRIPT_DIR}/patches/dex1_1_service.cmake.patch"
    rm -rf -- "${SERVER_BUILD}"
    cmake -S "${SERVER_SOURCE}" -B "${SERVER_BUILD}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBoost_NO_BOOST_CMAKE=OFF -DBoost_DIR="${BOOST_DIR}" \
        -DBOOST_HEADER_OVERLAY="${DEPS_DIR}/include" \
        -Dfmt_DIR="${SYSTEM_CMAKE_DIR}/fmt" \
        -Dyaml-cpp_DIR="${SYSTEM_CMAKE_DIR}/yaml-cpp" \
        -DUNITREE_SDK2_ROOT="${SDK_SOURCE}" \
        -DCYCLONEDDS_ROOT="${CYCLONEDDS_HOME}"
    cmake --build "${SERVER_BUILD}" --target dex1_1_gripper_server \
        --parallel "$(nproc)"
    mkdir -p "${VENDOR_DIR}/lib"
    cp "${SERVER_BUILD}/dex1_1_gripper_server" "${VENDOR_DIR}/"
    cp "${SERVER_SOURCE}/lib/libUnitreeMotorSDK_Linux64.so" "${VENDOR_DIR}/lib/"
    cp -a "${CYCLONEDDS_HOME}/lib/libddsc.so"* "${VENDOR_DIR}/lib/"
    cp -a "${SDK_SOURCE}/thirdparty/lib/x86_64/libddscxx.so"* "${VENDOR_DIR}/lib/"
    file "${VENDOR_DIR}/dex1_1_gripper_server" | grep -q 'x86-64' \
        || fail "服务端不是 x86_64 ELF"
    env LD_LIBRARY_PATH="${VENDOR_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        ldd -r "${VENDOR_DIR}/dex1_1_gripper_server" >"${DEPS_DIR}/server-ldd.txt"
    ! grep -Eq 'not found|undefined symbol' "${DEPS_DIR}/server-ldd.txt" \
        || fail "服务端动态库解析失败，见 ${DEPS_DIR}/server-ldd.txt"
fi

echo "构建: cd ${WORKSPACE} && source .venv-numpy1/bin/activate && python -m colcon build --symlink-install --packages-select unitree_gripper"

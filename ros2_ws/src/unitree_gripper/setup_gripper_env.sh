#!/usr/bin/env bash
# 准备工作区 NumPy 1 环境与匹配的 CycloneDDS 0.10.2 运行库。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV="${WORKSPACE}/.venv-numpy1"
DEPS_DIR="${WORKSPACE}/.deps/unitree_gripper"
CYCLONE_SOURCE="${DEPS_DIR}/src/cyclonedds"
CYCLONE_BUILD="${DEPS_DIR}/build/cyclonedds"
CYCLONEDDS_HOME="${DEPS_DIR}/install/cyclonedds"

fail() {
    echo "error: $*" >&2
    exit 1
}

[[ "$(uname -m)" == "aarch64" ]] || fail "只支持 Linux aarch64 厂商服务端"
[[ -f /opt/ros/humble/setup.bash ]] || fail "未找到 /opt/ros/humble/setup.bash"
for tool in uv git cmake cc; do
    command -v "${tool}" >/dev/null 2>&1 || fail "缺少构建工具: ${tool}"
done

if [[ ! -x "${VENV}/bin/python" ]]; then
    uv venv --python /usr/bin/python3 --system-site-packages "${VENV}"
    uv pip install --python "${VENV}/bin/python" -r "${WORKSPACE}/requirements-numpy1.txt"
fi

# ROS 2 Humble 的设置脚本不兼容 nounset。
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u

if [[ ! -f "${CYCLONEDDS_HOME}/lib/libddsc.so.0.10.2" ]]; then
    mkdir -p "${DEPS_DIR}/src" "${DEPS_DIR}/build" "${DEPS_DIR}/install"
    if [[ ! -d "${CYCLONE_SOURCE}/.git" ]]; then
        git clone --branch 0.10.2 --depth 1 \
            https://github.com/eclipse-cyclonedds/cyclonedds.git "${CYCLONE_SOURCE}"
    fi
    cmake -S "${CYCLONE_SOURCE}" -B "${CYCLONE_BUILD}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${CYCLONEDDS_HOME}" \
        -DBUILD_DDSPERF=OFF -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF -DENABLE_SHM=OFF
    cmake --build "${CYCLONE_BUILD}" --parallel "$(nproc)"
    cmake --install "${CYCLONE_BUILD}"
    rm -rf -- "${CYCLONE_SOURCE}" "${CYCLONE_BUILD}"
fi

export CYCLONEDDS_HOME
uv pip install --python "${VENV}/bin/python" \
    --reinstall --no-cache --no-binary cyclonedds --no-deps \
    cyclonedds==0.10.2

"${VENV}/bin/python" - <<'PY'
from pathlib import Path
import os

import cyclonedds
import rclpy

runtime = str(Path(os.environ["CYCLONEDDS_HOME"]).resolve())
loaded = Path("/proc/self/maps").read_text()
if runtime not in loaded:
    raise RuntimeError("Python 未加载工作区私有 CycloneDDS 运行库")
print("夹爪环境已就绪:", cyclonedds.__file__)
PY

echo "构建: cd ${WORKSPACE} && source .venv-numpy1/bin/activate && python -m colcon build --symlink-install --packages-select unitree_gripper"

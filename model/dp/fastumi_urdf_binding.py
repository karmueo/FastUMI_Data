"""校验 RM75 训练与部署 URDF 的完整文件摘要及已核对的单向迁移。

本模块仅依赖标准库，同时供独立训练代码和 ROS 2 启动前校验使用。
"""

import hashlib
from pathlib import Path


# 正式夹爪模型的完整文件摘要；七轴及 base_link→Link7 运动链保持不变。
RM75_URDF_SHA256 = "2724631527c06ac53efc8dd247671995e32b40f07818ba49a6f2c0408be63607"
# 仅登记已核对的单向迁移；历史数据和 checkpoint 中的训练摘要保持原值。
RM75_URDF_MIGRATIONS = {
    "c37545834f0d6a4719548e51337a258be0c3b95bb6b79467d835875a205bb056": RM75_URDF_SHA256,
}


def validate_urdf_binding(urdf_path, training_sha256, deployment_sha256=None):
    """核对完整文件摘要，仅允许原文件或已登记的固定夹爪迁移。

    training_sha256 是训练原始摘要；deployment_sha256 可额外限定部署版本。
    返回实际摘要；缺失文件、非法摘要或未登记的模型变更均抛出 ValueError。
    本迁移仅保证七轴 base_link→Link7 的米制运动学等价，不替代 TCP 标定。
    """
    path = Path(urdf_path)
    if not path.is_file():
        raise ValueError(f"URDF must name an existing file: {path}")
    digests = (training_sha256,) if deployment_sha256 is None else (training_sha256, deployment_sha256)
    for digest in digests:
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest.lower())):
            raise ValueError("A valid URDF SHA-256 is required")
    expected = training_sha256.lower()
    allowed = {expected, RM75_URDF_MIGRATIONS.get(expected)}
    if deployment_sha256 is not None:
        target = deployment_sha256.lower()
        if target not in allowed:
            raise ValueError("URDF SHA-256 migration is not registered")
        allowed = {target}
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual not in allowed:
        raise ValueError(
            f"Deployment URDF SHA-256 does not match checkpoint/training or registered migration: "
            f"expected {expected}, got {actual}"
        )
    return actual

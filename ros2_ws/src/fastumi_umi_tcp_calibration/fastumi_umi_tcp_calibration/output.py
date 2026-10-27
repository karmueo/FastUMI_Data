"""保存标定诊断及原子发布结果；默认拒绝覆盖已有标定文件。"""

import os
from pathlib import Path
import tempfile

import yaml


# 列向量约定与用户提供的 T_u_tag_tcp.yaml 保持一致。
RESULT_HEADER = (
    "# UMI 相机光学坐标系中的 TCP 位姿；平移单位米，采用列向量。\n"
    "# T_u_cam_tcp = ^u_cam T_tcp = T_u_cam_tag @ T_u_tag_tcp\n"
    "# p_u_cam = T_u_cam_tcp @ p_tcp；光学坐标 X 向右、Y 向下、Z 向前。\n"
)


def write_yaml(path, data):
    """写入普通中文诊断 YAML；调用方负责创建父目录。"""
    Path(path).write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def publish_result(path, matrix, overwrite=False):
    """同目录暂存并原子发布矩阵；无覆盖模式使用硬链接避免检查与写入竞争。"""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=".umi_tcp_", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(RESULT_HEADER)
            yaml.safe_dump({"T_u_cam_tcp": matrix.tolist()}, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

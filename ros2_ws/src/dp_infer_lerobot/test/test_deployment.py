"""验证部署档契约、checkpoint 指纹、URDF 与 TCP 外参检查。"""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from dp_infer_lerobot import deployment

PROFILE = Path(__file__).resolve().parents[1] / "config/dp_umi_v4_deployment.yaml"  # 随包部署档。
REPO = Path(__file__).resolve().parents[4]  # 仓库根目录。


def _write(tmp_path, profile):
    """写出部署档 YAML 并返回路径。"""
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile))
    return path


def test_packaged_profile_loads_and_contract_is_strict(tmp_path):
    """随包部署档可加载；契约任一字段变化即拒绝。"""
    profile = deployment.load_profile(PROFILE)
    assert profile["contract"]["end_frame"] == "tcp"
    changed = copy.deepcopy(profile)
    changed["contract"]["end_frame"] = "Link7"
    with pytest.raises(ValueError, match="contract"):
        deployment.load_profile(_write(tmp_path, changed))


def test_checkpoint_and_urdf_fingerprints(tmp_path):
    """文件集合或内容与部署档不符时拒绝；URDF 必须是训练记录的同一份。"""
    profile = deployment.load_profile(PROFILE)
    with pytest.raises(ValueError, match="pretrained directory"):
        deployment.validate_checkpoint(tmp_path / "missing", profile)
    urdf = tmp_path / "arm.urdf"
    urdf.write_text("<robot/>")
    with pytest.raises(ValueError, match="URDF"):
        deployment.validate_urdf(urdf, profile)
    profile = dict(profile, urdf_sha256=deployment.sha256_file(urdf))
    deployment.validate_urdf(urdf, profile)


def test_tool_offset_requires_files_and_matching_axes(tmp_path):
    """缺少外参文件拒绝；与示教 TCP 轴向相差 180° 拒绝，一致时返回 T_Link7_TCP。"""
    camera_from_link7 = np.eye(4)
    camera_from_link7[:3, 3] = [0.0, 0.09, -0.016]
    camera_from_tcp = np.eye(4)
    camera_from_tcp[:3, :3] = Rotation.from_rotvec([1.6, 0, 0]).as_matrix()
    camera_from_tcp[:3, 3] = [0.0, 0.074, 0.21]
    end = tmp_path / "T_cam_end.yaml"
    end.write_text(yaml.safe_dump({"cam0": {"T_cam_hand": camera_from_link7.tolist()}}))
    tcp = tmp_path / "T_d_cam_tcp.yaml"
    tcp.write_text(yaml.safe_dump({"T_d_cam_tcp": camera_from_tcp.tolist()}))
    profile = {"umi_camera_from_tcp": camera_from_tcp.tolist()}
    with pytest.raises(ValueError, match="require"):
        deployment.load_tool_offset(profile, "", str(tcp), 10.0)
    link7_from_tcp, offset_mm, angle = deployment.load_tool_offset(profile, str(end), str(tcp), 10.0)
    np.testing.assert_allclose(link7_from_tcp, np.linalg.inv(camera_from_link7) @ camera_from_tcp, atol=1e-12)
    assert offset_mm == pytest.approx(0) and angle == pytest.approx(0)
    flipped = camera_from_tcp.copy()
    flipped[:3, :3] = flipped[:3, :3] @ Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
    with pytest.raises(ValueError, match="axis convention"):
        deployment.load_tool_offset({"umi_camera_from_tcp": flipped.tolist()}, str(end), str(tcp), 10.0)


def test_repository_urdf_and_calibration_match_profile():
    """仓库训练 URDF 指纹与部署档一致；当前部署外参与示教 TCP 轴向一致，Link7→TCP 约 226 mm。"""
    profile = deployment.load_profile(PROFILE)
    deployment.validate_urdf(REPO / "assets/rm_75.urdf", profile)
    link7_from_tcp, offset_mm, angle = deployment.load_tool_offset(
        profile, REPO / "config/calibration/T_cam_end.yaml", REPO / "config/calibration/T_d_cam_tcp.yaml")
    assert 0.22 < np.linalg.norm(link7_from_tcp[:3, 3]) < 0.23
    assert offset_mm < 20 and angle < deployment.DEFAULT_MAX_TCP_AXIS_DEG

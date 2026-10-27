"""验证从仓库、工作空间或其他目录启动时的默认路径及显式参数行为。"""

from pathlib import Path

import pytest

from fastumi_umi_tcp_calibration.cli import default_repository_root, parser


REPO = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("directory", [REPO, REPO / "ros2_ws"])
def test_defaults_use_repository(directory, monkeypatch):
    """从仓库或工作空间启动时，配置和输出均定位到同一仓库。"""
    monkeypatch.chdir(directory)
    args = parser().parse_args(["--bag", "recording.mcap"])
    assert default_repository_root() == REPO
    assert Path(args.camera_config) == REPO / "config/calibration/UmiCalib.yaml"
    assert Path(args.tag_tcp_config) == REPO / "config/calibration/T_u_tag_tcp.yaml"
    assert Path(args.output) == REPO / "config/calibration/T_u_cam_tcp.yaml"
    assert args.bag == "recording.mcap"


def test_explicit_paths_remain_relative(monkeypatch):
    """用户显式指定的相对路径不应被仓库定位规则改写。"""
    monkeypatch.chdir(REPO / "ros2_ws")
    args = parser().parse_args(
        [
            "--bag",
            "recording.mcap",
            "--camera-config",
            "custom/camera.yaml",
            "--tag-tcp-config",
            "custom/tag.yaml",
            "--output",
            "custom/result.yaml",
            "--diagnostics-dir",
            "custom/diagnostics",
        ]
    )
    assert args.camera_config == "custom/camera.yaml"
    assert args.tag_tcp_config == "custom/tag.yaml"
    assert args.output == "custom/result.yaml"
    assert args.diagnostics_dir == "custom/diagnostics"


def test_other_directory_uses_package_location(tmp_path, monkeypatch):
    """从仓库外启动时，通过包源码或安装位置定位仓库。"""
    monkeypatch.chdir(tmp_path)
    assert default_repository_root() == REPO
    assert Path(parser().parse_args(["--bag", "x.mcap"]).camera_config).is_file()

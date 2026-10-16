"""验证标定与预测入口统一使用安装包标定，不受工作目录和旧用户文件影响。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from gripper_openness import paths
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ESTIMATOR_LAUNCH = (
    PACKAGE_ROOT.parent
    / "fastumi_gripper_estimator"
    / "launch"
    / "gripper_openness.launch.py"
)


def _launch_defaults(path: Path, monkeypatch, package_share: Path) -> dict:
    """加载启动文件并返回已解析的默认参数。"""
    spec = spec_from_file_location(path.stem.replace(".", "_"), path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_package_share_directory", lambda _: str(package_share))
    monkeypatch.setattr(paths, "get_package_share_directory", lambda _: str(package_share))
    context = LaunchContext()
    return {
        action.name: perform_substitutions(context, action.default_value)
        for action in module.generate_launch_description().entities
        if isinstance(action, DeclareLaunchArgument)
    }


def test_calibration_defaults_to_installed_file(tmp_path: Path, monkeypatch) -> None:
    """标定输出始终指向安装包文件，保留显式覆盖保护。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    defaults = _launch_defaults(
        PACKAGE_ROOT / "launch" / "gripper_calibration.launch.py",
        monkeypatch,
        tmp_path / "share/gripper_openness",
    )
    assert defaults["output_path"] == str(
        tmp_path / "share/gripper_openness/config/gripper_calibration.yaml"
    )
    assert defaults["overwrite"] == "true"
    assert defaults["image_topic"] == "/umi_camera/image_raw"
    assert defaults["camera_calibration_path"] == ""
    assert defaults["mcap_path"] == ""
    assert defaults["mcap_image_topic"] == ""
    assert defaults["roi_ratios"] == "[0.0, 0.0, 1.0, 1.0]"


@pytest.mark.parametrize(
    "launch_file",
    [PACKAGE_ROOT / "launch" / "gripper_openness.launch.py", ESTIMATOR_LAUNCH],
)
def test_prediction_uses_installed_calibration(
    tmp_path: Path, monkeypatch, launch_file: Path
) -> None:
    """两个预测入口均使用安装包标定，忽略旧的用户主目录文件。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    before = _launch_defaults(launch_file, monkeypatch, tmp_path / "share/gripper_openness")
    assert before["gripper_calibration_path"] == str(
        tmp_path / "share/gripper_openness/config/gripper_calibration.yaml"
    )
    assert before["image_topic"] == "/umi_camera/image_raw"

    generated = tmp_path / "fastumi_gripper_calibration.yaml"
    generated.write_text("gripper_calibration: {}\n", encoding="utf-8")
    after = _launch_defaults(launch_file, monkeypatch, tmp_path / "share/gripper_openness")
    assert after["gripper_calibration_path"] == str(
        tmp_path / "share/gripper_openness/config/gripper_calibration.yaml"
    )


def test_source_checkout_uses_installed_configuration(tmp_path: Path, monkeypatch) -> None:
    """存在源码仓库时仍返回安装包路径，文件缺失也不回退到源码。"""
    module_path = tmp_path / "ros2_ws/src/gripper_openness/gripper_openness/paths.py"
    module_path.parent.mkdir(parents=True)
    source_calibration = module_path.parents[1] / "config/gripper_calibration.yaml"
    source_calibration.parent.mkdir()
    source_calibration.write_text("source: true\n", encoding="utf-8")
    monkeypatch.setattr(paths, "__file__", str(module_path))
    monkeypatch.setattr(paths, "get_package_share_directory", lambda _: str(tmp_path / "share"))
    expected = tmp_path / "share/config/gripper_calibration.yaml"
    assert not expected.exists()
    assert paths.default_gripper_calibration_path() == str(expected)


def test_standalone_install_uses_packaged_configuration(tmp_path: Path, monkeypatch) -> None:
    """独立安装从包共享目录获取随包配置。"""
    monkeypatch.setattr(paths, "__file__", str(tmp_path / "site-packages/gripper_openness/paths.py"))
    monkeypatch.setattr(paths, "get_package_share_directory", lambda _: str(tmp_path / "share"))
    assert paths.default_gripper_calibration_path() == str(
        tmp_path / "share/config/gripper_calibration.yaml"
    )

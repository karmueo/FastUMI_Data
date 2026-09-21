"""验证夹爪标定输出与两个预测入口使用同一份用户标定文件。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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
    context = LaunchContext()
    return {
        action.name: perform_substitutions(context, action.default_value)
        for action in module.generate_launch_description().entities
        if isinstance(action, DeclareLaunchArgument)
    }


def test_calibration_defaults_to_new_user_file(tmp_path: Path, monkeypatch) -> None:
    """首次标定可写入用户目录中的新文件。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    defaults = _launch_defaults(
        PACKAGE_ROOT / "launch" / "gripper_calibration.launch.py",
        monkeypatch,
        PACKAGE_ROOT,
    )
    assert defaults["output_path"] == str(tmp_path / "fastumi_gripper_calibration.yaml")
    assert defaults["overwrite"] == "false"
    assert defaults["image_topic"] == "/umi_camera/image_raw"
    assert not Path(defaults["output_path"]).exists()


@pytest.mark.parametrize(
    "launch_file",
    [PACKAGE_ROOT / "launch" / "gripper_openness.launch.py", ESTIMATOR_LAUNCH],
)
def test_prediction_prefers_saved_user_calibration(
    tmp_path: Path, monkeypatch, launch_file: Path
) -> None:
    """标定前使用包内示例，标定后两个预测入口均读取新文件。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    packaged = PACKAGE_ROOT / "config" / "calibration.yaml"
    before = _launch_defaults(launch_file, monkeypatch, PACKAGE_ROOT)
    assert before["gripper_calibration_path"] == str(packaged)
    assert before["image_topic"] == "/umi_camera/image_raw"

    generated = tmp_path / "fastumi_gripper_calibration.yaml"
    generated.write_text("gripper_calibration: {}\n", encoding="utf-8")
    after = _launch_defaults(launch_file, monkeypatch, PACKAGE_ROOT)
    assert after["gripper_calibration_path"] == str(generated)

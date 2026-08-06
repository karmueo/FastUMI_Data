"""验证 paired 与 pivot CLI 写入 schema v2 外参及客观质量结论。"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from fastumi_data import calibration_cli
from fastumi_data.calibration_solver import transform_to_document


def _input_document() -> dict:
    """构造满足 CLI 输入校验的最小配对和 pivot 样本。"""
    identity = transform_to_document(np.eye(4, dtype=np.float64))
    return {
        "samples": [
            {"base_tcp": identity, "vive_tracker": identity}
            for _ in range(8)
        ]
    }


def _write_input(path: Path) -> None:
    """写入固定的标定输入 YAML。"""
    path.write_text(
        yaml.safe_dump(_input_document(), sort_keys=False), encoding="utf-8"
    )


def _arguments(
    method: str, input_path: Path, output_path: Path, *, allow: bool = False
) -> list[str]:
    """返回调用目标标定子命令所需的参数。"""
    arguments = [
        method,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--tracker-serial",
        "LHR-TEST",
        "--fixture-version",
        "fixture-v1",
    ]
    if method == "pivot":
        arguments.extend(["--quaternion-xyzw", "0", "0", "0", "1"])
    if allow:
        arguments.append("--allow-high-residual")
    return arguments


@pytest.mark.parametrize("method", ["paired", "pivot"])
def test_quality_gate_acceptance_writes_v2_extrinsic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """门限内的 paired/pivot 结果必须写为 accepted 的 schema v2 外参。"""
    input_path = tmp_path / f"{method}.yaml"
    output_path = tmp_path / f"{method}-output.yaml"
    _write_input(input_path)
    if method == "paired":
        monkeypatch.setattr(
            calibration_cli,
            "solve_robot_world_hand_eye",
            lambda *args, **kwargs: SimpleNamespace(
                base_to_vive=np.eye(4, dtype=np.float64),
                tracker_to_tcp=np.eye(4, dtype=np.float64),
                translation_rmse_mm=2.0,
                rotation_rmse_deg=1.0,
                training_sample_count=6,
                validation_sample_count=2,
            ),
        )
    else:
        monkeypatch.setattr(
            calibration_cli,
            "solve_pivot_translation",
            lambda *args, **kwargs: (np.zeros(3, dtype=np.float64), 2.0),
        )

    calibration_cli.main(_arguments(method, input_path, output_path))

    document = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 2
    assert document["accepted"] is True
    assert "verified" not in document
    assert "calibration_verified" not in document


@pytest.mark.parametrize("method", ["paired", "pivot"])
def test_allowed_high_residual_writes_unaccepted_v2_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """显式允许超限时，paired/pivot 只能写 accepted 为 false 的诊断外参。"""
    input_path = tmp_path / f"{method}.yaml"
    output_path = tmp_path / f"{method}-output.yaml"
    _write_input(input_path)
    if method == "paired":
        monkeypatch.setattr(
            calibration_cli,
            "solve_robot_world_hand_eye",
            lambda *args, **kwargs: SimpleNamespace(
                base_to_vive=np.eye(4, dtype=np.float64),
                tracker_to_tcp=np.eye(4, dtype=np.float64),
                translation_rmse_mm=2.1,
                rotation_rmse_deg=1.0,
                training_sample_count=6,
                validation_sample_count=2,
            ),
        )
    else:
        monkeypatch.setattr(
            calibration_cli,
            "solve_pivot_translation",
            lambda *args, **kwargs: (np.zeros(3, dtype=np.float64), 2.1),
        )

    calibration_cli.main(_arguments(method, input_path, output_path, allow=True))

    document = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 2
    assert document["accepted"] is False
    assert "verified" not in document
    assert "calibration_verified" not in document


@pytest.mark.parametrize("method", ["paired", "pivot"])
def test_unallowed_high_residual_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """未允许超限时，paired/pivot 必须保持原有非写入失败行为。"""
    input_path = tmp_path / f"{method}.yaml"
    output_path = tmp_path / f"{method}-output.yaml"
    _write_input(input_path)
    if method == "paired":
        monkeypatch.setattr(
            calibration_cli,
            "solve_robot_world_hand_eye",
            lambda *args, **kwargs: SimpleNamespace(
                base_to_vive=np.eye(4, dtype=np.float64),
                tracker_to_tcp=np.eye(4, dtype=np.float64),
                translation_rmse_mm=2.1,
                rotation_rmse_deg=1.0,
                training_sample_count=6,
                validation_sample_count=2,
            ),
        )
    else:
        monkeypatch.setattr(
            calibration_cli,
            "solve_pivot_translation",
            lambda *args, **kwargs: (np.zeros(3, dtype=np.float64), 2.1),
        )

    with pytest.raises(RuntimeError, match="超过门限"):
        calibration_cli.main(_arguments(method, input_path, output_path))
    assert not output_path.exists()

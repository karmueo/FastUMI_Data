"""绑定 checkpoint 文件指纹、数据契约、训练 URDF 与示教 TCP 外参的部署档。

部署档由 ``make-profile`` 子命令从 checkpoint 与数据集 ``meta/fastumi.json`` 生成，
启动时逐项核对；任一文件或契约变化都会拒绝启动，更换模型须重新生成并审核。

用法（仓库根目录）::

    python -m dp_infer_lerobot.deployment make-profile --checkpoint <pretrained_model> \\
        --dataset-meta <lerobot>/meta/fastumi.json --urdf assets/rm_75.urdf --output <profile.yaml>
"""

import argparse
import hashlib
from pathlib import Path
import json

import numpy as np
import yaml

from .processors import read_processor_spec

# 部署端必须与训练数据一致的契约字段（取自 meta/fastumi.json）。
CONTRACT = {
    "fps": 30,
    "state_layout": "xyz_rotvec_gripper",
    "action_layout": "xyz_rotvec_gripper",
    "action_source": "demonstration_trajectory",
    "end_frame": "tcp",
    "tcp_definition": "gripper_grasp_center",
    "deployment_tool_offset": "deployment_calibration",
    "position_unit": "m",
    "rotation_unit": "rad",
    "gripper_representation": "normalized_0_1",
    "image_size": 224,
    "image_preprocessing": "letterbox_inter_area_black_border",
}
BASE_FILES = ("config.json", "model.safetensors", "policy_preprocessor.json",
              "policy_postprocessor.json")  # 另加两份处理器统计 safetensors。


def sha256_file(path):
    """分块计算文件 SHA-256，避免把权重整体读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_files(checkpoint):
    """返回 checkpoint 中需要绑定指纹的文件名列表。"""
    pre, _ = read_processor_spec(checkpoint)
    post, _ = read_processor_spec(checkpoint, inverse=True)
    return [*BASE_FILES, pre.name, post.name]


def load_profile(path):
    """读取并校验部署档格式与契约。

    Raises:
        ValueError: 文件缺失、版本或契约不符、矩阵非法。
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"deployment_config must name an existing file: {path}")
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("version") != 1 or profile.get("contract") != CONTRACT:
        raise ValueError("Unsupported deployment contract")
    urdf = profile.get("urdf_sha256")
    if not isinstance(urdf, str) or len(urdf) != 64:
        raise ValueError("Deployment profile must bind urdf_sha256")
    matrix = np.asarray(profile.get("umi_camera_from_tcp"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Deployment profile must embed a 4x4 umi_camera_from_tcp")
    return profile


def validate_checkpoint(checkpoint, profile):
    """核对 checkpoint 目录中每个必需文件的指纹；不读取训练恢复状态。"""
    root = Path(checkpoint)
    if not root.is_dir():
        raise ValueError("checkpoint must name a local pretrained directory")
    files = profile.get("checkpoint_files", {})
    if set(files) != set(required_files(root)):
        raise ValueError("Deployment profile must bind every required checkpoint file")
    for name, digest in files.items():
        if Path(name).name != name or sha256_file(root / name) != digest:
            raise ValueError(f"Checkpoint fingerprint mismatch: {name}")


def validate_export(export_dir, profile, precision):
    """核对 ``lerobot-export`` 模型包并绑定到部署档，返回 manifest。

    模型包不含 ``model.safetensors``：``assets/`` 中其余 checkpoint 文件须与部署档指纹一致，
    manifest 记录的源权重摘要须等于部署档的 ``model.safetensors`` 摘要；manifest 列出的全部文件
    （含 engine）逐一核对摘要。``validation_report.json`` 必须针对当前 manifest 且所选精度已通过验收。

    Args:
        export_dir: 模型包根目录（含 ``manifest.json``、``assets/`` 与 engine）。
        profile: :func:`load_profile` 返回的部署档。
        precision: ``fp32`` 或 ``fp16``。

    Raises:
        ValueError: 模型包不完整、被修改、与部署档不符或该精度未通过验证。
    """
    root = Path(export_dir)
    manifest_path, report_path = root / "manifest.json", root / "validation_report.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError("export_dir must name a complete lerobot-export package")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "complete"
            or manifest.get("policy_type") != "diffusion" or manifest.get("batch_size") != 1):
        raise ValueError("Unsupported or incomplete export package")
    for name, digest in manifest.get("files", {}).items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Export artifact checksum mismatch: {name}")
    files = profile.get("checkpoint_files", {})
    assets = [name for name in required_files(root / "assets") if name != "model.safetensors"]
    if set(files) != {*assets, "model.safetensors"}:
        raise ValueError("Deployment profile must bind every required checkpoint file")
    for name in assets:
        if manifest["files"].get(f"assets/{name}") != files[name]:
            raise ValueError(f"Export asset does not match the deployment profile: {name}")
    if manifest.get("checkpoint_sha256") != files["model.safetensors"]:
        raise ValueError("Export package was built from different model weights")
    for stage in ("encoder", "denoiser"):
        engine = manifest.get("stages", {}).get(stage, {}).get("engines", {}).get(precision)
        if not engine or f"{engine.get('path')}" not in manifest["files"]:
            raise ValueError(f"Export package lacks a verified {precision} {stage} engine")
    if not report_path.is_file():
        raise ValueError("Export package lacks validation_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    backend = report.get("backends", {}).get(f"tensorrt_{precision}", {})
    if (report.get("package_manifest_sha256") != sha256_file(manifest_path)
            or report.get("checkpoint_sha256") != files["model.safetensors"]):
        raise ValueError("Validation report does not belong to this export package")
    if backend.get("passed") is not True:
        raise ValueError(f"TensorRT {precision} engines have not passed export validation")
    return manifest


def validate_urdf(path, profile):
    """部署 FK 必须使用训练数据记录的同一 URDF。"""
    path = Path(path)
    if not path.is_file() or sha256_file(path) != profile["urdf_sha256"]:
        raise ValueError("URDF fingerprint does not match the training data")


def load_tool_offset(profile, end_camera_path, deploy_tcp_path, max_axis_deg):
    """读取部署外参并与示教 TCP 比对，返回 ``(T_Link7_TCP, 平移差 mm, 旋转差 度)``。

    复用 ``dp_infer.core.load_deploy_tcp`` 与 ``compare_tcp_frames``：
    ``T_Link7_TCP = inverse(T_cam_end) × T_d_cam_tcp``；两端相机←TCP 轴向夹角超过阈值时拒绝。
    """
    from dp_infer.core import compare_tcp_frames, load_deploy_tcp

    if not end_camera_path or not deploy_tcp_path:
        raise ValueError("TCP checkpoints require end_camera_extrinsic and deploy_tcp_extrinsic")
    link7_from_tcp, camera_from_tcp = load_deploy_tcp(end_camera_path, deploy_tcp_path)
    offset_mm, angle_deg = compare_tcp_frames(profile["umi_camera_from_tcp"], camera_from_tcp, max_axis_deg)
    return link7_from_tcp, offset_mm, angle_deg


def make_profile(checkpoint, dataset_meta, urdf, model_name):
    """由 checkpoint 与数据集元信息生成部署档字典，并核对 URDF 指纹。"""
    meta = json.loads(Path(dataset_meta).read_text(encoding="utf-8"))
    contract = {key: meta.get(key) for key in CONTRACT}
    if contract != CONTRACT:
        raise ValueError(f"Dataset contract differs from the supported one: {contract}")
    if sha256_file(urdf) != meta["urdf_sha256"]:
        raise ValueError("URDF does not match the dataset urdf_sha256")
    root = Path(checkpoint)
    return {
        "version": 1,
        "model_name": model_name,
        "contract": contract,
        "checkpoint_files": {name: sha256_file(root / name) for name in required_files(root)},
        "urdf_sha256": meta["urdf_sha256"],
        "umi_camera_from_tcp": meta["umi_camera_from_tcp"],
        "sources": {
            "dataset_repo_id": meta.get("repo_id"),
            "dataset_meta_sha256": sha256_file(dataset_meta),
            "umi_tcp_extrinsic_sha256": meta.get("umi_tcp_extrinsic_sha256"),
        },
    }


def main(argv=None):
    """命令行入口：``make-profile`` 生成部署档 YAML。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-profile")
    make.add_argument("--checkpoint", required=True)
    make.add_argument("--dataset-meta", required=True)
    make.add_argument("--urdf", required=True)
    make.add_argument("--model-name", required=True)
    make.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    profile = make_profile(args.checkpoint, args.dataset_meta, args.urdf, args.model_name)
    text = "# 模型、训练数据契约与示教 TCP 外参的冻结部署档；更换模型或数据后重新生成并审核。\n"
    Path(args.output).write_text(text + yaml.safe_dump(profile, sort_keys=False, allow_unicode=True),
                                 encoding="utf-8")
    load_profile(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

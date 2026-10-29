"""绑定 checkpoint 文件指纹、数据契约、训练 URDF 与示教 TCP 外参的部署档。

部署档由 ``make-profile`` 子命令从 checkpoint（或 ``lerobot-export`` 模型包）与数据集
``meta/fastumi.json`` 生成，启动时逐项核对；任一文件或契约变化都会拒绝启动，更换模型须重新生成并审核。

用法（仓库根目录）::

    python -m dp_infer_lerobot.deployment make-profile --checkpoint <pretrained_model> \\
        --dataset-meta <lerobot>/meta/fastumi.json --urdf assets/rm_75.urdf --output <profile.yaml>

只有模型包、没有权重时，把 ``--checkpoint`` 换成 ``--export-dir <包根目录>``：``assets/`` 指纹
先按包 manifest 核对，``model.safetensors`` 取 manifest 记录的 ``checkpoint_sha256``。
"""

import argparse
import hashlib
from pathlib import Path
import json

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .processors import read_processor_spec

DEFAULT_MAX_TCP_AXIS_DEG = 10.0  # 示教 TCP 与部署 TCP 坐标轴允许的最大夹角（度）。
ROTATION_TOLERANCE = 1e-5  # 标定文件按有限小数保存，刚体检查的正交性容差。

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
    """核对 TensorRT 模型包（``lerobot-export`` 或本机重建）并绑定到部署档，返回 manifest。

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


def rigid_transform(value, name):
    """要求 4×4 有限齐次刚体变换（米），返回 float64 副本；否则抛出 ValueError。"""
    try:
        matrix = np.array(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not a numeric matrix") from error
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=ROTATION_TOLERANCE)
            or abs(np.linalg.det(rotation) - 1) > ROTATION_TOLERANCE):
        raise ValueError(f"{name} must be a rigid transform")
    return matrix


def _read_yaml(path, name):
    """读取 YAML 映射；文件缺失或格式错误时抛出 ValueError。"""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{name} must name an existing file: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{name} {path} must be a YAML mapping")
    return document


def load_deploy_tcp(end_camera_path, deploy_tcp_path):
    """读取部署外参，返回 ``(T_Link7_TCP, T_Cd_TCP)``，平移单位米。

    Args:
        end_camera_path: ``T_cam_end.yaml``，``cam0.T_cam_hand`` 为部署相机←Link7。
        deploy_tcp_path: ``T_d_cam_tcp.yaml``，顶层 ``T_d_cam_tcp`` 为部署相机←TCP。

    Raises:
        ValueError: 文件缺失、缺少矩阵或矩阵不是刚体变换。
    """
    end = _read_yaml(end_camera_path, "end_camera_extrinsic").get("cam0") or {}
    if "T_cam_hand" not in end:
        raise ValueError(f"{end_camera_path} is missing cam0.T_cam_hand")
    camera_from_link7 = rigid_transform(end["T_cam_hand"], "T_cam_end")
    document = _read_yaml(deploy_tcp_path, "deploy_tcp_extrinsic")
    if "T_d_cam_tcp" not in document:
        raise ValueError(f"{deploy_tcp_path} is missing T_d_cam_tcp")
    camera_from_tcp = rigid_transform(document["T_d_cam_tcp"], "T_d_cam_tcp")
    link7_from_tcp = np.linalg.inv(camera_from_link7) @ camera_from_tcp
    # 文件精度有限，重新正交化，避免右乘八个目标时引入缩放。
    link7_from_tcp[:3, :3] = Rotation.from_matrix(link7_from_tcp[:3, :3]).as_matrix()
    return link7_from_tcp, camera_from_tcp


def compare_tcp_frames(umi_camera_from_tcp, deploy_camera_from_tcp,
                       max_axis_deg=DEFAULT_MAX_TCP_AXIS_DEG):
    """比较示教与部署两端的相机←TCP 外参，返回 ``(平移差 mm, 旋转差 度)``。

    两端 TCP 视为同一物理点，坐标轴必须采用同一约定。旋转差超过 ``max_axis_deg``
    说明轴向约定不一致（常见为 90°/180°），继续部署会让相对动作整体错乱。
    平移差和较小的旋转差来自两套夹具/相机安装差异，只影响图像视角，返回供日志记录。

    Raises:
        ValueError: 矩阵非法或旋转差超过阈值。
    """
    umi = rigid_transform(umi_camera_from_tcp, "umi_camera_from_tcp")
    deploy = rigid_transform(deploy_camera_from_tcp, "T_d_cam_tcp")
    difference = np.linalg.inv(umi) @ deploy
    angle = float(np.degrees(np.linalg.norm(Rotation.from_matrix(difference[:3, :3]).as_rotvec())))
    offset_mm = float(np.linalg.norm(difference[:3, 3]) * 1000)
    if angle > max_axis_deg:
        raise ValueError(
            f"UMI and deployment TCP axes differ by {angle:.1f} deg (> {max_axis_deg} deg); "
            "both TCP frames must use the same axis convention")
    return offset_mm, angle


def load_tool_offset(profile, end_camera_path, deploy_tcp_path, max_axis_deg=DEFAULT_MAX_TCP_AXIS_DEG):
    """读取部署外参并与示教 TCP 比对，返回 ``(T_Link7_TCP, 平移差 mm, 旋转差 度)``。

    ``T_Link7_TCP = inverse(T_cam_end) × T_d_cam_tcp``；两端相机←TCP 轴向夹角超过阈值时拒绝。
    """
    if not end_camera_path or not deploy_tcp_path:
        raise ValueError("TCP checkpoints require end_camera_extrinsic and deploy_tcp_extrinsic")
    link7_from_tcp, camera_from_tcp = load_deploy_tcp(end_camera_path, deploy_tcp_path)
    offset_mm, angle_deg = compare_tcp_frames(profile["umi_camera_from_tcp"], camera_from_tcp, max_axis_deg)
    return link7_from_tcp, offset_mm, angle_deg


def export_checkpoint_files(export_dir):
    """从模型包推导部署档的 ``checkpoint_files``：assets 按 manifest 核对后取摘要，权重取 manifest 记录值。

    Raises:
        ValueError: manifest 缺失、assets 被修改或缺少权重摘要。
    """
    root = Path(export_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest.get("files", {})
    assets = root / "assets"
    files = {}
    for name in required_files(assets):
        if name == "model.safetensors":
            continue
        digest = sha256_file(assets / name)
        if recorded.get(f"assets/{name}") != digest:
            raise ValueError(f"Export asset does not match its manifest: {name}")
        files[name] = digest
    weights = manifest.get("checkpoint_sha256")
    if not isinstance(weights, str) or len(weights) != 64:
        raise ValueError("Export manifest lacks checkpoint_sha256")
    files["model.safetensors"] = weights
    return {name: files[name] for name in required_files(assets)}


def make_profile(checkpoint, dataset_meta, urdf, model_name, export_dir=""):
    """由 checkpoint（或 ``export_dir`` 模型包）与数据集元信息生成部署档字典，并核对 URDF 指纹。"""
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
        "checkpoint_files": (export_checkpoint_files(export_dir) if export_dir
                             else {name: sha256_file(root / name) for name in required_files(root)}),
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
    model = make.add_mutually_exclusive_group(required=True)
    model.add_argument("--checkpoint")
    model.add_argument("--export-dir", help="没有权重时从 lerobot-export 模型包推导指纹")
    make.add_argument("--dataset-meta", required=True)
    make.add_argument("--urdf", required=True)
    make.add_argument("--model-name", required=True)
    make.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    profile = make_profile(args.checkpoint or "", args.dataset_meta, args.urdf, args.model_name,
                           args.export_dir or "")
    text = "# 模型、训练数据契约与示教 TCP 外参的冻结部署档；更换模型或数据后重新生成并审核。\n"
    Path(args.output).write_text(text + yaml.safe_dump(profile, sort_keys=False, allow_unicode=True),
                                 encoding="utf-8")
    load_profile(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

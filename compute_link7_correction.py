#!/usr/bin/env python3
"""由手指 ArUco 测量计算 Link7 部署修正 C，使 RM75 手指对齐训练时 UMI 手指的位置。

坐标记号：``T_A_B`` 为 B 系在 A 系中的位姿（把 B 系点变换到 A 系），平移单位米。
Cu 为 UMI 相机光学系，Cd 为 RM75 腕部相机光学系，F 为由两枚手指码定义的夹爪系。

训练标签按 Cu≡Cd 生成虚拟 Link7：``P = T_W_Cu · T_Cd_Link7(训练)``。要求 RM75 手指与训练手指
重合，解得 ``T_W_Link7 = P · C``，其中::

    C = T_Link7_Cd(训练) · T_Cu_F · T_F_Cd · T_Cd_Link7(部署)

F 的定义在 ``T_Cu_F · T_F_Cd`` 中相互抵消，只要两种夹爪上手指码相对指尖的安装一致即可。
部署时策略输出右乘 C，喂给策略的 FK 位姿右乘 C⁻¹（见 ros2_ws/src/dp_infer）。

用法（ros2_ws/.venv-numpy1，已 source /opt/ros/humble）::

    python compute_link7_correction.py \\
        --training-dataset dataset/processed/jingbao_v1 \\
        --rollout-bag dataset/h5dy_data/dp_infer/rollout/episode_8/bag \\
        --output ros2_ws/src/dp_infer/config/link7_correction.yaml

rollout 录制须包含腕部原始鱼眼压缩图像（默认 ``/wrist_camera/image_raw/compressed``，手眼标定会话用
``--rollout-topic /hand_eye/sample_image``），且手指两枚码完整可见。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

# 两种夹爪手指上的码：DICT_4X4_50，左指 id 0、右指 id 1，边长 16 mm。
DICTIONARY = cv2.aruco.DICT_4X4_50
LEFT_ID, RIGHT_ID = 0, 1
MARKER_SIZE_M = 0.016
IMAGE_TOPIC = "/wrist_camera/image_raw/compressed"


def load_fisheye_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 Kalibr 风格 cam0 鱼眼标定，返回 K[3,3] 与 D[4,1]。"""
    cam = yaml.safe_load(path.read_text(encoding="utf-8"))["cam0"]
    if cam.get("distortion_model") != "fisheye":
        raise ValueError(f"{path} must use the fisheye distortion model")
    fx, fy, cx, cy = (float(value) for value in cam["intrinsics"])
    camera = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    distortion = np.asarray(cam["distortion_coeffs"], dtype=np.float64).reshape(4, 1)
    return camera, distortion


class FingerTagDetector:
    """检测两枚手指码并以 IPPE_SQUARE 求各自在相机光学系下的位姿。"""

    def __init__(self, camera: np.ndarray, distortion: np.ndarray):
        self.camera = camera
        self.distortion = distortion
        dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        half = MARKER_SIZE_M / 2
        # IPPE_SQUARE 要求物点从左上角起顺时针。
        self.object_points = np.array(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]])

    def detect(self, bgr: np.ndarray) -> dict[int, np.ndarray] | None:
        """返回 {id: T_C_tag[4,4]}；两枚码未同时检测到时返回 None。"""
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        if ids is None:
            return None
        poses = {}
        for corner, marker_id in zip(corners, ids.ravel()):
            if int(marker_id) not in (LEFT_ID, RIGHT_ID):
                continue
            undistorted = cv2.fisheye.undistortPoints(
                corner.reshape(-1, 1, 2).astype(np.float64), self.camera, self.distortion,
                P=self.camera).reshape(4, 2)
            ok, rvec, tvec = cv2.solvePnP(
                self.object_points, undistorted, self.camera, None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_rotvec(rvec.ravel()).as_matrix()
                pose[:3, 3] = tvec.ravel()
                poses[int(marker_id)] = pose
        return poses if LEFT_ID in poses and RIGHT_ID in poses else None


def finger_frame(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """由两枚码位姿构造夹爪系 T_C_F：原点为两码中点，x 沿左→右张合方向，z 为两码法向均值。"""
    x_axis = right[:3, 3] - left[:3, 3]
    x_axis /= np.linalg.norm(x_axis)
    normal = left[:3, 2] + right[:3, 2]
    normal -= normal.dot(x_axis) * x_axis
    normal /= np.linalg.norm(normal)
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack([x_axis, np.cross(normal, x_axis), normal])
    frame[:3, 3] = (left[:3, 3] + right[:3, 3]) / 2
    return frame


def mean_frame(frames: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
    """平均刚体变换；返回均值、平移标准差（米，三轴合成）与旋转相对均值的 RMS（度）。"""
    if not frames:
        raise ValueError("no frame contains both finger markers")
    stack = np.stack(frames)
    rotations = Rotation.from_matrix(stack[:, :3, :3])
    mean_rotation = rotations.mean()
    mean = np.eye(4)
    mean[:3, :3] = mean_rotation.as_matrix()
    mean[:3, 3] = stack[:, :3, 3].mean(axis=0)
    translation_std = float(np.linalg.norm(stack[:, :3, 3].std(axis=0)))
    rotation_rms = float(np.degrees(np.sqrt(np.mean(
        (rotations * mean_rotation.inv()).magnitude() ** 2))))
    return mean, translation_std, rotation_rms


def link7_correction(
    T_Cu_F: np.ndarray, T_Cd_F: np.ndarray,
    T_Cd_Link7_training: np.ndarray, T_Cd_Link7_deploy: np.ndarray,
) -> np.ndarray:
    """C = T_Link7_Cd(训练) · T_Cu_F · T_F_Cd · T_Cd_Link7(部署)，即真实 Link7 在虚拟 Link7 系中的位姿。"""
    return (np.linalg.inv(T_Cd_Link7_training) @ T_Cu_F @ np.linalg.inv(T_Cd_F)
            @ T_Cd_Link7_deploy)


def training_frames(dataset: Path, detector: FingerTagDetector,
                    episode_stride: int, frame_stride: int) -> tuple[list[np.ndarray], np.ndarray]:
    """从训练 HDF5 原图采样 T_Cu_F，并返回生成标签时使用的 T_Cd_Link7。"""
    import h5py

    files = sorted(dataset.glob("records/**/episodes/*.hdf5"))[::episode_stride]
    if not files:
        raise FileNotFoundError(f"no episode HDF5 under {dataset}/records")
    frames, hand_eye = [], None
    for path in files:
        with h5py.File(path, "r") as episode:
            value = episode.attrs["T_Cd_Link7"]
            # 转换器以 JSON 字符串保存 4x4 外参。
            current = np.asarray(json.loads(value) if isinstance(value, (str, bytes)) else value,
                                 dtype=np.float64)
            if hand_eye is None:
                hand_eye = current
            elif not np.allclose(hand_eye, current, atol=1e-9):
                raise ValueError(f"{path} uses a different T_Cd_Link7 from earlier episodes")
            encoding = episode.attrs["image_encoding"]
            if (encoding.decode() if isinstance(encoding, bytes) else encoding) != "rgb8":
                raise ValueError(f"{path} images must be rgb8")
            images = episode["observations/images/front"]
            for index in range(0, images.shape[0], frame_stride):
                poses = detector.detect(cv2.cvtColor(images[index], cv2.COLOR_RGB2BGR))
                if poses is not None:
                    frames.append(finger_frame(poses[LEFT_ID], poses[RIGHT_ID]))
    return frames, hand_eye


def rollout_frames(bags: list[Path], detector: FingerTagDetector,
                   frame_stride: int, topic: str = IMAGE_TOPIC) -> list[np.ndarray]:
    """从实机录制的压缩腕部图像采样 T_Cd_F。"""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage

    frames = []
    for bag in bags:
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                    rosbag2_py.ConverterOptions("cdr", "cdr"))
        reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
        count = 0
        while reader.has_next():
            _, raw, _ = reader.read_next()
            count += 1
            if count % frame_stride:
                continue
            message = deserialize_message(raw, CompressedImage)
            bgr = cv2.imdecode(np.frombuffer(bytes(message.data), np.uint8), cv2.IMREAD_COLOR)
            poses = None if bgr is None else detector.detect(bgr)
            if poses is not None:
                frames.append(finger_frame(poses[LEFT_ID], poses[RIGHT_ID]))
    return frames


def load_hand_eye(path: Path) -> np.ndarray:
    """读取 T_cam_end.yaml 的 cam0.T_cam_hand（^cam T_Link7）。"""
    return np.asarray(yaml.safe_load(path.read_text(encoding="utf-8"))["cam0"]["T_cam_hand"],
                      dtype=np.float64)


def _sha256(matrix: np.ndarray) -> str:
    return hashlib.sha256(json.dumps(np.round(matrix, 12).tolist()).encode()).hexdigest()


def correction_document(C, T_Cu_F, T_Cd_F, T_Cd_Link7_training, T_Cd_Link7_deploy,
                        stats, sources) -> dict:
    """组装输出 YAML；dp_infer 只读取 T_virtual_link7，其余字段供追溯与复核。"""
    rotation = Rotation.from_matrix(C[:3, :3])
    return {
        "link7_correction": {
            "T_virtual_link7": np.round(C, 9).tolist(),
            "rotation_deg": round(float(np.degrees(rotation.magnitude())), 3),
            "translation_mm": np.round(C[:3, 3] * 1000, 2).tolist(),
            "T_Cu_F": np.round(T_Cu_F, 9).tolist(),
            "T_Cd_F": np.round(T_Cd_F, 9).tolist(),
            "T_Cd_Link7_training_sha256": _sha256(T_Cd_Link7_training),
            "T_Cd_Link7_deploy_sha256": _sha256(T_Cd_Link7_deploy),
            "samples": stats,
            "sources": sources,
        }
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training-dataset", type=Path, required=True,
                        help="convert_hardware_mcap 输出目录，含 records/**/episodes/*.hdf5")
    parser.add_argument("--rollout-bag", type=Path, action="append", required=True,
                        help="实机录制 MCAP 目录，可重复指定")
    parser.add_argument("--rollout-topic", default=IMAGE_TOPIC,
                        help="腕部原始鱼眼 CompressedImage 话题；手眼录制用 /hand_eye/sample_image")
    parser.add_argument("--umi-camera", type=Path, default=Path("config/calibration/UmiCalib.yaml"))
    parser.add_argument("--robot-camera", type=Path,
                        default=Path("ros2_ws/src/fastumi_usb_camera/config/EndEffectorCalib.yaml"))
    parser.add_argument("--deploy-hand-eye", type=Path,
                        help="当前部署的 T_cam_end.yaml；省略时沿用训练数据的 T_Cd_Link7")
    parser.add_argument("--episode-stride", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=15)
    parser.add_argument("--rollout-frame-stride", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    umi = FingerTagDetector(*load_fisheye_camera(args.umi_camera))
    robot = FingerTagDetector(*load_fisheye_camera(args.robot_camera))
    umi_frames, T_Cd_Link7_training = training_frames(
        args.training_dataset, umi, args.episode_stride, args.frame_stride)
    robot_frames = rollout_frames(args.rollout_bag, robot, args.rollout_frame_stride,
                                  args.rollout_topic)
    T_Cu_F, umi_std, umi_rms = mean_frame(umi_frames)
    T_Cd_F, robot_std, robot_rms = mean_frame(robot_frames)
    T_Cd_Link7_deploy = (load_hand_eye(args.deploy_hand_eye) if args.deploy_hand_eye
                         else T_Cd_Link7_training)
    C = link7_correction(T_Cu_F, T_Cd_F, T_Cd_Link7_training, T_Cd_Link7_deploy)
    stats = {
        "umi_frames": len(umi_frames), "umi_translation_std_mm": round(umi_std * 1000, 2),
        "umi_rotation_rms_deg": round(umi_rms, 2),
        "robot_frames": len(robot_frames), "robot_translation_std_mm": round(robot_std * 1000, 2),
        "robot_rotation_rms_deg": round(robot_rms, 2),
    }
    sources = {
        "training_dataset": str(args.training_dataset),
        "rollout_bags": [str(path) for path in args.rollout_bag],
        "rollout_topic": args.rollout_topic,
        "umi_camera": str(args.umi_camera), "robot_camera": str(args.robot_camera),
        "deploy_hand_eye": str(args.deploy_hand_eye or "training T_Cd_Link7"),
    }
    document = correction_document(C, T_Cu_F, T_Cd_F, T_Cd_Link7_training,
                                   T_Cd_Link7_deploy, stats, sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Link7 部署修正 C = T_Link7_Cd(训练)·T_Cu_F·T_F_Cd·T_Cd_Link7(部署)，"
              "由 compute_link7_correction.py 生成。\n"
              "# dp_infer：发布目标 = 策略 Link7 · C，策略位姿观测 = FK Link7 · C⁻¹；平移单位米。\n"
              "# 重装相机、重做手眼标定或更换训练数据后须重新生成。\n")
    args.output.write_text(header + yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), **stats,
                      "rotation_deg": document["link7_correction"]["rotation_deg"],
                      "translation_mm": document["link7_correction"]["translation_mm"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

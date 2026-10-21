"""标定板（Kalibr AprilGrid / 棋盘格）配置、腕部鱼眼相机模型、角点检测与 PnP。

图像为 BGR uint8，像素坐标原点在左上角。鱼眼模型为 OpenCV fisheye（equidistant，
k1..k4）。PnP 先 ``cv2.fisheye.undistortPoints`` 到归一化平面，再按 K=I、无畸变求
^cam T_board；重投影误差按原始鱼眼像素计算。
"""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from fastumi_bringup.transforms import make_transform


TAG_FAMILIES = {"tag36h11": cv2.aruco.DICT_APRILTAG_36h11}


@dataclass(frozen=True)
class AprilGridSpec:
    """Kalibr AprilGrid：tag_cols 列 × tag_rows 行标签，tag_size_m 为标签黑框边长（米）。

    几何约定与 fastumi_data/tracker_camera_config.py 一致：板系原点为 Tag0 左下检测角，
    x 向右、y 向上、z 指向观察者（相机在 +z 侧）；ID 行优先，Tag0 在左下。
    特征点 id = tag_id * 4 + k，k 依次为左下、右下、右上、左上角。
    """

    tag_cols: int
    tag_rows: int
    tag_size_m: float
    tag_spacing: float
    tag_family: str = "tag36h11"

    kind = "aprilgrid"

    def __post_init__(self):
        if self.tag_cols < 2 or self.tag_rows < 2:
            raise ValueError("AprilGrid 至少需要 2x2 个标签")
        if not math.isfinite(self.tag_size_m) or self.tag_size_m <= 0.0:
            raise ValueError("AprilGrid tagSize 必须为正数")
        if not math.isfinite(self.tag_spacing) or self.tag_spacing < 0.0:
            raise ValueError("AprilGrid tagSpacing 不能为负数")
        if self.tag_family not in TAG_FAMILIES:
            raise ValueError(f"不支持的标签族 {self.tag_family}；仅支持 {sorted(TAG_FAMILIES)}")

    @property
    def tag_count(self):
        """板上标签数。"""
        return self.tag_cols * self.tag_rows

    @property
    def corner_count(self):
        """整板特征点数（每个标签 4 个检测角）。"""
        return 4 * self.tag_count

    @property
    def pitch_m(self):
        """相邻标签同名角点的间距（米）。"""
        return self.tag_size_m * (1.0 + self.tag_spacing)

    @property
    def size_m(self):
        """最外侧标签检测角包围框的宽、高（米）。"""
        return (self.tag_size_m + (self.tag_cols - 1) * self.pitch_m,
                self.tag_size_m + (self.tag_rows - 1) * self.pitch_m)

    def object_points(self, ids):
        """特征点 id 对应的 ^board p (N,3)，单位米。"""
        ids = np.asarray(ids, dtype=int).reshape(-1)
        tag, corner = np.divmod(ids, 4)
        row, column = np.divmod(tag, self.tag_cols)
        offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64) * self.tag_size_m
        points = np.zeros((len(ids), 3))
        points[:, 0] = column * self.pitch_m + offsets[corner, 0]
        points[:, 1] = row * self.pitch_m + offsets[corner, 1]
        return points

    def outline_points(self):
        """必须完整可见的区域：全部标签的包围框四角（米）。"""
        width, height = self.size_m
        return np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]],
                        dtype=np.float64)

    def center_point(self):
        """板中心在 board 系下的坐标（米）。"""
        width, height = self.size_m
        return np.array([width / 2.0, height / 2.0, 0.0])

    def hand_eye_pattern(self):
        """hand_eye_calibration 不支持 AprilTag，返回 None。"""
        return None

    def to_dict(self):
        """Kalibr target YAML 键，写入 MCAP 会话信息，可被 board_spec_from_dict 读回。"""
        return {"target_type": "aprilgrid", "tagCols": self.tag_cols, "tagRows": self.tag_rows,
                "tagSize": self.tag_size_m, "tagSpacing": self.tag_spacing,
                "tagFamily": self.tag_family}


@dataclass(frozen=True)
class CheckerboardSpec:
    """棋盘格：target_cols × target_rows 个内角点，间距单位米（Kalibr checkerboard 键）。

    几何约定与 fastumi_data/tracker_camera_config.py 及 OpenCV 一致：原点为左上内角点，
    x 沿列、y 沿行（向下）、z 指向板内（相机在 -z 侧）；特征点 id 为行优先序号。
    行列数须一奇一偶，使板不具 180° 对称，检测器据此统一角点顺序。
    """

    target_cols: int
    target_rows: int
    row_spacing_m: float
    col_spacing_m: float

    kind = "checkerboard"

    def __post_init__(self):
        if self.target_cols < 3 or self.target_rows < 3:
            raise ValueError("棋盘格内角点行列数至少为 3")
        if (self.target_cols + self.target_rows) % 2 == 0:
            raise ValueError("棋盘格内角点行列数须一奇一偶，否则 180° 方向无法区分")
        for value in (self.row_spacing_m, self.col_spacing_m):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("棋盘格间距必须为正数")

    @property
    def corner_count(self):
        """整板内角点数。"""
        return self.target_cols * self.target_rows

    @property
    def size_m(self):
        """含外圈方格的整板宽、高（米）。"""
        return ((self.target_cols + 1) * self.col_spacing_m,
                (self.target_rows + 1) * self.row_spacing_m)

    def object_points(self, ids):
        """内角点 id（行优先）对应的 ^board p (N,3)，单位米。"""
        row, column = np.divmod(np.asarray(ids, dtype=int).reshape(-1), self.target_cols)
        points = np.zeros((len(row), 3))
        points[:, 0] = column * self.col_spacing_m
        points[:, 1] = row * self.row_spacing_m
        return points

    def outline_points(self):
        """必须完整可见的区域：内角点包围框各向外扩一格（外圈方格也要可见才能检测）。"""
        dx, dy = self.col_spacing_m, self.row_spacing_m
        x1, y1 = self.target_cols * dx, self.target_rows * dy
        return np.array([[-dx, -dy, 0], [x1, -dy, 0], [x1, y1, 0], [-dx, y1, 0]],
                        dtype=np.float64)

    def center_point(self):
        """内角点中心在 board 系下的坐标（米）。"""
        return np.array([(self.target_cols - 1) * self.col_spacing_m / 2.0,
                         (self.target_rows - 1) * self.row_spacing_m / 2.0, 0.0])

    def hand_eye_pattern(self):
        """hand_eye_calibration 的 ``--calibration-pattern``：``2 rows cols cell``（内角点）。"""
        if not math.isclose(self.row_spacing_m, self.col_spacing_m):
            return None
        return f"2 {self.target_rows} {self.target_cols} {self.col_spacing_m:g}"

    def to_dict(self):
        """Kalibr target YAML 键，写入 MCAP 会话信息，可被 board_spec_from_dict 读回。"""
        return {"target_type": "checkerboard", "targetCols": self.target_cols,
                "targetRows": self.target_rows, "rowSpacingMeters": self.row_spacing_m,
                "colSpacingMeters": self.col_spacing_m}


def board_spec_from_dict(data):
    """由 Kalibr target 字典（target_type: aprilgrid | checkerboard）构造板规格。"""
    target_type = data.get("target_type")
    try:
        if target_type == "aprilgrid":
            return AprilGridSpec(
                tag_cols=int(data["tagCols"]), tag_rows=int(data["tagRows"]),
                tag_size_m=float(data["tagSize"]), tag_spacing=float(data["tagSpacing"]),
                tag_family=str(data.get("tagFamily", "tag36h11")))
        if target_type == "checkerboard":
            return CheckerboardSpec(
                target_cols=int(data["targetCols"]), target_rows=int(data["targetRows"]),
                row_spacing_m=float(data["rowSpacingMeters"]),
                col_spacing_m=float(data["colSpacingMeters"]))
    except KeyError as error:
        raise ValueError(f"{target_type} 标定板配置缺少键 {error}") from error
    raise ValueError(f"标定板 target_type 必须是 aprilgrid 或 checkerboard，实际为 {target_type!r}")


def load_board_spec(path):
    """读取 Kalibr target YAML（如 config/calibration/april_6x6.yaml）。"""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return board_spec_from_dict(data)


@dataclass(frozen=True)
class FisheyeCamera:
    """OpenCV fisheye 相机：K 3x3，D (k1..k4)，分辨率 width x height 像素。"""

    K: np.ndarray
    D: np.ndarray
    width: int
    height: int

    def scaled(self, width, height):
        """按新分辨率等比例缩放内参（畸变系数不变）。"""
        scale_x, scale_y = width / self.width, height / self.height
        K = self.K.copy()
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        return FisheyeCamera(K=K, D=self.D.copy(), width=width, height=height)

    def project(self, points_camera):
        """相机系 3D 点 (N,3) 投影到鱼眼像素 (N,2)；z ≤ 0 的点返回 NaN。"""
        points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
        pixels = np.full((len(points), 2), np.nan)
        valid = points[:, 2] > 1e-6
        if np.any(valid):
            projected, _ = cv2.fisheye.projectPoints(
                points[valid].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.D)
            pixels[valid] = projected.reshape(-1, 2)
        return pixels

    def to_dict(self):
        """写入 MCAP 会话信息用的普通字典。"""
        return {"model": "fisheye", "K": self.K.reshape(-1).tolist(),
                "D": self.D.reshape(-1).tolist(), "width": self.width, "height": self.height}


def load_fisheye_camera(path, camera_key="cam0"):
    """读取 Kalibr 风格内参 YAML：intrinsics [fx,fy,cx,cy]、distortion_coeffs、resolution。"""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    camera = data.get(camera_key, data)
    model = str(camera.get("distortion_model", "fisheye")).lower()
    if model not in ("fisheye", "equidistant"):
        raise ValueError(f"腕部相机需为 fisheye/equidistant 畸变模型，实际为 {model}")
    fx, fy, cx, cy = (float(v) for v in camera["intrinsics"])
    width, height = (int(v) for v in camera["resolution"])
    D = np.array([float(v) for v in camera["distortion_coeffs"]], dtype=np.float64)
    if D.shape != (4,):
        raise ValueError("fisheye 畸变系数必须为 k1..k4 四个值")
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return FisheyeCamera(K=K, D=D.reshape(4, 1), width=width, height=height)


@dataclass
class BoardObservation:
    """单帧检测结果：特征点 id（含义见各板规格）与鱼眼像素坐标；位姿可能为空。"""

    ids: np.ndarray
    pixels: np.ndarray
    marker_count: int
    camera_from_board: np.ndarray | None = None
    reprojection_rms_px: float = math.inf

    @property
    def corner_count(self):
        """检测到的特征点数。"""
        return int(len(self.ids))


class _FisheyePnP:
    """检测器共用：鱼眼去畸变后 PnP，填入 ^cam T_board 与鱼眼像素重投影 RMS。"""

    min_points = 6

    def __init__(self, spec, camera: FisheyeCamera):
        self.spec = spec
        self.camera = camera

    def _check_size(self, image):
        if image.shape[1] != self.camera.width or image.shape[0] != self.camera.height:
            raise ValueError(
                f"图像尺寸 {image.shape[1]}x{image.shape[0]} 与内参 "
                f"{self.camera.width}x{self.camera.height} 不符")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    def _finish(self, ids, pixels, marker_count):
        observation = BoardObservation(
            ids=np.asarray(ids, dtype=int).reshape(-1),
            pixels=np.asarray(pixels, dtype=np.float64).reshape(-1, 2),
            marker_count=marker_count)
        if observation.corner_count >= self.min_points:
            self.estimate_pose(observation)
        return observation

    def estimate_pose(self, observation):
        """鱼眼去畸变后 PnP，填入 ^cam T_board 与鱼眼像素重投影 RMS。"""
        object_points = self.spec.object_points(observation.ids)
        normalized = cv2.fisheye.undistortPoints(
            observation.pixels.reshape(-1, 1, 2), self.camera.K, self.camera.D
        ).reshape(-1, 2)
        identity = np.eye(3)
        ok, rvec, tvec = cv2.solvePnP(
            object_points, normalized, identity, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return observation
        rvec, tvec = cv2.solvePnPRefineLM(object_points, normalized, identity, None, rvec, tvec)
        rotation, _ = cv2.Rodrigues(rvec)
        transform = make_transform(rotation, tvec.reshape(3))
        if transform[2, 3] <= 0.0:
            return observation
        projected = self.camera.project(
            (rotation @ object_points.T).T + tvec.reshape(1, 3))
        error = projected - observation.pixels
        observation.camera_from_board = transform
        observation.reprojection_rms_px = float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))
        return observation


class AprilGridDetector(_FisheyePnP):
    """在原始鱼眼图上检测 AprilGrid 标签；局部遮挡时用剩余标签求位姿。

    检测参数与角点重排沿用 fastumi_data/tracker_camera_detection.py 的
    OpenCvAprilTagDetector（已在实板 Kalibr AprilGrid 上验证）：OpenCV 角点顺序
    （标签图像左上、右上、右下、左下）按 [1, 0, 3, 2] 重排为 Kalibr 的左下、右下、右上、左上。
    """

    KALIBR_CORNER_INDICES = np.array([1, 0, 3, 2])
    subpix_window = 3
    max_shift_px = 2.0

    def __init__(self, spec: AprilGridSpec, camera: FisheyeCamera):
        super().__init__(spec, camera)
        parameters = cv2.aruco.DetectorParameters()
        parameters.errorCorrectionRate = 1.0
        parameters.perspectiveRemovePixelPerCell = 16
        parameters.perspectiveRemoveIgnoredMarginPerCell = 0.25
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILIES[spec.tag_family])
        dictionary.maxCorrectionBits = 3
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect(self, image_bgr):
        """检测标签角点；超出板范围或重复的 ID 丢弃。"""
        gray = self._check_size(image_bgr)
        corners, tag_ids, _ = self.detector.detectMarkers(gray)
        if tag_ids is None:
            return self._finish(np.zeros(0, dtype=int), np.zeros((0, 2)), 0)
        tag_ids = tag_ids.reshape(-1)
        unique, counts = np.unique(tag_ids, return_counts=True)
        duplicated = set(unique[counts > 1].tolist())
        ids, pixels = [], []
        for tag_id, tag_corners in sorted(zip(tag_ids.tolist(), corners), key=lambda item: item[0]):
            if tag_id >= self.spec.tag_count or tag_id in duplicated:
                continue
            ids.extend(4 * tag_id + np.arange(4))
            pixels.append(np.asarray(tag_corners, dtype=np.float64).reshape(4, 2)[
                self.KALIBR_CORNER_INDICES])
        pixels = np.vstack(pixels) if pixels else np.zeros((0, 2))
        if len(pixels):
            pixels = self._refine_x_junctions(gray, pixels)
        return self._finish(ids, pixels, len(pixels) // 4)

    def _refine_x_junctions(self, gray, pixels):
        """Kalibr 板标签角与间隙小黑方块构成 X 形交点，用 cornerSubPix 细化。

        合成鱼眼图上角点误差由约 1 px 降至约 0.4 px；移动超过 max_shift_px 的点
        （无小方块的板或模糊）保留轮廓拟合结果。
        """
        refined = pixels.astype(np.float32).reshape(-1, 1, 2).copy()
        cv2.cornerSubPix(gray, refined, (self.subpix_window, self.subpix_window), (-1, -1), (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
        refined = refined.reshape(-1, 2).astype(np.float64)
        shift = np.linalg.norm(refined - pixels, axis=1)
        return np.where((shift <= self.max_shift_px)[:, None], refined, pixels)


class CheckerboardDetector(_FisheyePnP):
    """在原始鱼眼图上检测整块棋盘格，并把角点顺序统一到板系。

    OpenCV 返回的角点顺序可能整体旋转 180° 或镜像。统一规则：
    1. 手性：图像中列方向 × 行方向须与板系 x × y 同号（相机在正面，像素 y 向下）；
    2. 颜色：行列一奇一偶的板对角两个外圈方格颜色相反，要求角点 0 外侧对角方格比
       其右侧相邻外圈方格更暗（即左上外圈方格为黑），否则整体反序。
    """

    def __init__(self, spec: CheckerboardSpec, camera: FisheyeCamera):
        super().__init__(spec, camera)
        self.pattern_size = (spec.target_cols, spec.target_rows)

    def _find(self, gray):
        found, corners = cv2.findChessboardCornersSB(
            gray, self.pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
        if found:
            return corners.reshape(-1, 2).astype(np.float64)
        found, corners = cv2.findChessboardCorners(
            gray, self.pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            return None
        corners = corners.astype(np.float32).reshape(-1, 1, 2)
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
        return corners.reshape(-1, 2).astype(np.float64)

    @staticmethod
    def _mean_gray(gray, point, radius):
        x, y = int(round(point[0])), int(round(point[1]))
        height, width = gray.shape
        if not (radius <= x < width - radius and radius <= y < height - radius):
            return None
        return float(gray[y - radius:y + radius + 1, x - radius:x + radius + 1].mean())

    def canonical_order(self, gray, corners):
        """把 (rows*cols, 2) 角点重排为板系行优先顺序；无法判定颜色时返回 None。"""
        grid = corners.reshape(self.spec.target_rows, self.spec.target_cols, 2)
        column_step = grid[0, 1] - grid[0, 0]
        row_step = grid[1, 0] - grid[0, 0]
        if column_step[0] * row_step[1] - column_step[1] * row_step[0] < 0.0:
            grid = grid[:, ::-1]
            column_step = grid[0, 1] - grid[0, 0]
            row_step = grid[1, 0] - grid[0, 0]
        radius = max(1, int(0.15 * min(np.linalg.norm(column_step), np.linalg.norm(row_step))))
        outer_corner = grid[0, 0] - 0.5 * column_step - 0.5 * row_step
        outer_next = grid[0, 0] + 0.5 * column_step - 0.5 * row_step
        dark = self._mean_gray(gray, outer_corner, radius)
        light = self._mean_gray(gray, outer_next, radius)
        if dark is None or light is None:
            return None
        if dark > light:
            grid = grid[::-1, ::-1]
        return grid.reshape(-1, 2)

    def detect(self, image_bgr):
        """仅接受整板检测；角点顺序统一后求位姿。"""
        gray = self._check_size(image_bgr)
        corners = self._find(gray)
        if corners is not None:
            corners = self.canonical_order(gray, corners)
        if corners is None:
            return self._finish(np.zeros(0, dtype=int), np.zeros((0, 2)), 0)
        return self._finish(np.arange(self.spec.corner_count), corners, 1)


def make_detector(spec, camera: FisheyeCamera):
    """按板类型返回检测器（AprilGridDetector / CheckerboardDetector）。"""
    if isinstance(spec, AprilGridSpec):
        return AprilGridDetector(spec, camera)
    if isinstance(spec, CheckerboardSpec):
        return CheckerboardDetector(spec, camera)
    raise TypeError(f"不支持的标定板类型 {type(spec).__name__}")

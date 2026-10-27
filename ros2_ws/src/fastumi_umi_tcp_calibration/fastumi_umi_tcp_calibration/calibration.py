"""对双 TAG 观测执行质量筛选与鲁棒聚合，输出相机系到 TCP 的米制变换。"""

import math

import numpy as np

from fastumi_umi_tcp_calibration.dual_tag import robust_average
from fastumi_umi_tcp_calibration.transforms import rotation_angle


def transform_spread(transforms, mean):
    """返回相对于指定均值的平移/旋转 RMS 与 P95；单位为毫米和度。"""
    translations = np.array([np.linalg.norm(t[:3, 3] - mean[:3, 3]) * 1000 for t in transforms])
    rotations = np.array(
        [math.degrees(rotation_angle(mean[:3, :3], t[:3, :3])) for t in transforms]
    )
    return {
        "translation_rms_mm": float(np.sqrt(np.mean(translations**2))),
        "translation_p95_mm": float(np.percentile(translations, 95)),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotations**2))),
        "rotation_p95_deg": float(np.percentile(rotations, 95)),
    }


def solve(observations, tag_from_tcp, settings):
    """对已通过单帧门限的观测聚合；返回矩阵、内点掩码及诊断，由调用方检查最小内点数量。"""
    average = robust_average(
        [item.camera_from_utag for item in observations],
        k=settings.mad_k,
        translation_floor_m=settings.translation_floor_m,
        rotation_floor_deg=settings.rotation_floor_deg,
    )
    kept = [item for item, keep in zip(observations, average.inliers) if keep]
    camera_from_tcp = average.transform @ tag_from_tcp
    report = {
        "single_frame_valid": len(observations),
        "inliers": len(kept),
        "T_u_cam_tag": average.transform.tolist(),
        "T_u_tag_tcp": tag_from_tcp.tolist(),
        "T_u_cam_tcp": camera_from_tcp.tolist(),
        "tag_spread": transform_spread([item.camera_from_utag for item in kept], average.transform),
        "tcp_spread": transform_spread(
            [item.camera_from_utag @ tag_from_tcp for item in kept], camera_from_tcp
        ),
        "mean_reprojection_rms_px": float(np.mean([item.reprojection_rms_px for item in kept])),
        "tag_distance_mm": {
            "min": float(min(item.tag_distance_m for item in kept) * 1000),
            "max": float(max(item.tag_distance_m for item in kept) * 1000),
            "mean": float(np.mean([item.tag_distance_m for item in kept]) * 1000),
        },
    }
    return camera_from_tcp, average.inliers, report

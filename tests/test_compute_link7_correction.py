"""验证 Link7 部署修正 C 的几何含义与手指码夹爪系构造。"""

import numpy as np
from scipy.spatial.transform import Rotation

from compute_link7_correction import finger_frame, link7_correction, mean_frame


def _pose(euler_deg, translation):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    pose[:3, 3] = translation
    return pose


def test_correction_places_robot_fingers_at_training_fingers():
    """策略输出 P 右乘 C 后，RM75 手指位姿等于训练时 UMI 手指位姿。"""
    T_W_Cu = _pose([150, 5, -20], [0.4, 0.1, 0.3])
    T_Cu_F = _pose([-5, 0, 0], [0.001, 0.060, 0.086])
    T_Cd_F = _pose([-20, 0, 0], [-0.0015, 0.036, 0.0845])
    hand_eye = _pose([-180, 21, 0], [0.0015, 0.0956, -0.0072])
    P = T_W_Cu @ hand_eye                      # 训练标签：假设 Cu≡Cd 的虚拟 Link7
    C = link7_correction(T_Cu_F, T_Cd_F, hand_eye, hand_eye)
    robot_fingers = P @ C @ np.linalg.inv(hand_eye) @ T_Cd_F
    np.testing.assert_allclose(robot_fingers, T_W_Cu @ T_Cu_F, atol=1e-12)
    # 两台相机相对手指安装一致时无需修正。
    np.testing.assert_allclose(link7_correction(T_Cu_F, T_Cu_F, hand_eye, hand_eye),
                               np.eye(4), atol=1e-12)


def test_correction_uses_training_and_deploy_hand_eye_separately():
    """重装相机后：标签用旧手眼，部署用新手眼，手指仍对齐。"""
    T_W_Cu = _pose([140, -3, 10], [0.3, -0.1, 0.25])
    T_Cu_F = _pose([-5, 0, 0], [0.0, 0.06, 0.086])
    T_Cd_F_new = _pose([-6, 1, 0], [0.0, 0.058, 0.085])
    old, new = _pose([-180, 21, 0], [0.0, 0.0956, -0.007]), _pose([-180, 8, 0], [0.0, 0.09, 0.0])
    C = link7_correction(T_Cu_F, T_Cd_F_new, old, new)
    robot_fingers = T_W_Cu @ old @ C @ np.linalg.inv(new) @ T_Cd_F_new
    np.testing.assert_allclose(robot_fingers, T_W_Cu @ T_Cu_F, atol=1e-12)


def test_finger_frame_is_orthonormal_midpoint_and_mean_is_stable():
    """夹爪系原点为两码中点、轴正交；同一位姿多次平均不变。"""
    left, right = _pose([-95, 2, 0], [-0.05, 0.06, 0.086]), _pose([-95, -2, 0], [0.05, 0.06, 0.086])
    frame = finger_frame(left, right)
    np.testing.assert_allclose(frame[:3, 3], [0.0, 0.06, 0.086])
    np.testing.assert_allclose(frame[:3, :3] @ frame[:3, :3].T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(frame[:3, 0], [1, 0, 0], atol=1e-12)
    mean, translation_std, rotation_rms = mean_frame([frame, frame, frame])
    np.testing.assert_allclose(mean, frame, atol=1e-12)
    assert translation_std < 1e-12 and rotation_rms < 1e-6

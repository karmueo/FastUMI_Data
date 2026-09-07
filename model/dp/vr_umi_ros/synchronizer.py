"""按图像时间同步关节与夹爪，维护 30 Hz 两帧历史及 episode 重置状态。"""

from collections import deque

import numpy as np

from vr_umi_ros.core import Observation


class ObservationBuffer:
    """无 ROS 依赖的有限长度缓存；由节点的单线程回调访问。"""

    def __init__(self, sync_slop_s=0.05, input_timeout_s=0.2, history_tolerance_s=0.015):
        """设置同步、过期和历史采样容差，单位秒；非法配置抛出 ValueError。"""
        if not all(np.isfinite(value) and value > 0 for value in
                   (sync_slop_s, input_timeout_s, history_tolerance_s)):
            raise ValueError("Synchronization timeouts must be positive and finite")
        if history_tolerance_s >= 1 / 30:
            raise ValueError("History tolerance must be less than one observation period")
        self.slop_ns = round(sync_slop_s * 1e9)  # 三路时间匹配的最大偏差。
        self.timeout_ns = round(input_timeout_s * 1e9)  # 可接受的输入年龄。
        self.history_tolerance_ns = round(history_tolerance_s * 1e9)  # 历史采样误差。
        self.episode_id = 0  # 首轮编号为零，显式重置时递增。
        self.clear()

    def clear(self):
        """清空订阅缓存和起始参考，不修改 episode 编号。"""
        self.images = deque(maxlen=128)  # 尚未匹配的时间、图像。
        self.joints = deque(maxlen=128)  # 时间、已通过 FK 的位姿。
        self.grippers = deque(maxlen=128)  # 接收时间、归一化夹爪。
        self.history = deque(maxlen=64)  # 按时间递增的同步观测。
        self.start_pose = None  # 当前 episode 的首个有效同步位姿。
        self.last_offered_ns = -1  # 防止同一图像重复提交模型。

    def reset(self):
        """开始新的 episode，清除所有旧状态并返回新编号。"""
        self.episode_id += 1
        self.clear()
        return self.episode_id

    def add_image(self, stamp_ns, image):
        """缓存独立图像数组；重排在同步阶段完成。"""
        self.images.append((stamp_ns, image))

    def add_pose(self, stamp_ns, pose):
        """缓存关节 FK 得到的绝对位姿。"""
        self.joints.append((stamp_ns, pose))

    def add_gripper(self, stamp_ns, openness):
        """按接收时刻缓存有效归一化夹爪，非法值抛出 ValueError。"""
        if not np.isfinite(openness) or not 0 <= openness <= 1:
            raise ValueError("Gripper state must be finite and within [0,1]")
        self.grippers.append((stamp_ns, float(openness)))

    def _nearest(self, entries, stamp_ns, now_ns):
        """选择有效且时间差最小的缓存项，没有符合条件的数据时返回 None。"""
        valid = [entry for entry in entries if 0 <= now_ns - entry[0] <= self.timeout_ns
                 and abs(entry[0] - stamp_ns) <= self.slop_ns]
        return min(valid, key=lambda entry: abs(entry[0] - stamp_ns)) if valid else None

    def poll(self, now_ns):
        """更新同步历史并返回最新的两帧窗口；缺流、过期或历史不足时返回 None。"""
        pending = deque(maxlen=128)  # 暂时缺少状态、仍可等待的图像。
        for stamp_ns, image in sorted(self.images, key=lambda entry: entry[0]):
            if now_ns - stamp_ns > self.timeout_ns:
                continue
            if stamp_ns > now_ns:
                pending.append((stamp_ns, image))
                continue
            if self.history and stamp_ns <= self.history[-1].stamp_ns:
                continue
            pose = self._nearest(self.joints, stamp_ns, now_ns)
            gripper = self._nearest(self.grippers, stamp_ns, now_ns)
            if pose is None or gripper is None:
                pending.append((stamp_ns, image))
                continue
            observation = Observation(stamp_ns, image, pose[1].copy(), gripper[1])
            self.history.append(observation)
            if self.start_pose is None:
                self.start_pose = observation.pose.copy()
        self.images = pending
        if len(self.history) < 2:
            return None
        latest = self.history[-1]
        if latest.stamp_ns <= self.last_offered_ns or now_ns - latest.stamp_ns > self.timeout_ns:
            return None
        target_ns = latest.stamp_ns - round(1e9 / 30)
        previous = min(list(self.history)[:-1], key=lambda item: abs(item.stamp_ns - target_ns))
        if (abs(previous.stamp_ns - target_ns) > self.history_tolerance_ns
                or now_ns - previous.stamp_ns > self.timeout_ns):
            return None
        self.last_offered_ns = latest.stamp_ns
        return previous, latest

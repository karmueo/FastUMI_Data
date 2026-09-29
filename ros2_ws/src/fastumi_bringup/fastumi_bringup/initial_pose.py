"""RM75 初始关节目标，按 joint1 至 joint7 排列，单位为弧度。"""

import math


INITIAL_JOINT_POSITIONS = tuple(
    math.radians(degrees) for degrees in (0, 20, 0, 70, 0, 90, 90)
)

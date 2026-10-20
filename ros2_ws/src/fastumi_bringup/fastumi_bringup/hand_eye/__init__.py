"""RM75 眼在手上手眼标定采集用的纯算法模块（不依赖 ROS，可单测）。

坐标约定：^A T_B 为 4x4 齐次矩阵，把 B 系中的点变换到 A 系；平移单位米，角度弧度。
- base：RM75 base_link；flange：Link7（URDF 法兰，不含夹爪 TCP）。
- cam：腕部鱼眼相机光学系（x 右、y 下、z 沿光轴向前）。
- board：OpenCV ChArUco 板系，原点在板左上角，x 沿列、y 沿行，板面 z=0。
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预览摄像头，在按下空格时保存截图并读取机械臂位姿。

依赖支持图形窗口的 OpenCV、NumPy、SciPy 和睿尔曼 Robotic-Arm SDK，
不依赖 ROS。导入模块可使用 RobotArmController；只有执行 main() 才会
创建输出目录、连接机械臂、打开摄像头和预览窗口。

启动方式（在项目根目录执行，解释器须已安装上述依赖）：
    python3 capture_arm_cam.py -c 0 -o ./dataset/captures
    使用项目已有的 uv 环境：
    uv run --project model/dp --no-sync python capture_arm_cam.py \
        -c 0 -o ./dataset/captures

启动参数（均可选，无位置参数）：
    -h, --help：显示帮助并退出，不连接硬件。
    -c, --camera：整数摄像头设备号，默认 0。
    -o, --output：截图及位姿文件的输出目录，默认 ./captures；
        相对路径以运行时的工作目录为基准。
    -w, --width：请求的图像宽度，单位像素，默认 1920。
    -ht, --height：请求的图像高度，单位像素，默认 1080。
    --robot-ip：机械臂 IP 地址，默认 192.168.1.18。
    --robot-port：机械臂连接端口，默认 8080。
内部固定设置：请求帧率为 30 fps，预览窗口为 960×540，JPEG 质量为 95；
    这些值不是启动参数，实际分辨率和帧率由摄像头支持情况决定。

输入：摄像头视频流、通过 SDK 网络连接读取的机械臂状态，以及预览窗口
    中的按键；空格保存，预览窗口或终端中的 ESC 退出，Ctrl+C 也可退出。
    终端处于交互模式时临时读取标准输入，退出后恢复终端设置，不发送运动指令。
输出：目录下以同一系统时间戳命名的 .jpg 和 .txt 文件。JPEG 保存相机
    原始帧，不包含预览叠加文字；TXT 的表头为 timestamp_ns x y z qx qy qz qw，数据行以
    空格分隔，位置单位为 m，四元数顺序为 (qx, qy, qz, qw)，数值保留
    六位小数。坐标参考沿用 SDK 返回的位姿，脚本不设置或转换参考坐标系。
    时间戳是触发保存时的系统 Unix 时间（ns），不是相机曝光或机械臂
    采样时间；图像先保存，随后查询位姿，两者没有硬件同步保证。
    位姿获取失败时已保存的图像仍保留，可能没有对应的有效 TXT 文件。
    运行状态和捕获到的错误打印到标准输出，参数解析错误写入标准错误。
"""

import argparse
from contextlib import contextmanager
import os
import select
import sys
import termios
import time
import tty

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import Robotic_Arm.rm_robot_interface as rm_robot_interface
from Robotic_Arm.rm_robot_interface import *

# ==================== 命令行参数的默认值 ====================
# SDK 网络连接的默认地址和端口，可由 --robot-ip、--robot-port 覆盖。
ROBOT_IP = "192.168.1.18"
ROBOT_PORT = 8080
# OpenCV 摄像头设备索引；在 Linux 下日志按 /dev/video{索引} 显示。
DEFAULT_CAMERA_INDEX = 0
# 相对运行目录的默认输出路径，可由 --output 覆盖。
DEFAULT_OUTPUT_DIR = "./captures"
# 请求的帧尺寸（像素），摄像头可能采用不同的实际分辨率。
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# ==================== 机械臂控制 ====================
class RobotArmController:
    """管理 SDK 连接并读取位姿，由调用方显式断开连接。"""

    def __init__(self, ip, port):
        """初始化 SDK 并立即建立机械臂网络连接。

        Args:
            ip: 机械臂控制器的 IP 地址。
            port: 机械臂控制器的网络端口号。

        Raises:
            RuntimeError: SDK 返回的连接句柄 id 不大于 0。

        连接成功后向标准输出打印状态；本方法不发送运动指令。
        """
        # SDK 对象使用三线程模式，连接生命周期由本控制器管理。
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        # 保存连接句柄，兼容 SDK 中连接接口的两种拼写。
        if hasattr(self.arm, 'rm_create_robot_arm'):
            self.handle = self.arm.rm_create_robot_arm(ip, port)
        else:
            self.handle = self.arm.rm_createrobot_arm(ip, port)

        if self.handle.id <= 0:
            raise RuntimeError("机械臂连接失败")
        print("✅ 机械臂连接成功")

    def get_pose(self):
        """通过网络读取当前位姿，并将欧拉角转换为四元数。

        沿用 SDK 返回的坐标参考；按 SciPy 的外禀 xyz 顺序解释欧拉角，
        不进行相机坐标系变换，也不设置机械臂的工作或工具坐标系。

        Returns:
            七元素元组 (x, y, z, qx, qy, qz, qw)。位置单位为 m，
            四元数无量纲，标量分量 qw 位于末尾。

        Raises:
            RuntimeError: SDK 状态码非 0 或返回状态中缺少 pose 字段。
            ValueError: 位姿元素无法转为浮点数，或欧拉角形状不符合要求。
        """
        # res 为 SDK 状态码（0 表示成功），state 为机械臂当前状态字典。
        res, state = self.arm.rm_get_current_arm_state()
        if res != 0 or 'pose' not in state:
            raise RuntimeError("无法读取位姿")

        # SDK 的六维 pose 顺序为 [x, y, z, rx, ry, rz]，单位为 m 和 rad。
        pose = [float(x) for x in state['pose'][:6]]
        pos = np.array(pose[:3])  # 形状 (3,) 的位置数组，沿用 SDK 坐标参考。
        euler = np.array(pose[3:6])  # 形状 (3,) 的欧拉角数组，单位为 rad。

        rot = R.from_euler('xyz', euler, degrees=False)
        quat = rot.as_quat()  # 形状 (4,)，顺序为 [qx, qy, qz, qw]。

        return (pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3])

    def disconnect(self):
        """请求 SDK 删除当前连接句柄，并向标准输出打印断开提示。

        SDK 删除接口的状态码未被检查，提示文字不代表删除操作一定成功。
        """
        self.arm.rm_delete_robot_arm()
        print("🔌 机械臂已断开")

@contextmanager
def terminal_keyboard():
    """临时启用终端逐键读取，并在退出或异常时恢复原设置。

    非交互标准输入不读取；保留 Ctrl+C 的信号行为。
    """
    fd = None
    settings = None
    try:
        if sys.stdin.isatty():
            fd = sys.stdin.fileno()
            settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        yield fd
    finally:
        if settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, settings)


def terminal_exit_requested(fd):
    """非阻塞检测终端 ESC，忽略方向键等以 ESC 开头的转义序列。"""
    if fd is None or not select.select([fd], [], [], 0)[0]:
        return False
    keys = os.read(fd, 1024)
    if keys != b"\x1b":
        return False
    # 方向键的后续字节可能稍晚到达；单独 ESC 无需回车。
    if select.select([fd], [], [], 0.03)[0]:
        os.read(fd, 1024)
        return False
    return True


def main():
    """解析命令行参数，运行摄像头预览和按键触发的图像、位姿保存。

    创建输出目录并先连接机械臂，再打开摄像头与图形窗口。空格触发
    一次保存；位姿获取或 TXT 写入失败时打印错误并继续预览。ESC 或
    读帧失败、关闭窗口或 Ctrl+C 时结束循环，随后释放摄像头、关闭
    窗口、断开机械臂并销毁 SDK 后台线程。终端 ESC 无需回车。

    Raises:
        SystemExit: --help 时以 0 退出，参数解析错误时以 2 退出；
            机械臂连接或摄像头打开失败时以 1 退出。
        OSError: 输出目录创建失败。
        cv2.error: 未捕获的摄像头、图形窗口或图像操作错误。

    finally 清理资源；脚本独占 SDK，退出时 rm_destroy 会关闭所有 SDK 连接。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='摄像头截图 + 机械臂位姿记录',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python3 capture_arm_cam.py -c 0 -o ./data
  python3 capture_arm_cam.py --camera 1 --output /home/user/photos
  python3 capture_arm_cam.py -c 2 -o ./captures --width 1920 --height 1080
        '''
    )
    parser.add_argument('-c', '--camera', type=int, default=DEFAULT_CAMERA_INDEX,
                        help=f'摄像头设备号 (默认: {DEFAULT_CAMERA_INDEX})')
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('-w', '--width', type=int, default=FRAME_WIDTH,
                        help=f'图像宽度 (默认: {FRAME_WIDTH})')
    parser.add_argument('-ht', '--height', type=int, default=FRAME_HEIGHT,
                        help=f'图像高度 (默认: {FRAME_HEIGHT})')
    parser.add_argument('--robot-ip', type=str, default=ROBOT_IP,
                        help=f'机械臂 IP (默认: {ROBOT_IP})')
    parser.add_argument('--robot-port', type=int, default=ROBOT_PORT,
                        help=f'机械臂端口 (默认: {ROBOT_PORT})')

    args = parser.parse_args()

    # 将命令行值用于本次运行，不修改模块级默认配置。
    camera_index = args.camera
    output_dir = args.output
    frame_width = args.width
    frame_height = args.height
    robot_ip = args.robot_ip
    robot_port = args.robot_port

    # 创建或复用输出目录；相对路径以当前工作目录为基准。
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")

    # 连接机械臂
    try:
        robot = RobotArmController(robot_ip, robot_port)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    cap = None
    # 空格触发次数，不等同于成功保存的图像、位姿对数。
    capture_count = 0
    try:
        # 打开摄像头
        print(f"📹 正在打开摄像头: /dev/video{camera_index}")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头 /dev/video{camera_index}")
            print("   请检查: 摄像头是否连接? 是否有权限?")
            sys.exit(1)

        # 请求帧尺寸及 30 fps；设备可以拒绝或调整这些设置。
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # 从设备读取实际帧尺寸（像素）和帧率（fps），用于日志显示。
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"✅ 摄像头已打开: /dev/video{camera_index}")
        print(f"📹 实际分辨率: {actual_width}x{actual_height} @ {actual_fps:.2f}fps")

        # 创建可缩放的图形预览窗口，需要 OpenCV GUI 支持和桌面显示环境。
        window_name = f"Camera /dev/video{camera_index} (SPACE=截图, ESC=退出)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

        print("\n" + "="*60)
        print("📷 摄像头预览")
        print(f"   设备: /dev/video{camera_index}")
        print(f"   分辨率: {actual_width}x{actual_height}")
        print(f"   输出目录: {output_dir}")
        print("   [SPACE] → 截图并记录位姿")
        print("   [ESC]   → 退出程序（预览窗口或终端均可，无需回车）")
        print("   [Ctrl+C] 或关闭预览窗口 → 退出程序")
        print("="*60 + "\n")

        with terminal_keyboard() as terminal_fd:
            while True:
                if terminal_exit_requested(terminal_fd):
                    print("🛑 退出")
                    break

                # ret 表示读帧是否成功；彩色 frame 通常为 (高度, 宽度, 3) 的 BGR 图像。
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ 无法读取帧")
                    break

                # 仅在预览副本上叠加文字，保留原始 frame 用于保存。
                preview_frame = frame.copy()
                info = f"Device: /dev/video{camera_index} | Captures: {capture_count} | SPACE=save, ESC=quit"
                cv2.putText(preview_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow(window_name, preview_frame)

                # 处理窗口事件并读取按键低 8 位；窗口获得焦点后响应空格和 ESC。
                key = cv2.waitKey(1) & 0xFF

                if key == 27 or terminal_exit_requested(terminal_fd):  # ESC
                    print("🛑 退出")
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("🛑 预览窗口已关闭")
                    break
                if key == 32:  # SPACE
                    # 系统 Unix 时间（ns）作为文件名及 TXT 时间戳，并非设备采样时间。
                    timestamp_ns = time.time_ns()
                    filename_base = f"{timestamp_ns}"
                    capture_count += 1

                    # 同一次触发的图像和位姿共用文件名主干。
                    jpg_path = os.path.join(output_dir, f"{filename_base}.jpg")
                    txt_path = os.path.join(output_dir, f"{filename_base}.txt")

                    # 以质量 95 保存 JPEG；此处未检查 imwrite 返回的成功标志。
                    cv2.imwrite(jpg_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"📸 [{capture_count}] 图像已保存: {jpg_path} ({frame.shape[1]}x{frame.shape[0]})")

                    # 图像保存后才查询位姿；获取或写入失败时保留图像并继续预览。
                    try:
                        x, y, z, qx, qy, qz, qw = robot.get_pose()
                        with open(txt_path, 'w') as f:
                            # 表头和数据字段顺序一致；位置为 m，四元数标量分量在末尾。
                            f.write("timestamp_ns x y z qx qy qz qw\n")
                            f.write(f"{timestamp_ns} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
                        print(f"📄 [{capture_count}] 位姿已保存: {txt_path}")
                    except Exception as e:
                        print(f"❌ 获取位姿失败: {e}")

                    print()

    except KeyboardInterrupt:
        print("\n🛑 退出")
    finally:
        # 即使某项清理失败，也继续释放其余资源和 SDK 的全局线程。
        try:
            if cap is not None:
                cap.release()
        finally:
            try:
                cv2.destroyAllWindows()
            finally:
                try:
                    robot.disconnect()
                finally:
                    robot.arm.rm_destroy()
    print(f"\n👋 程序结束，共拍摄 {capture_count} 张")
    print(f"📁 文件保存在: {output_dir}")

if __name__ == "__main__":
    main()

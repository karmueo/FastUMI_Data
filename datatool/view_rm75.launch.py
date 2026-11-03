"""离线预览当前 RM75 和夹爪模型，只启动显示及关节状态发布节点。"""
from pathlib import Path
import xml.etree.ElementTree as ET

# 单侧预览行程，单位米；按 CAD 接触面间隙取约 20.4 mm，并非硬件标定。
FINGER_TRAVEL_M = 0.0204


def enable_gripper_preview(robot):
    """启用沿 X 轴对称的刚性平移近似，q=0 闭合，q=行程还原 CAD 姿态。

    只修改内存中的 URDF，柔性手指不模拟受力变形；右指跟随左指。
    """
    for side, direction in (("left", 1), ("right", -1)):
        joint = robot.find(f"joint[@name='gripper_{side}_finger_joint']")
        if joint is None or joint.get("type") != "fixed":
            raise ValueError(f"缺少预期的固定夹指关节：{side}")
        joint.set("type", "prismatic")
        joint.find("origin").set("xyz", f"{-direction * FINGER_TRAVEL_M} 0 0")
        ET.SubElement(joint, "axis", xyz=f"{direction} 0 0")
        # effort 和 velocity 仅用于完整描述预览关节，不是硬件控制参数。
        ET.SubElement(joint, "limit", lower="0", upper=str(FINGER_TRAVEL_M),
                      effort="1", velocity="0.02")
        if side == "right":
            ET.SubElement(joint, "mimic", joint="gripper_left_finger_joint",
                          multiplier="1", offset="0")


def generate_launch_description():
    """解析网格资源，启动模型发布器、关节滑块和 RViz。"""
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch_ros.actions import Node

    root_dir = Path(__file__).resolve().parents[1]
    urdf_path = root_dir / 'assets' / 'rm_75.urdf'
    robot = ET.parse(urdf_path).getroot()
    for mesh in robot.iter('mesh'):
        filename = mesh.attrib['filename']
        if filename.startswith('package://'):
            package, relative = filename[len('package://'):].split('/', 1)
            mesh_path = Path(get_package_share_directory(package)) / relative
        elif '://' not in filename:
            mesh_path = urdf_path.parent / filename
        else:
            continue
        if not mesh_path.is_file():
            raise FileNotFoundError(f'模型网格不存在：{mesh_path}')
        mesh.set('filename', mesh_path.resolve().as_uri())
    enable_gripper_preview(robot)
    description = ET.tostring(robot, encoding='unicode')
    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': description}],
            output='screen',
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            parameters=[{'zeros.gripper_left_finger_joint': FINGER_TRAVEL_M}],
            output='screen',
        ),
        Node(
            package='rviz2', executable='rviz2',
            arguments=['-d', str(root_dir / 'assets' / 'rm75_preview.rviz')],
            output='screen',
        ),
    ])

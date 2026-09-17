"""用 ROS 2 launch 启动 DP 推理节点并仅覆盖显式指定的参数。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


OVERRIDE_NAMES = (  # 可从 launch 命令行覆盖的字符串型 ROS 参数。
    "checkpoint", "urdf_path", "device", "expected_urdf_sha256",
    "image_topic", "joint_topic", "gripper_topic", "output_topic", "reset_service",
    "postprocessors",
)


def _create_node(context):
    """解析启动参数，忽略空覆盖并生成唯一的推理节点。"""
    config_file = LaunchConfiguration("config_file").perform(context)  # 已解析的参数 YAML 路径。
    overrides = {  # 只传递用户显式设置的值，保留 YAML 中其余配置。
        name: value
        for name in OVERRIDE_NAMES
        if (value := LaunchConfiguration(name).perform(context))
    }
    parameters = [config_file]  # YAML 优先提供全部参数。
    if overrides:
        parameters.append(overrides)
    return [Node(
        package="dp_infer", executable="dp_infer_node", name="dp_infer",
        output="screen", parameters=parameters,
    )]


def generate_launch_description():
    """声明默认 YAML 及可选覆盖，并运行 DP 推理节点。"""
    default_config = PathJoinSubstitution(  # 安装后包资源中的参数文件。
        [FindPackageShare("dp_infer"), "config", "dp_infer.yaml"]
    )
    arguments = [  # 所有参数使用空字符串标记“未覆盖”。
        DeclareLaunchArgument("config_file", default_value=default_config),
        *[DeclareLaunchArgument(name, default_value="") for name in OVERRIDE_NAMES],
    ]
    return LaunchDescription([*arguments, OpaqueFunction(function=_create_node)])

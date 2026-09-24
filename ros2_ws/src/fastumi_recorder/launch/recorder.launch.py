"""独立启动录制服务；不启动设备，不自动开始录制。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """提供本地存储、任务和主要采集参数。"""
    defaults = dict(dataset_root='', dir_name='test', name='default_test',
                    record_camera='true',
                    image_topic='/wrist_camera/image_raw/ffmpeg',
                    image_transport='ffmpeg', image_reliability='reliable',
                    image_qos_depth='30', shutdown_save_timeout='120.0')
    parameters = {key: ParameterValue(LaunchConfiguration(key), value_type=str)
                  for key in ('dataset_root', 'dir_name', 'name', 'image_topic',
                              'image_transport', 'image_reliability')}
    parameters.update(
        record_camera=ParameterValue(LaunchConfiguration('record_camera'), value_type=bool),
        image_qos_depth=ParameterValue(LaunchConfiguration('image_qos_depth'), value_type=int),
        shutdown_save_timeout=ParameterValue(LaunchConfiguration('shutdown_save_timeout'), value_type=float),
    )
    return LaunchDescription([
        *[DeclareLaunchArgument(key, default_value=value) for key, value in defaults.items()],
        Node(package='fastumi_recorder', executable='recorder', output='screen',
             parameters=[parameters],
             sigterm_timeout=PythonExpression([LaunchConfiguration('shutdown_save_timeout'), ' + 10.0']),
             sigkill_timeout='5'),
    ])

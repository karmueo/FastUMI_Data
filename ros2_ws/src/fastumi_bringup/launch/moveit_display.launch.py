"""在 RViz2 中通过 MoveIt2 插件只读显示 RM75 实际状态。

与 hardware.launch.py 并行运行：关节状态来自 rm_driver 发布的 /joint_states
（joint1..joint7，弧度），模型、SRDF 与运动学配置复用子模块 rm_75_config。
move_group 关闭轨迹执行且不加载控制器，MotionPlanning 面板只能规划与预览，
不会向机械臂下发任何命令。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


# 只读显示：禁止轨迹执行，也不让 move_group 管理任何控制器。
MOVE_GROUP_DISPLAY_PARAMETERS = {
    'publish_robot_description_semantic': True,
    'allow_trajectory_execution': False,
    'moveit_manage_controllers': False,
    # 发布实际机械臂的规划场景，供 RViz 插件显示当前状态。
    'publish_planning_scene': True,
    'publish_geometry_updates': True,
    'publish_state_updates': True,
    'publish_transforms_updates': True,
    'monitor_dynamics': False,
}


def generate_launch_description():
    """启动 robot_state_publisher、只读 move_group 与 MoveIt RViz 界面。"""
    moveit_config = MoveItConfigsBuilder(
        'rm_75_description', package_name='rm_75_config').to_moveit_configs()
    # Humble 的 move_group 总会创建轨迹执行管理器，缺少插件会崩溃；
    # 保留 simple controller manager 插件但不声明任何控制器，确保没有执行通路。
    moveit_config.trajectory_execution = {
        'moveit_controller_manager':
            'moveit_simple_controller_manager/MoveItSimpleControllerManager',
    }

    return LaunchDescription([
        DeclareLaunchArgument('start_rsp', default_value='true',
                              description='发布 robot_description 与 TF；其它入口已发布时设为 false'),
        DeclareLaunchArgument('start_move_group', default_value='true',
                              description='启动只读 move_group；false 时只有模型显示'),
        DeclareLaunchArgument('rviz_config',
                              default_value=str(moveit_config.package_path / 'config/moveit.rviz')),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[moveit_config.robot_description],
            condition=IfCondition(LaunchConfiguration('start_rsp')),
        ),
        Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[moveit_config.to_dict(), MOVE_GROUP_DISPLAY_PARAMETERS],
            condition=IfCondition(LaunchConfiguration('start_move_group')),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            output='log',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.planning_pipelines,
                moveit_config.robot_description_kinematics,
            ],
        ),
    ])

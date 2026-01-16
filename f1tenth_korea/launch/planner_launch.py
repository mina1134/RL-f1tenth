from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('f1tenth_korea'),
        'config',
        'planner_params.yaml'
        )
    config_dict = yaml.safe_load(open(config, 'r'))

    planner_node = Node(
        package='f1tenth_korea',
        executable='f1tenth_kor_planner',
        name='planner',
        parameters=[config]
    )

    # finalize
    ld.add_action(planner_node)

    return ld

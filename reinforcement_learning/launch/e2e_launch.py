from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('reinforcement_learning'),
        'config',
        'env.yaml'
        )
    config_dict = yaml.safe_load(open(config, 'r'))['ros__parameters']
    use_sim = config_dict['use_sim']
    train = config_dict['train']
    logging = config_dict['logging']

    e2e_node = Node(
        package='reinforcement_learning',
        executable='e2e',
        name='e2e_component',
        parameters=[config_dict]
    )
    logger_node = Node(
        package='planner',
        executable='vehicle_state_logger',
        name='vehicle_state_logger',
        parameters=[config_dict]
    )

    # finalize
    ld.add_action(e2e_node)
    
    if logging:
        ld.add_action(logger_node)

    return ld

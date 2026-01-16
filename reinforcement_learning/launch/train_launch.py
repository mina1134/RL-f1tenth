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
    if config_dict['launch_param']['use_sim']:
        exec_config = config_dict['sim_param']
    else:
        exec_config = config_dict['real_param']
        
    train = config_dict['launch_param']['train']
    logging = config_dict['launch_param']['logging']

    e2e_node = Node(
        package='reinforcement_learning',
        executable='e2e',
        name='e2e_component',
        parameters=[config_dict]
    )
    lookahead_planner = Node(
        package='reinforcement_learning',
        executable='lookahead_planner',
        name='lookahead_planner',
        parameters=[config_dict] 
    )
    control_node = Node(
        package='reinforcement_learning',
        executable='control_node',
        name='control_node',
        parameters=[config_dict] 
    )
    obs_control_node = Node(
        package='reinforcement_learning',
        executable='obs_avoid_controller',
        name='obs_avoid_controller',
        parameters=[config_dict] 
    )
    logger_node = Node(
        package='planner',
        executable='vehicle_state_logger',
        name='vehicle_state_logger',
        parameters=[config_dict]
    )

    # finalize
    # ld.add_action(e2e_node)
    # ld.add_action(lookahead_planner)
    # ld.add_action(obs_control_node)
    ld.add_action(control_node)
    if logging:
        ld.add_action(logger_node)

    return ld

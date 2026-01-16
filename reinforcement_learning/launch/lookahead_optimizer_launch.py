from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    ld = LaunchDescription()
    launch_config = os.path.join(
        get_package_share_directory('reinforcement_learning'),
        'config',
        'env.yaml'
        )
    env_config = yaml.safe_load(open(launch_config, 'r'))['ros__parameters']
        
    logging = env_config['logging']
    package_name = 'reinforcement_learning'

    if env_config['use_sim']:
        env_params = env_config['sim_param']
    else:
        env_params = env_config['real_param']
        
    exec_config = os.path.join(
        get_package_share_directory('reinforcement_learning'),
        'config',
        'lookahead_optimizer.yaml'
        )
    exec_params = yaml.safe_load(open(exec_config, 'r'))['ros__parameters']
	
    lookahead_planner = Node(
        package=package_name,
        executable='lookahead_planner',
        name='lookahead_planner',
        parameters=[env_params, exec_params] 
    )
    obs_control_node = Node(
        package=package_name,
        executable='obs_avoid_controller',
        name='obs_avoid_controller',
        parameters=[env_params, exec_params] 
    )
    logger_node = Node(
        package='planner',
        executable='vehicle_state_logger',
        name='vehicle_state_logger',
        parameters=[env_params, env_config['logger_param']]
    )

    ld.add_action(lookahead_planner)
    # ld.add_action(obs_control_node)

    if logging:
        ld.add_action(logger_node)

    return ld

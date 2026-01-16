from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'reinforcement_learning'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'waypoints'), glob('waypoints/*.csv')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'log'), glob('log/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mina',
    maintainer_email='mina@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'e2e=reinforcement_learning.e2e_component:main',
        	'control_node=reinforcement_learning.controller:main',
        	'ref=reinforcement_learning.mina:main',
        	'test=reinforcement_learning.mina_model_test:main',
        	'lookahead_planner=reinforcement_learning.lookahead_planner:main',
        	'obs_avoid_planner=reinforcement_learning.obstacle_avoid:main',
        	'obs_avoid_controller=reinforcement_learning.obs_controller:main',
        	'f1tenth_kor_planner=reinforcement_learning.f1tenth_korea:main'
        ],
    },
)

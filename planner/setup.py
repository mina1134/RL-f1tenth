from setuptools import setup
from glob import glob

package_name = 'planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
        'rrt_planner=planner.rrt:main',
        'rrt_star_planner=planner.rrt_star:main',
        'purepursuit=planner.purepursuit:main',
        'mina_algorithm=planner.mina:main',
        'obs_planner=planner.obs_planner:main',
        'vehicle_state_logger=planner.state_logger:main',
        'fgm=planner.reactive_node:main'
        ],
    },
)

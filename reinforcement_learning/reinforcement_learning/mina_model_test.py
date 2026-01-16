import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.optim import Adam
from reinforcement_learning import PolicyNet, QNet, Mina
from collections import namedtuple, deque
import numpy as np
import random
from transforms3d import euler
import os

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from interfaces.msg import Goal, Train

class Test_Env(Node):
    def __init__(self):
        super().__init__('test_env')

        self.actor = PolicyNet()
        self.actor.load_state_dict(torch.load(os.getcwd() + '/actor.pt'))

        self.pose_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 1)
        self.forward_sub = self.create_subscription(Goal, '/goal_point', self.forward, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.reinit_pub = self.create_publisher(Train, '/train', 10)

        self.prev_x = None
        self.prev_y = None
        self.prev_theta = None
        self.prev_state = None
        self.action = [0,0]
        self.fric = 0.0
        self.collision = False
        self.warning = False
        self.eps = 0
        self.velocity = 0.0
        

    def odom_callback(self, odom_msg):
        self.velocity = odom_msg.twist.twist.linear.x
        

    def calc_goal_currnt_vec(self, x, y, heading, goal_x, goal_y):
        rotation_matrix = np.array([[np.cos(heading), -np.sin(heading)],
                                    [np.sin(heading), np.cos(heading)]])

        goal = (np.array([goal_x, goal_y]) - np.array([x, y])).T
        goal = goal@rotation_matrix

        vector_norm = np.linalg.norm(goal)
        orientation = np.arctan2(goal[1], goal[0])
        
        return vector_norm, orientation

    def calc_diff_prev_desired(self, x, y, heading, prev_x, prev_y, prev_theta, prev_action):
        accel = prev_action[0]
        acted_theta = prev_theta + prev_action[1]
        acted_theta = acted_theta % (2*np.pi)
        if acted_theta >= np.pi/2 and acted_theta <= 3*np.pi/2:
            acted_orientation = -np.array([1, np.tan(acted_theta)]) / np.linalg.norm(np.array([1, np.tan(acted_theta)]))
        else:
            acted_orientation = np.array([1, np.tan(acted_theta)]) / np.linalg.norm(np.array([1, np.tan(acted_theta)]))
        goal = np.array([prev_x, prev_y]) + accel * acted_orientation
        pos_diff = np.array([x, y]) - np.array([prev_x, prev_y])

        if x == prev_x and y == prev_y:
            return 0

        cos = abs(acted_theta - heading) % (2*np.pi)
        if cos == 0:
            cos = 1 
        proj = np.linalg.norm(pos_diff) * np.cos(cos)

        return proj


    # def calc_diff_prev_desired(self, velocity, desired_speed):
    #     self.get_logger().info(f'{velocity}, {desired_speed}')
    #     return desired_speed - velocity
    
    def pose_callback(self, pose_msg):
        ## update position
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y
        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, 
                pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        theta = euler.quat2euler(quat, axes='sxyz')[0]
        if theta < 0:                   # 0 <= theat <= 2pi
            theta = 2*np.pi + theta

        if self.prev_x == None:
            self.prev_x = x
            self.prev_y = y
            self.prev_theta = theta
        else:
            self.prev_x = self.pose_x
            self.prev_y = self.pose_y
            self.prev_theta = self.pose_theta
            
        self.pose_x = x
        self.pose_y = y
        self.pose_theta = theta

    def check_collision(self, scan_msg):
        scans = np.array(scan_msg.ranges)

        if np.any(scans < 0.3):
            self.collision = True
        elif np.any(scans < 0.6):
            self.collision = False
            self.warning = True
        else:
            self.collision = False
            self.warning = False
        
            
    def reset(self):
        self.prev_x = None
        self.prev_y = None
        self.prev_theta = None
        self.prev_state = None
        self.action = [0,0]

        self.fric = 0.0
        self.collision = False

        return 

    def forward(self, goal_msg):
        ## get state
        norm, ori = self.calc_goal_currnt_vec(self.pose_x, self.pose_y, self.pose_theta, goal_msg.goal_point.x, goal_msg.goal_point.y)
        self.fric = self.calc_diff_prev_desired(self.pose_x, self.pose_y, self.pose_theta, self.prev_x, self.prev_y, self.prev_theta, self.action)
        state = [norm, ori, self.fric]

        action = self.actor(torch.tensor(state).to(torch.float32))

        speed = float(action[0])
        steer = float(action[1])
                
        self.action = [speed, steer]

        if self.collision:
            reinit_msg = Train()
            reinit_msg.init = True
            self.reinit_pub.publish(reinit_msg)
            return self.reset()
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steer
        self.drive_pub.publish(drive_msg)

        self.prev_state = state
        self.prev_goal_x = goal_msg.goal_point.x
        self.prev_goal_y = goal_msg.goal_point.y


def main(args=None):
    rclpy.init(args=args)
    env = Test_Env()
    planner_node = Mina()

    executor = MultiThreadedExecutor()

    executor.add_node(env)
    executor.add_node(planner_node)

    try:
        executor.spin()
    finally:
        executor.remove_node(env)
        executor.remove_node(planner_node)
        rclpy.shutdown()

if __name__ == '__main__':
    main()
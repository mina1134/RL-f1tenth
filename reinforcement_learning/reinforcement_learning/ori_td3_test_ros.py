import torch
import torch.nn as nn
import os
import numpy as np
from argparse import Namespace
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class Env(Node):
    def __init__(self):
        super().__init__('env')

        param_file = os.getcwd() + '/src/reinforcement_learning/config/EndToEnd.yaml'
        with open(param_file, 'r') as file:
            self.planner_params = Namespace(**yaml.load(file, Loader=yaml.FullLoader))
        self.skip_n = int(1080 / self.planner_params.number_of_beams)
        self.state_space = self.planner_params.number_of_beams *2 + 1 
        self.scan_buffer = np.zeros((self.planner_params.n_scans, self.planner_params.number_of_beams))
        self.actor = torch.load(os.getcwd() + '/src/reinforcement_learning/results/EndToEnd_actor.pt')

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.velocity = 0.0
        self.max_steer = 0.4
    

    def odom_callback(self, odom_msg):
        self.velocity = odom_msg.twist.twist.linear.x

    def transform_obs(self, scan):
        velocity = self.velocity / self.planner_params.max_speed
        scan = np.array(scan)
        scan = np.clip(scan[::self.skip_n] / self.planner_params.range_finder_scale, 0, 1)

        if self.scan_buffer.all() == 0:
            for i in range(self.scan_buffer.shape[0]):
                self.scan_buffer[i, :] = scan
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        dual_scan = np.reshape(self.scan_buffer, (-1))
        nn_obs = np.concatenate((dual_scan, [velocity]))

        return nn_obs


    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        speed = (nn_action[1] + 1) * (self.planner_params.max_speed / 2 - 0.5) + 1
        speed = min(speed, self.planner_params.max_speed)

        action = np.array([steering_angle, speed])

        return action
    
    def scan_callback(self, scan_msg):
        nn_state = self.transform_obs(scan_msg.ranges)
        nn_act = self.actor(torch.FloatTensor(nn_state).cuda()).cpu().data.numpy().flatten()
        action = self.transform_action(nn_act)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(action[1])
        drive_msg.drive.steering_angle = float(action[0])
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    env = Env()
    print("Start")
    rclpy.spin(env)

    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

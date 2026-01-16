#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from interfaces.msg import Action

import csv
import numpy as np
import os, time
from transforms3d import euler

class Logger(Node):
    def __init__(self):
        super().__init__('vehice_state_logger')

        self.declare_parameter('count_lap', True)
        self.count_lap = self.get_parameter('count_lap').value

        if self.count_lap:
            self.declare_parameter('package_name')
            self.declare_parameter('waypoint_dir')
            self.declare_parameter('waypoint_file')
            self.declare_parameter('delimiter', ',')
            self.declare_parameter('x', 0)
            self.declare_parameter('y', 1)
            self.declare_parameter('psi', 2)
            self.declare_parameter('use_sim')

            waypoint_dir = self.get_parameter('waypoint_dir').value
            waypoint_file = self.get_parameter('waypoint_file').value
            delimiter = self.get_parameter('delimiter').value
            package_name = self.get_parameter('package_name').value
            self.x = self.get_parameter('x').value
            self.y = self.get_parameter('y').value
            self.psi = self.get_parameter('psi').value

            waypoints_path = os.path.join('src', package_name, waypoint_dir, waypoint_file)
            waypoints = np.loadtxt(waypoints_path, delimiter=delimiter)
            self.path = waypoints[ : ,self.x:self.x+2]
            self.initial_idx = None
        else:
            self.make_logfile('Log.csv')

        self.sim = self.get_parameter('use_sim').value
        self.pose_topic = '/ego_racecar/odom' if self.sim else '/pf/pose/odom'
        self.odom_topic = '/ego_racecar/odom' if self.sim else '/odom'
        self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.ld_sub = self.create_subscription(Action, '/lookahead_distance', self.get_lookahead_distance, 1)
        # self.timer = self.create_timer(0.04, self.timer_callback)

        self.step = 0
        self.lap_count = 1
        self.clock = time.time()
        self.time_step = 0
        self.velocity = 0.0
        self.lookahead_distance = 0.0

    def make_logfile(self, file_name):
        file_name = file_name
        self.f = open(file_name, 'w')
        self.writer = csv.writer(self.f)
        self.writer.writerow(['# pose_x', 'pose_y', 'velocity'])

    def pose_callback(self, pose_msg):
        pose_x = pose_msg.pose.pose.position.x
        pose_y = pose_msg.pose.pose.position.y

        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, 
                pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        pose_theta = euler.quat2euler(quat, axes='sxyz')[0]
        if pose_theta < 0:    # 0 < heading < 2pi
            pose_theta = 2*np.pi + pose_theta

        if self.count_lap:
            pose = np.array([pose_x, pose_y])
            dists = np.linalg.norm(self.path - pose, axis=1)
            nearest_idx = np.argmin(dists)
            self.time_step = time.time() - self.clock

            if self.step == 0:
                self.initial_idx = nearest_idx
                self.make_logfile(f'Log_Lap{self.lap_count}.csv')
                self.clock = time.time()
                self.time_step = 0
            elif self.step >= 50 and nearest_idx==self.initial_idx:
                self.lap_count += 1
                self.step = 0
                return self.f.close()
            
        self.writer.writerow([pose_x, pose_y, pose_theta, self.velocity, self.lookahead_distance, self.time_step])
        self.step += 1

    def odom_callback(self, odom_msg):
        self.velocity = odom_msg.twist.twist.linear.x
            
    def get_lookahead_distance(self, ld_msg):
        self.lookahead_distance = ld_msg.lookahead_distance

def main(args=None):
    rclpy.init(args=args)
    logger = Logger()
    print("Start")
    rclpy.spin(logger)

    logger.destroy_node()
    rclpy.shutdown()

    logger.f.close()

if __name__=='__main__':
    main()

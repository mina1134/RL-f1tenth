#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class SafetyNode(Node):

    def __init__(self):
        super().__init__('aeb_node')
        self.speed = 0.  # v_x
        self.desired_speed = 2.0
        self.thres = 1.5
        self.ackermann_msg = AckermannDriveStamped()

        # create ROS subscribers and publishers.
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # parameters
        self.angle_min = -20
        self.angle_max = 20
        self.scan_min_index = 539 + int(self.angle_min * 4)
        self.scan_max_index = 539 + int(self.angle_max * 4)

    def odom_callback(self, odom_msg):
        # update current speed
        self.speed = odom_msg.twist.twist.linear.x


    def scan_callback(self, scan_msg):

        # calculate TTC
        r = np.array(scan_msg.ranges)[self.scan_min_index:self.scan_max_index]
        theta = np.linspace(self.angle_min, self.angle_max, self.scan_max_index - self.scan_min_index)
        r_dot = self.speed * np.cos(theta)  # v_x projection & range rate are different in definition, but numerically equivalent
        ttc = r / np.clip(r_dot, a_min=0.001, a_max=None)  # 0.001 reduces inf & nan
        min_ttc = np.min(np.clip(ttc, 0.0, 60.0))  # clip ittc between 0 ~ 60s

        # publish command to brake
        if (self.speed > 0 and min_ttc < self.thres) or (self.speed < 0 and min_ttc < (self.thres + 0.8)):  # reversing should consider car length & blind spots
            print('min_ttc is {}, brake!!!!'.format(round(min_ttc, 2)))
            self.desired_speed = 0.0
            
        self.ackermann_msg.drive.speed = self.desired_speed
        self.pub_drive.publish(self.ackermann_msg)


def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

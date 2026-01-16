import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA

import numpy as np
import os
from pathlib import Path

class Lane_visualize(Node):
    def __init__(self):
        super().__init__('raceline_visualize_node')

        self.lane_pub = self.create_publisher(Marker, '/raceline', 1)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.publish, 1)
        
        csv_dir = os.path.join(Path(__file__).resolve().parents[1], 'maps/')
        csv_name = '/home/mina/sim_ws/src/reinforcement_learning/waypoints/test_optimal.csv'
        # self.waypoints = np.loadtxt(csv_dir+csv_name, delimiter=',')
        self.waypoints = np.loadtxt(csv_name, delimiter=',')
        
        self.column_x = 0
        self.column_y = 1
        self.column_v = 4

        self.marker = Marker()
        self.marker.header.frame_id = 'map'
        self.marker.type = 4
        self.marker.action = Marker.ADD
        self.marker.color.r = 0.0
        self.marker.color.g = 0.0
        self.marker.color.b = 1.0
        self.marker.color.a = 1.0
        self.marker.scale.x = 0.1
        self.marker.scale.y = 0.1
        self.marker.scale.z = 0.1
        self.marker.id = 0
        self.marker.points = []
        self.marker.colors = []

        # self.marker.points = [Point(x=pt[self.column_x], y=pt[self.column_y], z=0.0) for pt in self.waypoints]
        for pt in range(self.waypoints.shape[0]):
            point = Point()
            point.x = self.waypoints[pt, self.column_x]
            point.y = self.waypoints[pt, self.column_y]
            self.marker.points.append(point)

            # v = self.waypoints[pt, self.column_v] / (8.5 + 1e-6)
            # color = ColorRGBA()
            # color.a = 1.0
            # if v < 0.25:
            #     # Blue → Cyan
            #     t = v / 0.25
            #     color.r = 0.0
            #     color.g = t
            #     color.b = 1.0
            # elif v < 0.5:
            #     # Cyan → Green
            #     t = (v - 0.25) / 0.25
            #     color.r = 0.0
            #     color.g = 1.0
            #     color.b = 1.0 - t
            # elif v < 0.75:
            #     # Green → Yellow
            #     t = (v - 0.5) / 0.25
            #     color.r = t
            #     color.g = 1.0
            #     color.b = 0.0
            # else:
            #     # Yellow → Red
            #     t = (v - 0.75) / 0.25
            #     color.r = 1.0
            #     color.g = 1.0 - t
            #     color.b = 0.0

            # self.marker.colors.append(color)


    def publish(self, odom_msg):
        self.lane_pub.publish(self.marker)

if __name__ == '__main__':
    rclpy.init(args=None)
    lane_visualize_node = Lane_visualize()
    rclpy.spin(lane_visualize_node)
    lane_visualize_node.destroy_node()
    rclpy.shutdown()

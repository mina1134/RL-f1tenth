import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from transforms3d import euler
from visualization_msgs.msg import Marker

import numpy as np
import os

class PurePursuit(Node):
    def __init__(self):
        super().__init__('purepursuit_planner')
        
        self.declare_parameter('pose_topic', '/ego_racecar/odom')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('wheelbase', 0.3302)
        self.declare_parameter('wpt_path', '/home/mina/sim_ws/src/planner/maps/test_optimal_ld.csv')
        self.declare_parameter('wpt_delim', ',')
        self.declare_parameter('wpt_rowskip', 0)
        self.declare_parameter('car_width', 0.33)
        self.declare_parameter('sx', 18.586)
        self.declare_parameter('sy', 0.564)
        self.declare_parameter('stheta', 2.685)
        
        self.pose_topic = self.get_parameter('pose_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.load_waypoints(self.get_parameter('wpt_path').value, self.get_parameter('wpt_delim').value, self.get_parameter('wpt_rowskip').value)
        self.car_width = self.get_parameter('car_width').value
        self.pose = np.array([self.get_parameter('sx').value, self.get_parameter('sy').value])
        self.pose_theta = self.get_parameter('stheta').value

        
        ## hyper parameters ##
        self.vgain = 1.0
        
        # Subscriber
        self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 1)
        
        # Publisher
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.visualize_pub = self.create_publisher(Marker, '/waypoint', 1)
        
        self.velocity = 0.0
        

    def load_waypoints(self, path, delim, rowskip):
        self.waypoints = np.loadtxt(path, delimiter=delim, skiprows=rowskip)
         
    def get_actuation(self, pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
        # Extract the Waypoint information
        waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
        speed = 4.0
        if np.abs(waypoint_y) < 1e-6:
            return 3.0, 0.
        # Define the radius of the arc to follow
        radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)

        # Calculate the steering angle based on the curvature of the arc to follow
        steering_angle = np.arctan(wheelbase / radius)

        return speed, steering_angle
    

    def get_waypoint(self, pose, waypoints, L):
        diff = np.linalg.norm(pose - waypoints, axis=1)
        current_idx = np.argmin(diff)

        in_range = np.where(diff <= L)[0]
        if np.max(in_range) == waypoints.shape[0]-1 or current_idx == waypoints.shape[0]-1:
            i = 0
            while (i in list(in_range)):
                i+=1
            return i+1
        return np.max(in_range)+1   
         
    def odom_callback(self, odom_msg):
        self.velocity = odom_msg.twist.twist.linear.x
        
    def pose_callback(self, pose_msg):
        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, 
                pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        self.pose_theta = euler.quat2euler(quat, axes='sxyz')[0]
        self.pose = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])

        waypoints = self.waypoints[:, 0:2]

        # lookahead_distance = 0.8 + (self.velocity/8) * 4
        # lookahead_distance = min(max(0.5, 2.0*self.velocity/6.0),2.0)
        
        diff = np.linalg.norm(self.pose - waypoints, axis=1)
        current_idx = np.argmin(diff)
        
        lookahead_distance = self.waypoints[current_idx, 5]
        lookahead_idx = self.get_waypoint(self.pose, waypoints, lookahead_distance)
        lookahead_point = waypoints[lookahead_idx,:]
        
        if lookahead_point is None:
            return 0.0, 0.0

        true_lookahead_distance = np.linalg.norm(lookahead_point - self.pose)
        speed, steer = self.get_actuation(self.pose_theta, lookahead_point, self.pose, true_lookahead_distance, self.wheelbase)
        speed = self.waypoints[current_idx, 4] * self.vgain

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed          = speed
        drive_msg.drive.steering_angle = steer
        self.drive_pub.publish(drive_msg)
        
        self.visualize(lookahead_point)

    def visualize(self, lookahead_point):
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.id = 0
        marker.ns = "target_waypoints"
        marker.type = 1
        marker.action = 0
        marker.pose.position.x = lookahead_point[0]
        marker.pose.position.y = lookahead_point[1]

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0     

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.lifetime.nanosec = int(1e8)

        self.visualize_pub.publish(marker)
        

def main(args=None):
    rclpy.init(args=args)
    planner = PurePursuit()
    rclpy.spin(planner)

    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import numpy as np
import math
import os
from typing import Union
import scipy.spatial
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from multiprocessing import shared_memory
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from collections import deque

from sensor_msgs.msg import LaserScan
from transforms3d import euler

WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)

Mina = False
Avoid = False
Optimal_speed = 0.0

def safe_changeIdx(length, inp, plus):
    return (inp + plus + length) % (length)

class LaneFollow(Node):

    def __init__(self):
        super().__init__("lane_follow_node")

        # ROS Params
        self.declare_parameter("visualize")

        self.declare_parameter("lane_occupied_dist") 
        self.declare_parameter("obs_activate_dist")

        self.declare_parameter("real_test")
        self.declare_parameter("map_name")
        self.declare_parameter("num_lanes")
        self.declare_parameter("lane_files")

        self.declare_parameter("lookahead_distance") 
        self.declare_parameter("lookahead_idx")

        self.declare_parameter("max_steer")
        self.declare_parameter("follow_speed")

        # interp
        self.declare_parameter('minL')
        self.declare_parameter('maxL')
        self.declare_parameter('minP')
        self.declare_parameter('maxP')
        self.declare_parameter('interpScale')
        self.declare_parameter('Pscale')
        self.declare_parameter('Lscale')
        self.declare_parameter('D')
        self.declare_parameter('vel_scale')

        self.declare_parameter('minL_corner')
        self.declare_parameter('maxL_corner')
        self.declare_parameter('minP_corner') 
        self.declare_parameter('maxP_corner')
        self.declare_parameter('Pscale_corner') 
        self.declare_parameter('Lscale_corner') 

        self.declare_parameter('avoid_L_scale')
        self.declare_parameter('pred_v_buffer')
        self.declare_parameter('avoid_buffer')
        self.declare_parameter('avoid_span')

        # PID Control Params
        self.prev_steer_error = 0.0
        self.steer_integral = 0.0
        self.prev_steer = 0.0
        self.prev_ditem = 0.0

        # Global Map Params
        self.real_test = self.get_parameter("real_test").get_parameter_value().bool_value
        self.map_name = self.get_parameter("map_name").get_parameter_value().string_value

        # Lanes Waypoints
        self.num_lanes = self.get_parameter("num_lanes").get_parameter_value().integer_value
        self.lane_files = self.get_parameter("lane_files").get_parameter_value().string_array_value

        self.num_lane_pts = []
        self.lane_x = []
        self.lane_y = []
        self.lane_v = []
        self.lane_pos = []
        
        assert len(self.lane_files) == self.num_lanes

        for i in range(self.num_lanes):
            lane_csv_loc = os.path.join("src", "shinhwa", "csv", self.map_name, self.lane_files[i] + ".csv")
            lane_data = np.loadtxt(lane_csv_loc, delimiter=",")
            
            self.num_lane_pts.append(len(lane_data))
            self.lane_x.append(lane_data[:, 0])
            self.lane_y.append(lane_data[:, 1])
            self.lane_v.append(lane_data[:, 2])
            self.lane_pos.append(np.vstack((self.lane_x[-1], self.lane_y[-1]), ).T)
        
        #### Only Pure Pursuit ####
        self.traj_x = self.lane_x[-1][:]
        self.traj_y = self.lane_y[-1][:]
        self.traj_v = self.lane_v[-1][:]
        self.traj_pos = self.lane_pos[-1][:]
        #### Only Pure Pursuit ####
    
        #### mina ####
        #overtaking_idx_csv_loc = os.path.join("src", "shinhwa", "csv", self.map_name, 'overtaking_wp_idx.npy') # src/lane_follow/csv/map_name/overtaking_wp_idx.npy
        #data = np.load(overtaking_idx_csv_loc, mmap_mode = 'r')
        data = list(range(1, 2))
        self.overtake_wpIdx = list(range(1, 2))
        self.overtake_wpIdx = set(self.overtake_wpIdx) 
        
        #slow_idx_csv_loc = os.path.join("src", "shinhwa", "csv", self.map_name, 'slowdown_wp_idx.npy') # src/lane_follow/csv/map_name/slowdown_wp_idx.npy
        #data2 = np.load(slow_idx_csv_loc, mmap_mode='r')
        #self.slow_wpIdx = set(list(data2))

        data2 = []
        for i in range(len(self.lane_x[0])):
            if (i not in list(data)):
                data2.append(i)
        self.slow_wpIdx = set(data2)
        
        data3 = []
        for i in range(len(self.lane_x[0])):
            if (i not in list(data)) and (i not in list(data2)):
                data3.append(i)
        self.corner_wpIdx = set(data3)
        
        #### mina ####
        # Car Status Variables
        self.lane_idx = 0
        self.curr_idx = None
        self.goal_idx = None
        self.curr_vel = 0.0
        self.target_point = None 

        # Obstacle Variables
        self.obstacles = None
        self.opponent = np.array([np.inf, np.inf])
        self.lane_free = [True] * self.num_lanes
        self.declare_parameter('avoid_dist')
        self.opponent_v = 0.0
        self.opponent_last = np.array([0.0, 0.0])
        self.opponent_timestamp = 0.0
        self.pred_v_buffer = self.get_parameter('pred_v_buffer').get_parameter_value().integer_value
        self.pred_v_counter = 0
        self.avoid_buffer = self.get_parameter('avoid_buffer').get_parameter_value().integer_value
        #self.avoid_counter = 0
        self.detect_oppo = False
        self.avoid_L_scale = self.get_parameter('avoid_L_scale').get_parameter_value().double_value
        self.last_lane = -1
        #### Only Lane Follow ####

        # Topics & Subs, Pubs
        pose_topic = "/pf/viz/inferred_pose" if self.real_test else "/ego_racecar/odom"
        odom_topic = "/odom" if self.real_test else "/ego_racecar/odom"
        obstacle_topic = "/opp_predict/bbox"
        opponent_topic = "/opp_predict/state"
        drive_topic = "/lane_follow/drive"
        waypoint_topic = "/waypoint"
                
        if self.real_test:
            self.pose_sub_ = self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 1)

        else:
            self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, 1)
        self.odom_sub_ = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        self.obstacle_sub_ = self.create_subscription(PoseArray, obstacle_topic, self.obstacle_callback, 1)
        self.opponent_sub_ = self.create_subscription(PoseStamped, opponent_topic, self.opponent_callback, 1)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.waypoint_pub_ = self.create_publisher(Marker, waypoint_topic, 10)

        print('node_init_files')

    def odom_callback(self, odom_msg: Odometry): 
        self.curr_vel = odom_msg.twist.twist.linear.x

    def obstacle_callback(self, obstacle_msg: PoseArray):
        obstacle_list = []
        for obstacle in obstacle_msg.poses:
            x = obstacle.position.x
            y = obstacle.position.y
            obstacle_list.append([x, y])
        self.obstacles = np.array(obstacle_list) if obstacle_list else None 
                
        if self.obstacles is None:
            self.lane_free = np.array([True] * self.num_lanes)
            return

        lane_occupied_dist = self.get_parameter("lane_occupied_dist").get_parameter_value().double_value

        for i in range(self.num_lanes):
            d = scipy.spatial.distance.cdist(self.lane_pos[i], self.obstacles)
            self.lane_free[i] = (np.min(d) > lane_occupied_dist)

    def opponent_callback(self, opponent_msg: PoseStamped):
        opponent_x = opponent_msg.pose.position.x
        opponent_y = opponent_msg.pose.position.y
        self.opponent = np.array([opponent_x, opponent_y])

        # velocity
        if not np.any(np.isinf(self.opponent)):
            if self.detect_oppo:
                oppoent_dist_diff = np.linalg.norm(self.opponent - self.opponent_last)
                if self.pred_v_counter == 4:
                    self.pred_v_counter = 0
                    cur_time = opponent_msg.header.stamp.nanosec/1e9 + opponent_msg.header.stamp.sec
                    time_interval = cur_time - self.opponent_timestamp
                    self.opponent_timestamp = cur_time
                    opponent_v = oppoent_dist_diff / max(time_interval, 0.005)
                    self.opponent_last = self.opponent.copy()
                    self.opponent_v = opponent_v

                else:
                    self.pred_v_counter += 1
            else: 
                self.detect_oppo = True 
                self.opponent_last = self.opponent.copy()
        else:
            self.detect_oppo = False

    def find_interp_point(self, L, begin, target):
        interpScale = self.get_parameter('interpScale').get_parameter_value().integer_value
        x_array = np.linspace(begin[0], target[0], interpScale) # x
        y_array = np.linspace(begin[1], target[1], interpScale) # y
        xy_interp = np.vstack([x_array, y_array]).T # [x, y]
        dist_interp = np.linalg.norm(xy_interp-target, axis=1) - L
        i_interp = np.argmin(np.abs(dist_interp))
        interp_point = np.array([x_array[i_interp], y_array[i_interp]])
        return interp_point

    def pose_callback(self, pose_msg: Union[PoseStamped, Odometry]):
        global Mina, Optimal_speed
       
        cur_speed = self.curr_vel

        #### Read pose data #### 
        if self.real_test: 
            curr_x = pose_msg.pose.position.x
            curr_y = pose_msg.pose.position.y
            curr_pos = np.array([curr_x, curr_y])
            curr_quat = pose_msg.pose.orientation

        else:
            curr_x = pose_msg.pose.pose.position.x
            curr_y = pose_msg.pose.pose.position.y
            curr_pos = np.array([curr_x, curr_y])
            curr_quat = pose_msg.pose.pose.orientation
        
        curr_yaw = math.atan2(2 * (curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
                              1 - 2 * (curr_quat.y ** 2 + curr_quat.z ** 2))

        curr_pos_idx = np.argmin(np.linalg.norm(self.lane_pos[-1][:, :2] - curr_pos, axis=1))


        #### mina ####
        curr_lane_nearest_idx = np.argmin(np.linalg.norm(self.lane_pos[self.last_lane][:, :2] - curr_pos, axis=1))

        traj_distances = np.linalg.norm(self.lane_pos[self.last_lane][:, :2] - self.lane_pos[self.last_lane][curr_lane_nearest_idx, :2], axis=1) 

        segment_end = np.argmin(traj_distances)
        num_lane_pts = len(self.lane_pos[self.last_lane])

        sp = self.traj_v[curr_lane_nearest_idx]
        Optimal_speed = sp

        if curr_pos_idx in self.corner_wpIdx:
            L = self.get_L_w_speed(sp, corner=True)
        else:
            L = self.get_L_w_speed(sp)
        #### mina ####

        interpScale = self.get_parameter('interpScale').get_parameter_value().integer_value

        while traj_distances[segment_end] <= L: 
            segment_end = (segment_end + 1) % num_lane_pts

        segment_begin = (segment_end - 1 + num_lane_pts) % num_lane_pts

        x_array = np.linspace(self.lane_x[self.last_lane][segment_begin], self.lane_x[self.last_lane][segment_end], interpScale)
        y_array = np.linspace(self.lane_y[self.last_lane][segment_begin], self.lane_y[self.last_lane][segment_end], interpScale)
        v_array = np.linspace(self.lane_v[self.last_lane][segment_begin], self.lane_v[self.last_lane][segment_end], interpScale)

        xy_interp = np.vstack([x_array, y_array]).T
        dist_interp = np.linalg.norm(xy_interp-curr_pos, axis=1) - L
        i_interp = np.argmin(np.abs(dist_interp))

        #### Only Lane Follow ####
        target_global = np.array([x_array[i_interp], y_array[i_interp]])

        if self.last_lane == -1:
            target_v = v_array[i_interp]
        else:
            target_v = self.lane_v[-1][curr_pos_idx] * 0.25
        
        self.target_point = np.array([x_array[i_interp], y_array[i_interp]])
        #### interp for finding target ####

        # Choose new target point from the closest lane if obstacle exists
        avoid_dist = self.get_parameter('avoid_dist').get_parameter_value().double_value
        avoid_span = self.get_parameter('avoid_span').get_parameter_value().double_value

        if np.any(np.isinf(self.opponent)):
            if Mina == True:
                Mina = False

        R = np.array([[np.cos(curr_yaw), np.sin(curr_yaw)],
                      [-np.sin(curr_yaw), np.cos(curr_yaw)]])
        
        target_x, target_y = R @ np.array([self.target_point[0] - curr_x,
                                           self.target_point[1] - curr_y])

        vel_scale = self.get_parameter('vel_scale').get_parameter_value().double_value

        speed = target_v * vel_scale

        L = np.linalg.norm(curr_pos - self.target_point)
        gamma = 2 / L ** 2
        error = gamma * target_y

        if curr_pos_idx in self.corner_wpIdx:
            steer = self.get_steer_w_speed(cur_speed, error, corner=True)
            speed = speed
        else:
            steer = self.get_steer_w_speed(cur_speed, error)
            
        # Publish drive message
        message = AckermannDriveStamped()
        message.drive.speed = speed
        message.drive.steering_angle = steer
        self.drive_pub_.publish(message)

        # Visualize waypoints
        visualize = self.get_parameter("visualize").get_parameter_value().bool_value
        if visualize:
            self.visualize_target()

        return None

    def visualize_target(self):
        # Publish target waypoint
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.id = 0 
        marker.ns = "target_waypoint"
        marker.type = 1
        marker.action = 0
        marker.pose.position.x = self.target_point[0]
        marker.pose.position.y = self.target_point[1]

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        this_scale = 0.2
        marker.scale.x = this_scale
        marker.scale.y = this_scale
        marker.scale.z = this_scale

        marker.pose.orientation.w = 1.0

        marker.lifetime.nanosec = int(1e8)

        self.waypoint_pub_.publish(marker) # marker publish

    def get_L_w_speed(self, speed, corner=False):
        if corner:
            maxL = self.get_parameter('maxL_corner').get_parameter_value().double_value
            minL = self.get_parameter('minL_corner').get_parameter_value().double_value
            Lscale = self.get_parameter('Lscale_corner').get_parameter_value().double_value
        else:
            maxL = self.get_parameter('maxL').get_parameter_value().double_value
            minL = self.get_parameter('minL').get_parameter_value().double_value
            Lscale = self.get_parameter('Lscale').get_parameter_value().double_value
        interp_L_scale = (maxL-minL) / Lscale

        return interp_L_scale * speed + minL

    def get_steer_w_speed(self, speed, error, corner=False):
        if corner:
            maxP = self.get_parameter('maxP_corner').get_parameter_value().double_value
            minP = self.get_parameter('minP_corner').get_parameter_value().double_value
            Pscale = self.get_parameter('Pscale_corner').get_parameter_value().double_value
        else:
            maxP = self.get_parameter('maxP').get_parameter_value().double_value
            minP = self.get_parameter('minP').get_parameter_value().double_value
            Pscale = self.get_parameter('Pscale').get_parameter_value().double_value

        interp_P_scale = (maxP-minP) / Pscale
        cur_P = maxP - speed * interp_P_scale
        max_control = self.get_parameter("max_steer").get_parameter_value().double_value
        kd = self.get_parameter('D').get_parameter_value().double_value

        d_error = error - self.prev_steer_error

        if not self.real_test:
            if d_error == 0:
                d_error = self.prev_ditem
            else:
                self.prev_ditem = d_error
                self.prev_steer_error = error
        else:
            self.prev_ditem = d_error
            self.prev_steer_error = error
        if corner:
            steer = cur_P * error
        else:
            steer = cur_P * error + kd * d_error
        new_steer = np.clip(steer, -max_control, max_control)
        return new_steer
    
class mina(Node):

    # lidar preprocessing parameters
    PREPROCESS_CONV_SIZE = 3
    MAX_LIDAR_DIST = 3000000

    # Planning Parameter
    SAFE_DIST = 3  # Adjust when obstacle

    def __init__(self):
        super().__init__('f1tenth_kor_planner')
        
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('wheelbase', 0.3302)
        self.declare_parameter('car_width', 0.31)
        self.declare_parameter('sx', -3.01)
        self.declare_parameter('sy', 6.63)
        self.declare_parameter('stheta', 3.16)
        self.declare_parameter('lookahead_ind', 4)
        self.declare_parameter("map_name")
        self.declare_parameter("maxL")
        self.declare_parameter("minL")
        self.declare_parameter("Lscale")
        self.declare_parameter("lane_files")

        self.map_name = self.get_parameter("map_name").get_parameter_value().string_value
        self.wpt_path = os.path.join("src", "shinhwa", "csv", self.map_name, 'centerline.csv')
        self.wpt_delim = ','
        self.wpt_rowskip = 0

        self.scan_topic = self.get_parameter('scan_topic').value
        self.pose_topic = self.get_parameter('odom_topic').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.load_waypoints(self.wpt_path, self.wpt_delim, self.wpt_rowskip)
        self.car_width = self.get_parameter('car_width').value
        self.angle_offset = -np.pi/4
        self.pose = np.array([self.get_parameter('sx').value, self.get_parameter('sy').value])
        self.pose_theta = self.get_parameter('stheta').value
        self.points = []
        self.lookahead_ind = self.get_parameter('lookahead_ind').value
        self.lookahead_distance = 0.8
        self.lookahead_idx = None
        self.current_speed = 0
        self.obs = []

        self.maxL = self.get_parameter('maxL').get_parameter_value().double_value 
        # self.maxL = 1.5
        # self.minL = self.get_parameter('minL').get_parameter_value().double_value
        self.minL = 0.8
        self.Lscale = self.get_parameter('Lscale').get_parameter_value().double_value

        ## hyper parameters ##
        self.wpts_interval = 0.10 # check
        self.vgain = 0.8
        self.boundary = self.car_width
        self.overtake_wpIdx = list(range(80, 116))
        
        # Subscriber
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 1)
        self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_callback, 1)
        
        # Publisher
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/mina/drive', 10)
        self.waypoint_pub = self.create_publisher(MarkerArray, '/mina/points', 10)


    def load_waypoints(self, path, delim, rowskip):
        self.waypoints = np.loadtxt(path, delimiter=delim, skiprows=rowskip)

    def preprocess_lidar(self, ranges, scan_range):
        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges)
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges
            
    def get_steer(self, position, pose_theta, lookahead_point, lookahead_distance, wheelbase):
        # Extract the Waypoint information
        waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
        if np.abs(waypoint_y) < 1e-6:
            return 0.
        # Define the radius of the arc to follow
        radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)
        # Calculate the steering angle based on the curvature of the arc to follow
        steering_angle = np.arctan(wheelbase / radius)

        return steering_angle
    
    def global_to_local(self, coordinate, angle, origin):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        coordinate = (coordinate - origin).T
        
        return (rotation_matrix@coordinate).T

    def calc_norm_vec(self, waypoints, waypoint_id):
        wpts = np.vstack((waypoints, waypoints))

        v = wpts[waypoint_id+2] - wpts[waypoint_id]

        if v[1] == 0:
            norm_vec = np.array((0,1))
        else:
            norm_vec = np.array((1,-v[0]/v[1]))
        unit_vec = norm_vec/np.linalg.norm(norm_vec)
        return unit_vec
    
    def get_L(self, speed):
        # L = self.stacked_waypoints[current_idx, self.lookahead_ind] * 1.0
        interp_L_scale = (self.maxL-self.minL) / self.Lscale
        L = interp_L_scale * speed + self.minL

        return L

    def get_waypoint(self, pose, waypoints, L):
        stacked_waypoints = np.vstack((waypoints, waypoints))
        
        diff = np.linalg.norm(pose - stacked_waypoints, axis=1)
        current_idx = np.argmin(diff)

        round = (stacked_waypoints[:, 0] - pose[0])**2 + (stacked_waypoints[:, 1] - pose[1])**2
        in_range = np.where(round < (L)**2)[0]
        in_range = in_range[np.where(in_range - current_idx < 30)[0]]

        if not len(in_range) == 0:
            wpt_pos = stacked_waypoints[np.max(in_range)]
        else:
            wpt_pos = stacked_waypoints[current_idx + 5]

        diff = np.linalg.norm(wpt_pos - waypoints, axis=1)
        return np.argmin(diff)

    def pose_callback(self, pose_msg):
        global Mina, Optimal_speed

        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, 
                pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        self.pose_theta = euler.quat2euler(quat, axes='sxyz')[0]
        self.pose = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])

        self.current_speed = pose_msg.twist.twist.linear.x

        centerline = self.waypoints[:, :2]

        diff = np.linalg.norm(self.pose - centerline, axis=1)
        current_idx = np.argmin(diff)

        if Mina:
            self.lookahead_distance = self.get_L(Optimal_speed)
        else:
            if current_idx in self.overtake_wpIdx:
                self.lookahead_distance = 4.0
            else:
                 self.lookahead_distance = self.get_L(Optimal_speed)
                
        self.lookahead_distance = 4.0
        self.lookahead_idx = self.get_waypoint(self.pose, centerline, self.lookahead_distance)
        
        if self.lookahead_idx == None:
            return

        lookahead_point = centerline[self.lookahead_idx, :]
        track_bound_l = self.waypoints[self.lookahead_idx, 2]
        track_bound_r = self.waypoints[self.lookahead_idx, 3]

        norm_vec = self.calc_norm_vec(centerline, self.lookahead_idx)
        
        self.points = []
        # speed = self.waypoints[self.lookahead_idx, 5] * self.vgain
        speed = 0.0
        
        if (self.lookahead_idx == centerline.shape[0]-2) or (self.lookahead_idx == centerline.shape[0]-1):
            wp_vec = centerline[0] - centerline[self.lookahead_idx]
        else:
            wp_vec = centerline[self.lookahead_idx+2] - centerline[self.lookahead_idx]
            
        if (wp_vec[0] == 0.0):
            if wp_vec[1] >= 0:
                theta = np.pi/2
            else:  
                theta = 3*np.pi/2
        else:              
            theta = np.arctan(wp_vec[1]/wp_vec[0])

        if Mina:
            self.boundary = self.car_width/4
        else:
            self.boundary = self.car_width/4

        if speed < 4.0:
            if theta < np.pi:
                self.points.append(lookahead_point)
                i=self.wpts_interval
                while(i <= track_bound_l - self.boundary):
                    self.points.append(-i*norm_vec + lookahead_point)
                    i += self.wpts_interval
                self.points.reverse()
                i = self.wpts_interval
                while(i <= track_bound_r - self.boundary):
                    self.points.append( i*norm_vec + lookahead_point)
                    i += self.wpts_interval
            elif theta > np.pi:
                self.points.append(lookahead_point)
                i=self.wpts_interval
                while(i <= track_bound_l - self.boundary):
                    self.points.append( i*norm_vec + lookahead_point)
                    i += self.wpts_interval
                self.points.reverse()
                i = self.wpts_interval
                while(i <= track_bound_r - self.boundary):
                    self.points.append(-i*norm_vec + lookahead_point)
                    i += self.wpts_interval
        else:
            if theta < np.pi:
                self.points.append(lookahead_point)
                i=self.wpts_interval
                while(i <= track_bound_l - self.boundary):
                    self.points.append(-i*norm_vec + lookahead_point)
                    i += self.wpts_interval
                self.points.reverse()
                i = self.wpts_interval
                while(i <= track_bound_r - self.boundary):
                    self.points.append( i*norm_vec + lookahead_point)
                    i += self.wpts_interval
            elif theta > np.pi:
                self.points.append(lookahead_point)
                i=self.wpts_interval
                while(i <= track_bound_l - self.boundary):
                    self.points.append( i*norm_vec + lookahead_point)
                    i += self.wpts_interval
                self.points.reverse()
                i = self.wpts_interval
                while(i <= track_bound_r - self.boundary):
                    self.points.append(-i*norm_vec + lookahead_point)
                    i += self.wpts_interval

        if len(self.points) == 0:
            self.points.append(lookahead_point)
                    
    def scan_callback(self, scan_msg):
        global Mina, Avoid

        scan_range = len(scan_msg.ranges)
        scan = self.preprocess_lidar(scan_msg.ranges, scan_range)
        
        if len(self.points) == 0 or self.lookahead_idx == None:
            speed = 1.0
            steer = 0.0
        else:
            if self.pose_theta > np.pi / 2 and self.pose_theta < 3* np.pi / 2:
                # lidar_position = self.pose - 0.27 * np.array([1, np.tan(self.pose_theta)] / np.linalg.norm(np.array([1, np.tan(self.pose_theta)])))
                lidar_position = self.pose
            else:
                # lidar_position = self.pose + 0.27 * np.array([1, np.tan(self.pose_theta)] / np.linalg.norm(np.array([1, np.tan(self.pose_theta)])))
                lidar_position = self.pose
            
            points_on_local = self.global_to_local(np.array(self.points), (np.pi/2 - self.pose_theta), lidar_position)
                
            m = points_on_local[:,1]/points_on_local[:,0]
            angles = np.arctan(m) + np.heaviside(-m, 1) * np.pi - self.angle_offset
            point_indices = angles // scan_msg.angle_increment
            point_indices = point_indices.astype(int)
            dist2pt = np.linalg.norm(points_on_local, axis=1)

            obstacle = []
            self.obs = []
            i = 0
                
            while i < len(self.points):
                if dist2pt[i] > scan[point_indices[i]] - 0.1:
                    start = i
                    while dist2pt[i] > scan[point_indices[i]] - 0.1:
                        self.obs.append(i)
                        if i == len(self.points)-1:
                            break
                        i+=1
                    end = i
                    obstacle.append((start,end))
                i+=1

            ## points 중에서 cost가 가장 낮은 점 찾기
            
            cost = np.linalg.norm(np.array(self.points) - self.pose, axis=1)

            for obs in obstacle:
                for i in range(obs[0] - self.SAFE_DIST, obs[1] + self.SAFE_DIST+1):
                    if i >= 0 and i <= len(self.points)-1:
                        cost[i] = float('inf')
            lookahead_point = self.points[np.argmin(cost)]
            
            if np.min(cost) == float('inf') and Mina == True:
                steer = 0.0
                speed = 0.0
            elif len(obstacle) != 0 and Mina == False:
                steer = 0.
                speed = 0.
                Avoid = True
            else:
                steer = self.get_steer(self.pose, self.pose_theta, lookahead_point, np.min(cost), self.wheelbase)
                # speed = self.waypoints[self.lookahead_idx, 5] * self.vgain
                speed = 0.0

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed          = speed
        drive_msg.drive.steering_angle = steer
        self.drive_pub.publish(drive_msg)
        
        marker_arr = MarkerArray()
        for i in range(len(self.points)):
            marker = Marker()
            marker.header.frame_id = "/map"
            marker.id = i
            marker.ns = "target_waypoints"
            marker.type = 1
            marker.action = 0
            marker.pose.position.x = self.points[i][0]
            marker.pose.position.y = self.points[i][1]

            if i in self.obs:
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0

            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05

            marker.lifetime.nanosec = int(1e8)

            marker_arr.markers.append(marker)

        self.waypoint_pub.publish(marker_arr)

class Driver(Node):
    def __init__(self):
        super().__init__('Driving_node')

        self.pose_topic = '/ego_racecar/odom'
        self.lane_follow_topic = '/lane_follow/drive'
        self.mina_topic = '/mina/drive'
        self.drive_topic = '/drive'

        # Subscriber
        self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_cb, 1)
        self.lane_follow_sub = self.create_subscription(AckermannDriveStamped, self.lane_follow_topic, self.lane_follow_cb, 1)
        self.mina_sub = self.create_subscription(AckermannDriveStamped, self.mina_topic, self.mina_cb, 1)
        
        # Publisher
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.lane_follow_speed = 0.0
        self.lane_follow_steer = 0.0
        self.mina_speed = 0.0
        self.mina_steer = 0.0
    
    def pose_cb(self, pose_msg):
        global Mina, Avoid
        print(f'Mina: {Mina}, Avoid: {Avoid}')
        if Avoid:
            speed = 0.0
            steer = 0.0
        elif Mina:
            speed = self.mina_speed
            steer = self.mina_steer
        else:
            speed = self.lane_follow_speed
            steer = self.lane_follow_steer

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed          = speed
        drive_msg.drive.steering_angle = steer
        self.drive_pub.publish(drive_msg)
        if Avoid:
            # time.sleep(0.05)
            Avoid = False
            Mina = False

    def lane_follow_cb(self, drive_msg):
        self.lane_follow_speed = drive_msg.drive.speed
        self.lane_follow_steer = drive_msg.drive.steering_angle

    def mina_cb(self, drive_msg):
        self.mina_speed = drive_msg.drive.speed
        self.mina_steer = drive_msg.drive.steering_angle
            

def main(args=None):
    rclpy.init(args=args)
    print("Lane Follow Initialized")
    drive_node = Driver()
    lane_follow_node = LaneFollow()
    mina_node = mina()

    executor = MultiThreadedExecutor()
    executor.add_node(drive_node)
    executor.add_node(lane_follow_node)
    executor.add_node(mina_node)

    try:
        executor.spin()
    finally:
        executor.remove_node(drive_node)
        executor.remove_node(lane_follow_node)
        executor.remove_node(mina_node)
        rclpy.shutdown()

if __name__ == "__main__":
    main()

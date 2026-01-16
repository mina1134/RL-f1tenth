from turtle import heading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from transforms3d import euler
from visualization_msgs.msg import Marker, MarkerArray
from interfaces.msg import Train, Action

import torch
from torch.nn.functional import mse_loss
from torch.optim import Adam
import numpy as np
import os, csv
from reinforcement_learning import LDValue, SACLookaheadPlanner


batch_size = 100
memory_size = 10000
raceline_path = os.path.join('src/reinforcement_learning', 'waypoints', 'map_1104_optimal.csv')
centerline_path = os.path.join('src/reinforcement_learning/waypoints/map_1104_centerline.csv')

class Agent():
    INPUTS = 12
    KAPPA_STEP = 0.3 # [m]
    MAX_LD = 2.0
    MIN_LD = 0.5

    def __init__(self):
        self.pose = None
        self.pose_theta = None
        self.curr_idx = None
        self.curr_state = []
        self.prev_L = None
        self.min_delta_L = -0.1
        self.max_delta_L = 0.1
        self.psi = 2
        self.col_curv = 3
        self.col_vel = 4

        ## load lane
        self.raceline = np.loadtxt(raceline_path, delimiter=',')
        self.actor = SACLookaheadPlanner(input=self.INPUTS+1).cuda()
        self.actor.load_state_dict(torch.load(os.getcwd() + '/lookahead_planner_actor.pt'))
    
    def get_nearest_idx(self, pose, yaw, waypoints):
        heading_err = abs(np.arctan2(np.sin(yaw - waypoints[:, self.psi]),
                                     np.cos(yaw - waypoints[:, self.psi])))
        diffs = np.linalg.norm(pose - waypoints[:,:2], axis=1)
        in_range = np.where(heading_err < np.pi/2, diffs, float('inf'))

        nearest_idx = np.argmin(in_range)

        return nearest_idx	
    
    def get_waypoint(self, pose, waypoints, current_idx, L):	
        stacked_waypoints = np.vstack((waypoints, waypoints))

        round = (stacked_waypoints[:, 0] - pose[0])**2 + (stacked_waypoints[:, 1] - pose[1])**2
        in_range = np.where(round < (L)**2)[0]
        in_range = in_range[np.where(in_range - current_idx < 20)[0]]
        
        if not len(in_range) == 0:
            wpt_pos = stacked_waypoints[np.max(in_range)]
        else:
            wpt_pos = stacked_waypoints[current_idx + 5]

        diff = np.linalg.norm(wpt_pos - waypoints, axis=1)
        return np.argmin(diff)
    
    def get_kappa(self, idx):
        diff = np.linalg.norm(self.raceline[ :-1, :2] - self.raceline[1: , :2], axis=1)
        stacked_diff = np.hstack((diff, diff))
        curvs = np.empty((self.INPUTS,))

        for i in range(self.INPUTS):
            curv = self.raceline[idx, self.col_curv]
            cnt = stacked_diff[idx]
            while cnt <= self.KAPPA_STEP:
                cnt += stacked_diff[idx]
                idx += 1
                if idx >= len(self.raceline):
                    idx = idx - len(self.raceline)
                curv += self.raceline[idx, self.col_curv]
            curv = np.rad2deg(self.raceline[idx, self.col_curv])

            curvs[i] = np.abs(curv)
            idx += 1
            if idx >= len(self.raceline):
                idx = idx - len(self.raceline)
        
        return curvs

    def pose_callback(self, pose, pose_theta):
        self.curr_idx = self.get_nearest_idx(pose, pose_theta, self.raceline)

        # self.track_err = min(diffs)

        self.track_err = 1e-2

        self.curvs = self.get_kappa(self.curr_idx)

        self.curr_state = np.append(self.curvs, self.track_err)

        _, _, output = self.actor(torch.FloatTensor(self.curr_state).cuda())
        lookahead_distance = self.MIN_LD + (self.MAX_LD - self.MIN_LD)/2 * (float(output[0]) + 1.0)

        delta_L = 0.0
        if self.prev_L is not None:
            delta_L = lookahead_distance - self.prev_L
            delta_L = min(max(delta_L, self.min_delta_L), self.max_delta_L)
            lookahead_distance = self.prev_L + delta_L
        print('cliped lookahead_distance:', lookahead_distance)

        lookahead_idx = self.get_waypoint(pose, self.raceline[:, 0:2], self.curr_idx, lookahead_distance)
        lookahead_point = self.raceline[lookahead_idx, :2]

        self.prev_L = lookahead_distance

        return lookahead_distance, lookahead_point, self.raceline[lookahead_idx, self.col_vel]

class Planner(Node):

    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 3000000

    def __init__(self):
        super().__init__('f1tenth_kor_planner')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('pose_topic', '/ego_racecar/odom')
        self.declare_parameter('wheelbase', 0.4)
        self.declare_parameter('wpt_path', centerline_path)
        self.declare_parameter('wpt_delim', ',')
        self.declare_parameter('wpt_rowskip', 0)
        self.declare_parameter('car_width', 0.31)
        self.declare_parameter('sx', 3.64)
        self.declare_parameter('sy', 3.97)
        self.declare_parameter('stheta', 4.71)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.load_waypoints(self.get_parameter('wpt_path').value, self.get_parameter('wpt_delim').value, self.get_parameter('wpt_rowskip').value)
        self.car_width = self.get_parameter('car_width').value
        self.angle_offset = -np.pi/4
        self.pose = np.array([self.get_parameter('sx').value, self.get_parameter('sy').value])
        self.pose_theta = self.get_parameter('stheta').value
        self.points = []
        self.past_idx = 0
        self.lookahead_idx = None
        self.obs = []
        self.lp = [0, 0]
        self.train_pause = False
        self.delay_noise = 0.0

        ## hyper parameters ##
        self.wpts_interval = 0.15
        self.track_boundary = self.car_width
        self.SAFE_BOX = 3 # index

        self.psi = 2
        self.w_tr_right = 3
        self.w_tr_left = 4

        # Subscriber
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 1)
        self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_callback, 1)
        self.done_sub = self.create_subscription(Train, '/update', self.done_callback, 1)

        # Publisher
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.visualize_pub = self.create_publisher(MarkerArray, '/visualize/points', 10)
        self.agent = Agent()

    def load_waypoints(self, path, delim, rowskip):
        self.waypoints = np.loadtxt(path, delimiter=delim, skiprows=rowskip)

    def preprocess_lidar(self, ranges, scan_range):
        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges)
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    
    def global_to_local(self, coordinate, angle, origin):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        coordinate = (coordinate - origin)
        
        return (coordinate@rotation_matrix)


    def calc_norm_vec(self, waypoints, waypoint_id):
        wpts = np.vstack((waypoints, waypoints))

        v = wpts[waypoint_id+3] - wpts[waypoint_id-3]
        if v[1] == 0:
            norm_vec = np.array((0,1))
        else:
            norm_vec = np.array((1,-v[0]/v[1]))
        unit_vec = norm_vec/np.linalg.norm(norm_vec)
        return unit_vec

    def get_waypoint(self, pose, waypoints, current_idx, L):	
        stacked_waypoints = np.vstack((waypoints, waypoints))

        round = (stacked_waypoints[:, 0] - pose[0])**2 + (stacked_waypoints[:, 1] - pose[1])**2
        in_range = np.where(round < (L)**2)[0]
        in_range = in_range[np.where(in_range - current_idx < 20)[0]]
        
        if not len(in_range) == 0:
            wpt_pos = stacked_waypoints[np.max(in_range)]
        else:
            wpt_pos = stacked_waypoints[current_idx + 5]

        diff = np.linalg.norm(wpt_pos - waypoints, axis=1)
        return np.argmin(diff)
    
    def pure_pursuit(self, pose_theta, current_point, lookahead_point, L, wheelbase=0.3302):
        waypoint_vec = (lookahead_point - current_point).T
        waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), waypoint_vec)
        radius = 1 / (2.0 * waypoint_y / L**2)
        return np.arctan(wheelbase / radius)
    
    def get_nearest_idx(self, pose, yaw, waypoints):
        heading_err = abs(np.arctan2(np.sin(yaw - waypoints[:, self.psi]),
                                     np.cos(yaw - waypoints[:, self.psi])))
        diffs = np.linalg.norm(pose - waypoints[:,:2], axis=1)
        in_range = np.where(heading_err < np.pi/2, diffs, float('inf'))

        nearest_idx = np.argmin(in_range)

        return nearest_idx	
    
    def pose_callback(self, pose_msg):
        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, 
        pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        self.pose_theta = euler.quat2euler(quat, axes='sxyz')[0]
        if self.pose_theta < 0:
            self.pose_theta = 2*np.pi + self.pose_theta
        self.pose = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])

        current_idx = self.get_nearest_idx(self.pose, self.pose_theta, self.waypoints)
        self.lookahead_distance, self.optimal_lookahead_point, self.velocity_profile = self.agent.pose_callback(self.pose, self.pose_theta)
        self.lookahead_distance = self.lookahead_distance * 1.0
        
        centerline = self.waypoints[:, :2]
        self.lookahead_idx = self.get_waypoint(self.pose, centerline, current_idx, self.lookahead_distance)
        # self.lookahead_idx = self.agent.get_waypoint(self.pose, centerline, current_idx, self.lookahead_distance)
        if self.lookahead_idx == None:
            return

        center_point = centerline[self.lookahead_idx, :]
        track_bound_l = self.waypoints[self.lookahead_idx, self.w_tr_left]
        track_bound_r = self.waypoints[self.lookahead_idx, self.w_tr_right]
        norm_vec = self.calc_norm_vec(centerline, self.lookahead_idx)

        self.points = []
        
        if self.pose_theta < np.pi:
            i=0
            while(i <= track_bound_l - self.track_boundary):
                self.points.append(-i*norm_vec + center_point)
                i += self.wpts_interval
            self.points.reverse()
            i = self.wpts_interval
            while(i <= track_bound_r - self.track_boundary):
                self.points.append( i*norm_vec + center_point)
                i += self.wpts_interval
        elif self.pose_theta > np.pi:
            i=0
            while(i <= track_bound_l - self.track_boundary):
                self.points.append( i*norm_vec + center_point)
                i += self.wpts_interval
            self.points.reverse()
            i = self.wpts_interval
            while(i <= track_bound_r - self.track_boundary):
                self.points.append(-i*norm_vec + center_point)
                i += self.wpts_interval

                    
        if len(self.points) == 0:
            self.points.append(center_point)


    def scan_callback(self, scan_msg):
        scan_range = len(scan_msg.ranges)
        # scan = self.preprocess_lidar(scan_msg.ranges, scan_range)
        scan = scan_msg.ranges
        drive_msg = AckermannDriveStamped()
        
        if len(self.points) == 0 or self.lookahead_idx == None:
            drive_msg.drive.speed          = 0.0
            drive_msg.drive.steering_angle = 0.0
        else:    
            if self.pose_theta > np.pi / 2 and self.pose_theta < 3* np.pi / 2:
                lidar_position = self.pose - 0. * np.array([1, np.tan(self.pose_theta)] / np.linalg.norm(np.array([1, np.tan(self.pose_theta)])))
            else:
                lidar_position = self.pose + 0. * np.array([1, np.tan(self.pose_theta)] / np.linalg.norm(np.array([1, np.tan(self.pose_theta)])))

            points_on_local = self.global_to_local(np.array(self.points), (self.pose_theta - np.pi/2), lidar_position)

            angles = np.arctan2(points_on_local[:,1],points_on_local[:,0]) - self.angle_offset
            point_indices = angles // scan_msg.angle_increment
            point_indices = point_indices.astype(int)
            dist2pt = np.linalg.norm(points_on_local, axis=1)
            obstacle = []
            self.obs = []

            ## Compare Lidar value - Euclidean distance
            i = 0
            while i < len(self.points):
                if dist2pt[i] > scan[point_indices[i]]:
                    start = i
                    while dist2pt[i] > scan[point_indices[i]]:
                        self.obs.append(i) # for visualization
                        if i == len(self.points)-1:
                            break
                        i+=1
                    end = i
                    obstacle.append((start,end))
                i+=1

            ## 거리
            # cost = np.linalg.norm(np.array(self.points) - self.pose, axis=1)
            cost = np.linalg.norm(np.array(self.points) - self.optimal_lookahead_point, axis=1)
            ## 조향
            # cost = self.pure_pursuit(self.pose_theta, self.pose, np.array(self.points), self.lookahead_distance, wheelbase=self.wheelbase)

            ## Change obstacle cost
            for obs in obstacle:
                for i in range(obs[0]-self.SAFE_BOX, obs[1]+self.SAFE_BOX):
                    if i >= 0 and i < len(self.points):
                        cost[i] = float('inf')
            
            # lookahead_point = self.points[np.argmin(cost)]
            # self.lp = lookahead_point

            if len(obstacle) != 0:
                lookahead_point = self.points[np.argmin(cost)]
                drive_msg.drive.speed = 1.0
                drive_msg.drive.steering_angle = self.pure_pursuit(self.pose_theta, self.pose, lookahead_point, np.linalg.norm(self.pose-lookahead_point)/3, wheelbase=self.wheelbase)
            else:
                lookahead_point = self.optimal_lookahead_point
                drive_msg.drive.speed          = self.velocity_profile * 0.8
                drive_msg.drive.steering_angle = self.pure_pursuit(self.pose_theta, self.pose, lookahead_point, np.linalg.norm(self.pose-lookahead_point), wheelbase=self.wheelbase)
            
            if np.min(cost) == float('inf'):
                lookahead_point = np.array([0.0, 0.0])
                drive_msg.drive.speed          = -0.0
                drive_msg.drive.steering_angle = 0.0

            self.drive_pub.publish(drive_msg)

            self.lp = lookahead_point
            self.visualize()
            
    def visualize(self):
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
        
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.id = 0
        marker.ns = "target_waypoints"
        marker.type = 1
        marker.action = 0
        marker.pose.position.x = self.lp[0]
        marker.pose.position.y = self.lp[1]
              
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.lifetime.nanosec = int(1e8)

        marker_arr.markers.append(marker)

        self.visualize_pub.publish(marker_arr)

    def done_callback(train_msg):
        done = train_msg.done
        if done:
            print("========================Done================================")
            Agent.prev_L = None
            return

def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    rclpy.spin(planner)

    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

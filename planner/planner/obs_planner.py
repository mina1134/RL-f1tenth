import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from transforms3d import euler
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import os 

class PurePursuit(Node):
    def __init__(self):
    	super().__init__('purepursuit_planner')
    	
    	self.declare_parameter('odom_topic', '/ego_racecar/odom')
    	self.declare_parameter('wheelbase', 0.3302)
    	self.declare_parameter('wpt_path', '/home/mina/sim_ws/src/planner/maps/Spielberg_raceline.csv')
    	self.declare_parameter('wpt_delim', ';')
    	self.declare_parameter('wpt_rowskip', 0)
    	self.declare_parameter('car_width', 0.31)
    	self.declare_parameter('sx', -0.0440806)
    	self.declare_parameter('sy', -0.8491629)
    	self.declare_parameter('stheta', 3.4034118)
    	
    	self.pose_topic = self.get_parameter('odom_topic').value
    	self.wheelbase = self.get_parameter('wheelbase').value
    	self.load_waypoints(self.get_parameter('wpt_path').value, self.get_parameter('wpt_delim').value, self.get_parameter('wpt_rowskip').value)
    	self.car_width = self.get_parameter('car_width').value
    	self.pose = np.array([self.get_parameter('sx').value, self.get_parameter('sy').value])
    	self.pose_theta = self.get_parameter('stheta').value
  
    	
    	## hyper parameters ##
    	self.vgain = 1.0
    	self.lookahead_distance = 1.2
    	
    	# Subscriber
    	self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_callback, 1)
    	self.obs_ld_sub = self.create_subscription(Point, '/obs_ld', self.obs_callback, 1)
    	
    	# Publisher
    	self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
    	
    	self.obs = False
    	

    def obs_callback(self, point):
        self.obs = True
        self.obs_ld = np.array([point.x, point.y])
        
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
    

    def get_waypoint(self, pose, waypoints):
        diff = np.linalg.norm(pose - waypoints, axis=1)
        current_idx = np.argmin(diff)
   
        in_range = np.where(diff <= self.lookahead_distance)[0]
        if np.max(in_range) == waypoints.shape[0]-1 or current_idx == waypoints.shape[0]-1:
            i = 0
            while (i in list(in_range)):
                i+=1
            return i+1
        return np.max(in_range)+1   
         


    def pose_callback(self, pose_msg):
        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, 
                pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        self.pose_theta = euler.quat2euler(quat, axes='sxyz')[0]
        self.pose = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])

        waypoint = self.waypoints[:, 1:3]

        lookahead_idx = self.get_waypoint(self.pose, waypoint)
        lookahead_point = waypoint[lookahead_idx,:]
        
        if self.obs:
            lookahead_point = self.obs_ld
            self.obs = False
        
        if lookahead_point is None:
            return 0.0, 0.0

        speed, steer = self.get_actuation(self.pose_theta, lookahead_point, self.pose, self.lookahead_distance, self.wheelbase)
        speed = self.vgain * speed

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed          = speed
        drive_msg.drive.steering_angle = steer
        self.drive_pub.publish(drive_msg)

class Planner(Node):

    def __init__(self):
    	super().__init__('f1tenth_kor_planner')
    	
    	self.declare_parameter('scan_topic', '/scan')
    	self.declare_parameter('odom_topic', '/ego_racecar/odom')
    	self.declare_parameter('wheelbase', 0.3302)
    	self.declare_parameter('wpt_path', 'src/f1tenth_korea/csv/Spielberg_centerline.csv')
    	self.declare_parameter('wpt_delim', ',')
    	self.declare_parameter('wpt_rowskip', 0)
    	self.declare_parameter('car_width', 0.31)
    	self.declare_parameter('sx', 9.53)
    	self.declare_parameter('sy', 0.69)
    	self.declare_parameter('stheta', 3.14)
    	self.declare_parameter('lookahead_ind', 4)
    	
    	self.scan_topic = self.get_parameter('scan_topic').value
    	self.pose_topic = self.get_parameter('odom_topic').value
    	self.wheelbase = self.get_parameter('wheelbase').value
    	self.load_waypoints(self.get_parameter('wpt_path').value, self.get_parameter('wpt_delim').value, self.get_parameter('wpt_rowskip').value)
    	self.car_width = self.get_parameter('car_width').value
    	self.angle_offset = -np.pi/4
    	self.pose = np.array([self.get_parameter('sx').value, self.get_parameter('sy').value])
    	self.pose_theta = self.get_parameter('stheta').value
    	self.points = []
    	self.obs_lookahead_distance = 2.0
    	self.lookahead_ind = self.get_parameter('lookahead_ind').value
    	self.lookahead_distance = 2.0
    	self.lookahead_idx = None
    	self.track_boundary = 0.25
    	
    	## hyper parameters ##
    	self.wpts_interval = 0.15
    	self.vgain = 0.5
    	
    	# Subscriber
    	self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 1)
    	self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_callback, 1)
    	
    	# Publisher
    	self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
    	self.visualize_pub = self.create_publisher(MarkerArray, '/visualize/points', 10)
    	self.ld_pub = self.create_publisher(Point, '/obs_ld', 10)
    	

    def load_waypoints(self, path, delim, rowskip):
        self.waypoints = np.loadtxt(path, delimiter=delim, skiprows=rowskip)


    def pure_pursuit(self, position, pose_theta, lookahead_point, lookahead_distance, wheelbase):
        # Extract the Waypoint information
        waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
        if np.abs(waypoint_y) < 1e-6:
            return 3.0, 0.
        # Define the radius of the arc to follow
        radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)
        # Calculate the steering angle based on the curvature of the arc to follow
        steering_angle = np.arctan(wheelbase / radius)

        return steering_angle
    
    def get_steer(self, abs_position):  # transfer to polar coordinate
        rotation_matrix = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)],
                                    [np.sin(np.pi/2), np.cos(np.pi/2)]])
        coordinate = abs_position@rotation_matrix

        r = np.linalg.norm(coordinate)
        theta = np.tan(coordinate[1]/coordinate[0])
        return r, theta
    
    def global_to_local(self, coordinate, angle, origin):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        coordinate = (coordinate - origin).T
        
        return (rotation_matrix@coordinate).T


    def calc_norm_vec(self, waypoints, waypoint_id):
        wpts = np.vstack((waypoints, waypoints))

        v = wpts[waypoint_id+4] - wpts[waypoint_id]
        if v[1] == 0:
            norm_vec = np.array((0,1))
        else:
            norm_vec = np.array((1,-v[0]/v[1]))
        unit_vec = norm_vec/np.linalg.norm(norm_vec)
        return unit_vec
    

    def get_waypoint(self, pose, waypoints, current_idx, L):
        diff = np.linalg.norm(pose - waypoints, axis=1)
        in_range = np.where(diff <= self.obs_lookahead_distance)[0]

        if len(in_range) == 0:
            return None
            
        if np.max(in_range) == waypoints.shape[0]-1:
            if np.any(np.array(in_range) < len(in_range)):
                in_range = np.where(np.array(in_range) < len(in_range))

        return np.max(in_range)

    def pose_callback(self, pose_msg):
        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, 
                pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        self.pose_theta = euler.quat2euler(quat, axes='sxyz')[0]
        if self.pose_theta < 0:
            self.pose_theta = 2*np.pi + self.pose_theta
        self.pose = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])

        centerline = self.waypoints[:, :2]
        diff = np.linalg.norm(self.pose - centerline, axis=1)
        current_idx = np.argmin(diff)
        self.lookahead_idx = self.get_waypoint(self.pose, centerline, current_idx, self.obs_lookahead_distance)
        
        if self.lookahead_idx == None:
            return
            
        lookahead_point = centerline[self.lookahead_idx, :]
        track_bound_l = self.waypoints[self.lookahead_idx, 2]
        track_bound_r = self.waypoints[self.lookahead_idx, 3]
        norm_vec = self.calc_norm_vec(centerline, self.lookahead_idx)

        self.points = []
        # speed = self.waypoints[self.lookahead_idx, 5] * self.vgain
        speed = 1.0
        
        if self.lookahead_idx == self.waypoints.shape[0]-1:
            wp_vec = 0
        else:
            wp_vec = centerline[self.lookahead_idx+1] - centerline[self.lookahead_idx]

        if self.pose_theta < np.pi:
            i=0
            while(i <= track_bound_l - self.track_boundary):
                self.points.append(-i*norm_vec + lookahead_point)
                i += self.wpts_interval
            self.points.reverse()
            i = self.wpts_interval
            while(i <= track_bound_r - self.track_boundary):
                self.points.append( i*norm_vec + lookahead_point)
                i += self.wpts_interval
        elif self.pose_theta > np.pi:
            i=0
            while(i <= track_bound_l - self.track_boundary):
                self.points.append( i*norm_vec + lookahead_point)
                i += self.wpts_interval
            self.points.reverse()
            i = self.wpts_interval
            while(i <= track_bound_r - self.track_boundary):
                self.points.append(-i*norm_vec + lookahead_point)
                i += self.wpts_interval

        if len(self.points) == 0:
            self.points.append(lookahead_point)

    def get_costs(self, position, pose_theta, points, wheelbase):
        costs = []
        for lookahead_point in points:
            lookahead_distance = np.linalg.norm(lookahead_point - position)
            cost = self.pure_pursuit(position, pose_theta, lookahead_point, lookahead_distance, wheelbase)
            costs.append(abs(cost))

        return costs
                    
        
    def scan_callback(self, scan_msg):
        scan_range = len(scan_msg.ranges)
        
        if len(self.points) == 0:
            return

        if self.pose_theta >= np.pi / 2 and self.pose_theta < 3* np.pi / 2:
            lidar_position = self.pose - 0. * np.array([1, np.tan(self.pose_theta)] / np.linalg.norm(np.array([1, np.tan(self.pose_theta)])))
        else:
            lidar_position = self.pose + 0. * np.array([1, np.tan(self.pose_theta)] / np.linalg.norm(np.array([1, np.tan(self.pose_theta)])))
        
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
            if dist2pt[i] > scan[point_indices[i]]:
                start = i
                while dist2pt[i] > scan[point_indices[i]]:
                    if i == len(self.points)-1:
                        break
                    i+=1
                end = i
                obstacle.append((start,end))
            i+=1
        
        ## points 중에서 cost가 가장 낮은 점 찾기
        cost = self.get_costs(self.pose, self.pose_theta, self.points, self.wheelbase)
        for obs in obstacle:
            for i in range(obs[0]-2, obs[1]+3):
                if i >= 0 and i <= len(self.points)-1:
                    cost[i] = float('inf')
                    self.obs.append(i)
                    
        lookahead_point = self.points[np.argmin(cost)]
        
        if len(self.obs) > 2:
            ld = Point()
            ld.x = lookahead_point[0]
            ld.y = lookahead_point[1]
            self.ld_pub.publish(ld)

        self.visualize(lookahead_point, self.obs)
        
    def visualize(self, lookahead_point, obstacle):
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

            if lookahead_point[0] == self.points[i][0]:
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0     
            elif i in obstacle:
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
        
        self.visualize_pub.publish(marker_arr)
        
def main(args=None):
    rclpy.init(args=args)
    pp = PurePursuit()
    obs_planner = Planner()

    executor = MultiThreadedExecutor()

    executor.add_node(pp)
    executor.add_node(obs_planner)

    try:
        executor.spin()
    finally:
        executor.remove_node(pp)
        executor.remove_node(obs_planner)
        rclpy.shutdown()


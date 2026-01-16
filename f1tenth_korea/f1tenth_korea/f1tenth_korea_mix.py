import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from transforms3d import euler

import numpy as np

class Planner(Node):

    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 3000000

    def __init__(self):
    	super().__init__('f1tenth_kor_planner')
    	
    	self.declare_parameter('scan_topic', '/scan')
    	self.declare_parameter('odom_topic', '/ego_racecar/odom')
    	self.declare_parameter('wheelbase', 0.3302)
    	self.declare_parameter('wpt_path', '/home/mina/sim_ws/src/f1tenth_korea/maps/centerline_edited.csv')
    	self.declare_parameter('wpt_delim', ',')
    	self.declare_parameter('wpt_rowskip', 0)
    	self.declare_parameter('car_width', 0.31)
    	self.declare_parameter('sx', -9.25)
    	self.declare_parameter('sy', 5.61)
    	self.declare_parameter('stheta', 4.71)
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
    	self.lookahead_ind = self.get_parameter('lookahead_ind').value
    	self.lookahead_distance = 0.8
    	self.past_idx = 0
    	self.lookahead_idx = None
    	
    	# FGM Params
    	self.RACECAR_LENGTH = 0.3302
    	self.PI = 3.141592
    	self.ROBOT_SCALE = 0.2032
    	self.LOOK = 0.01
    	self.THRESHOLD = 1.2
    	self.FILTER_SCALE = 1.1
    	self.GAP_THETA_GAIN = 20.0
    	self.REF_THETA_GAIN = 1.5
    	self.BEST_POINT_CONV_SIZE = 100
    	self.interval = 0.00435
    	self.front_idx = 0
    	
    	## hyper parameters ##
    	self.wpts_interval = 0.15
    	self.vgain = 0.7
    	
    	# Subscriber
    	self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 1)
    	self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_callback, 1)
    	
    	# Publisher
    	self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
    	

    def load_waypoints(self, path, delim, rowskip):
        self.waypoints = np.loadtxt(path, delimiter=delim, skiprows=rowskip)


    def preprocess_lidar(self, ranges, scan_range):
        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges)
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges
    
    def find_gap(self, scan, scan_range):
        self.gap = []
        gap_start = 0
        gap_end = 0

        i = 0
        best_gap_size = 0
       
        while i < scan_range:
            if scan[i] > self.THRESHOLD: 
                start_idx_temp = i
                end_idx_temp = i

                while ((scan[i] > self.THRESHOLD) and (i + 1 < scan_range)):
                    i += 1

                end_idx_temp = i

                gap_size = np.fabs(end_idx_temp - start_idx_temp)
              
                if gap_size > 30:
                    gap_start = start_idx_temp
                    gap_end = end_idx_temp
                    
                
               

            i += 1
          
        if best_gap_size > 0:
            self.gap.append(gap_start)
            self.gap.append(gap_end)


    def find_best_point(self, scan):
        averaged_max_gap = np.convolve(scan[self.gap[0]:self.gap[1]], np.ones(self.BEST_POINT_CONV_SIZE),
                                       'same') / self.BEST_POINT_CONV_SIZE
        
        return averaged_max_gap.argmax() + self.gap[0]  


    def calculate_steering_and_speed(self, best_point, angle_increment):
        #steer = best_point * angle_increment - np.radians(135)
        angle = best_point * angle_increment + np.radians(45)
        radius = 0.9 / (2*np.sin(angle))
        steer = -np.arctan(self.wheelbase / radius)
        speed = 1.5
        return speed, steer

    def fgm(self, scan, scan_range, angle_increment):
        self.find_gap(scan, scan_range)
        best_point = self.find_best_point(scan)
        speed, steer = self.calculate_steering_and_speed(best_point, angle_increment)
        
        return speed, steer
            
    def get_steer(self, position, pose_theta, lookahead_point, lookahead_distance, wheelbase):
        # Extract the Waypoint information
        waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
        if np.abs(waypoint_y) < 1e-6:
            return 3.0, 0.
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

        v = wpts[waypoint_id+4] - wpts[waypoint_id]
        if v[1] == 0:
            norm_vec = np.array((0,1))
        else:
            norm_vec = np.array((1,-v[0]/v[1]))
        unit_vec = norm_vec/np.linalg.norm(norm_vec)
        return unit_vec
    

    def get_waypoint(self, pose, waypoints):
        diff = np.linalg.norm(pose - waypoints, axis=1)
        current_idx = np.argmin(diff)
        print(current_idx)
        
        self.lookahead_distance = self.waypoints[current_idx, self.lookahead_ind]
   
        in_range = np.where(diff <= self.lookahead_distance)[0]
        if len(in_range) == 0:
            return None
            
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

        print(self.pose)
        centerline = self.waypoints[:, :2]
        self.lookahead_idx = self.get_waypoint(self.pose, centerline)
        if self.lookahead_idx == None:
            return

        lookahead_point = centerline[self.lookahead_idx, :]
        track_bound_l = self.waypoints[self.lookahead_idx, 2]
        track_bound_r = self.waypoints[self.lookahead_idx, 3]
        norm_vec = self.calc_norm_vec(centerline, self.lookahead_idx)

        self.points = []
        speed = self.waypoints[self.lookahead_idx, 5] * self.vgain
        if self.lookahead_idx == self.waypoints.shape[0]-1:
            wp_vec = 0
        else:
            wp_vec = centerline[self.lookahead_idx+1] - centerline[self.lookahead_idx]
                    
        theta = np.arctan(wp_vec[1]/wp_vec[0])
        
        if speed < 3.0:
            if self.pose_theta < np.pi:
                i=0
                while(i <= track_bound_l - self.car_width - 0.3):
                    self.points.append(-i*norm_vec + lookahead_point)
                    i += self.wpts_interval
                self.points.reverse()
                i = self.wpts_interval
                while(i <= track_bound_r - self.car_width - 0.2):
                    self.points.append( i*norm_vec + lookahead_point)
                    i += self.wpts_interval
            elif self.pose_theta > np.pi:
                i=0
                while(i <= track_bound_l - self.car_width - 0.3):
                    self.points.append( i*norm_vec + lookahead_point)
                    i += self.wpts_interval
                self.points.reverse()
                i = self.wpts_interval
                while(i <= track_bound_r - self.car_width - 0.2):
                    self.points.append(-i*norm_vec + lookahead_point)
                    i += self.wpts_interval
        else:
            if self.pose_theta < np.pi:
                i=0
                while(i <= track_bound_l - self.car_width):
                    self.points.append(-i*norm_vec + lookahead_point)
                    i += self.wpts_interval
                self.points.reverse()
                i = self.wpts_interval
                while(i <= track_bound_r - self.car_width):
                    self.points.append( i*norm_vec + lookahead_point)
                    i += self.wpts_interval
            elif self.pose_theta > np.pi:
                i=0
                while(i <= track_bound_l - self.car_width):
                    self.points.append( i*norm_vec + lookahead_point)
                    i += self.wpts_interval
                self.points.reverse()
                i = self.wpts_interval
                while(i <= track_bound_r - self.car_width):
                    self.points.append(-i*norm_vec + lookahead_point)
                    i += self.wpts_interval
                    
        if len(self.points) == 0:
            self.points.append(lookahead_point)
                    
        
    def scan_callback(self, scan_msg):
        scan_range = len(scan_msg.ranges)
        scan = self.preprocess_lidar(scan_msg.ranges, scan_range)
        
        if len(self.points) == 0 or self.lookahead_idx == None:
            print('No points')
            speed, steer = self.fgm(scan, scan_range, scan_msg.angle_increment)
        else:
            if self.pose_theta > np.pi / 2 and self.pose_theta < 3* np.pi / 2:
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
            i = 0
                
            while i < len(self.points):
                if dist2pt[i]> scan[point_indices[i]] - 0.2:
                    start = i
                    while dist2pt[i] - 0.2 > scan[point_indices[i]] - 0.2:
                        if i == len(self.points)-1:
                            break
                        i+=1
                    end = i
                    obstacle.append((start,end))
                i+=1

            ## points 중에서 cost가 가장 낮은 점 찾기
            cost = np.linalg.norm(np.array(self.points) - self.pose, axis=1)
            for obs in obstacle:
                for i in range(obs[0]-4, obs[1]+5):
                    if i >= 0 and i <= len(self.points)-1:
                        cost[i] = float('inf')
            lookahead_point = self.points[np.argmin(cost)]

            if np.min(cost) == float('inf'):
                print('FGM')
                speed, steer = self.fgm(scan, scan_range, scan_msg.angle_increment)
            else:
                steer = self.get_steer(self.pose, self.pose_theta, lookahead_point, np.min(cost), self.wheelbase)
                speed = self.waypoints[self.lookahead_idx, 5] * self.vgain

 
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed          = speed
        drive_msg.drive.steering_angle = steer
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    print("Start")
    rclpy.spin(planner)

    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from transforms3d import euler
from interfaces.msg import Goal, Train

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
        self.declare_parameter('wpt_path', 'src/reinforcement_learning/waypoints/Spielberg_map_centerline.csv')
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
        self.lookahead_distance = 1.0
        self.past_idx = 0
        self.lookahead_idx = None
        self.obs = []
        self.lp = [0, 0]
        self.train_pause = False

        ## hyper parameters ##
        self.wpts_interval = 0.15
        self.track_boundary = self.car_width/2
        self.SAFE_DIST = 0 # index

        # Subscriber
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 1)
        self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_callback, 1)
        self.action_sub = self.create_subscription(Train, '/agent/action', self.get_lookahead_distance, 1)

        # Publisher
        self.goal_pub = self.create_publisher(Goal, '/goal_point', 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.visualize_pub = self.create_publisher(MarkerArray, '/visualize/points', 10)

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

        v = wpts[waypoint_id+4] - wpts[waypoint_id]
        if v[1] == 0:
            norm_vec = np.array((0,1))
        else:
            norm_vec = np.array((1,-v[0]/v[1]))
        unit_vec = norm_vec/np.linalg.norm(norm_vec)
        return unit_vec
    
    def get_lookahead_distance(self, action_msg):
        self.lookahead_distance = action_msg.action.lookahead_distance
        

    def get_waypoint(self, pose, waypoints, current_idx, L):
        stacked_waypoints = np.vstack((waypoints, waypoints))

        round = (stacked_waypoints[:, 0] - pose[0])**2 + (stacked_waypoints[:, 1] - pose[1])**2
        in_range = np.where(round < (L)**2)[0]
        in_range = in_range[np.where(in_range - current_idx < 30)[0]]

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

        self.lookahead_idx = self.get_waypoint(self.pose, centerline, current_idx, self.lookahead_distance)
        if self.lookahead_idx == None:
            return

        lookahead_point = centerline[self.lookahead_idx, :]
        track_bound_l = self.waypoints[self.lookahead_idx, 2]
        track_bound_r = self.waypoints[self.lookahead_idx, 3]
        norm_vec = self.calc_norm_vec(centerline, self.lookahead_idx)


        self.points = []
        
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

        
    def scan_callback(self, scan_msg):
        scan_range = len(scan_msg.ranges)
        scan = self.preprocess_lidar(scan_msg.ranges, scan_range)
        
        if len(self.points) == 0 or self.lookahead_idx == None:
            drive_msg = AckermannDriveStamped()
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
            ## 조향
            cost = self.pure_pursuit(self.pose_theta, self.pose, np.array(self.points), self.lookahead_distance, wheelbase=self.wheelbase)

            for obs in obstacle:
                for i in range(obs[0]-self.SAFE_DIST, obs[1]+self.SAFE_DIST):
                    if i >= 0 and i <= len(self.points)-1:
                        cost[i] = float('inf')

            if np.min(cost) == float('inf'):
                self.train_pause = True
                drive_msg = AckermannDriveStamped()
                drive_msg.drive.speed          = 0.0
                drive_msg.drive.steering_angle = 0.0
                return
            
            lookahead_point = self.points[np.argmin(cost)]
            self.lp = lookahead_point

            lookahead_point = self.waypoints[self.lookahead_idx, :2]
            self.lp = lookahead_point
            self.get_logger().info(f'obs_avoid: {self.pose}')
            goal_msg = Goal()
            goal_msg.goal_point.x = lookahead_point[0]
            goal_msg.goal_point.y = lookahead_point[1]
            goal_msg.train_pause = self.train_pause
            
            self.goal_pub.publish(goal_msg)
            self.train_pause = False

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

def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    print("Start")
    rclpy.spin(planner)

    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

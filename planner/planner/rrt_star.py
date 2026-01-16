import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from scipy.interpolate import interp1d
from transforms3d import euler

import numpy as np
from numba import njit
import math
import random


class RandomPlanner(Node):
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 3000000

    MAX_RRT_SAMPLING_ITERATION = 50
    RRT_RADIUS = 3.0
    BUBBLE_RANGE = 15

    def __init__(self):
        super().__init__('rrt_star')

        self.declare_parameter('waypoints_path', '/home/mina/sim_ws/src/planner/maps/Spielberg_raceline.csv')
        self.declare_parameter('lookahead_distance', 4.0)
        self.declare_parameter('odom_topic', '/ego_racecar/odom')

        self.L = float(self.get_parameter('lookahead_distance').value)
        self.waypoints_path = str(self.get_parameter('waypoints_path').value)
        self.pure_pursuit = PurePursuit(L=self.L, filepath=self.waypoints_path)
        self.odom_topic = str(self.get_parameter('odom_topic').value)


        # Subscriber
        self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_callback, 1)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 1)
        self.timer = self.create_timer(0.5, self.timer_callback)

        # Publisher
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        self.wheelbase = 0.17145 + 0.15875
        self.pose = np.array([-0.0440806, -0.8491629])
        self.pose_theta = 3.4034118
        self.tree = dict()
        self.path = []
        self.local_goal_point = None
        self.get_scan = True

    def timer_callback(self):
        self.get_scan = True

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges)
        # sets each value to the mean over a given window
        # proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    def scan_callback(self, scan_msg):
        if self.get_scan == False:
            return

        scans = self.preprocess_lidar(scan_msg.ranges)

        self.make_tree(self.pose, self.pose_theta, scans, scan_msg.angle_increment)
        self.get_scan = False


    def coordinate_transform(self, coordinate, angle, origin):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        
        coordinate = np.array([coordinate - origin]).T

        return rotation_matrix@coordinate.T[0]
        
        
    def local_to_global(self, coordinate, angle, reference_frame):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        
        coordinate = rotation_matrix@coordinate

        return coordinate + reference_frame
        

    def line_on_2p(self, p1, p2):
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        m = (y1- y2) / (x1 - x2)
        bias = -m*x1 +y1             # y = m * (x - x1) + y1
        return m, bias

    def collision(self, scan, pose, heading, source, target, radians_per_elem, first_scan_theta = -45):
        """
        if heading > np.pi / 2 and heading < 3* np.pi / 2:
            lidar_position = pose - 0.29 * np.array([1, np.tan(heading)] / np.linalg.norm(np.array([1, np.tan(heading)])))
        else:
            lidar_position = pose + 0.29 * np.array([1, np.tan(heading)] / np.linalg.norm(np.array([1, np.tan(heading)])))

        pose = self.coordinate_transform(pose, -(heading - np.pi/2), lidar_position)
        source = self.coordinate_transform(source, -(heading - np.pi/2), lidar_position)
        target = self.coordinate_transform(target, -(heading - np.pi/2), lidar_position)
        
        base_vec = (1, np.tan(math.radians(first_scan_theta)))

        if target[0] == pose[0] and target[1] == pose[1]:
            if base_vec[1] * source[0] <= source[1]:
                theta = np.arccos(np.dot(source, base_vec) / (np.linalg.norm(source)*np.linalg.norm(base_vec)))
            else:
                theta = 2*np.pi - np.arccos(np.dot(source, base_vec) / (np.linalg.norm(source)*np.linalg.norm(base_vec)))

            scan_id = int(theta / radians_per_elem)

            if scan_id >= 1080:
                return True

            # collision checking
            start = scan_id - self.BUBBLE_RANGE
            end = scan_id + self.BUBBLE_RANGE

            if end > 1079:
                end = 1079

            for i in range(start, end + 1, 4):
                if scan[i] <= np.linalg.norm(source):
                   return True
            return False
        """
        
        source = self.coordinate_transform(source, -(heading - np.pi/2), pose)
        target = self.coordinate_transform(target, -(heading - np.pi/2), pose)

        base_vec = (1, np.tan(math.radians(first_scan_theta)))

        if target[0] == 0 and target[1] == 0:
            if base_vec[1] * source[0] <= source[1]:
                theta = np.arccos(np.dot(source, base_vec) / (np.linalg.norm(source)*np.linalg.norm(base_vec)))
            else:
                theta = 2*np.pi - np.arccos(np.dot(source, base_vec) / (np.linalg.norm(source)*np.linalg.norm(base_vec)))

            scan_id = int(theta / radians_per_elem)

            if scan_id >= 1080:
                return True

            # collision checking
            start = scan_id - self.BUBBLE_RANGE
            end = scan_id + self.BUBBLE_RANGE

            if end > 1079:
                end = 1079

            for i in range(start, end + 1, 4):
                if scan[i] <= np.linalg.norm(source):
                   return True
            return False

        else:
            if base_vec[1] * source[0] <= source[1]:
                source_theta = np.arccos(np.dot(source, base_vec) / (np.linalg.norm(source)*np.linalg.norm(base_vec)))
            else:
                source_theta = 2*np.pi - np.arccos(np.dot(source, base_vec) / (np.linalg.norm(source)*np.linalg.norm(base_vec)))

            if base_vec[1] * target[0] <= target[1] :
                target_theta = np.arccos(np.dot(target, base_vec) / (np.linalg.norm(target)*np.linalg.norm(base_vec)))
            else:
                target_theta = 2*np.pi - np.arccos(np.dot(target, base_vec) / (np.linalg.norm(target)*np.linalg.norm(base_vec)))


            if source_theta  == None or target_theta == None:
                return True

            source_scan_id = int(source_theta / radians_per_elem)
            target_scan_id = int(target_theta / radians_per_elem)

            if source_scan_id >= 1080:
                return True

            
            # collision checking
            edge_m, edge_b = self.line_on_2p(source, target)

            if source_scan_id < target_scan_id:
                start = source_scan_id - self.BUBBLE_RANGE
                end = target_scan_id + self.BUBBLE_RANGE
            elif source_scan_id > target_scan_id:
                start = target_scan_id - self.BUBBLE_RANGE
                end = source_scan_id + self.BUBBLE_RANGE
            else:
                if scan[source_scan_id] <= np.linalg.norm(source):
                    return True
                return False
            
            if end > 1079:
                end = 1079

            for i in range(start, end + 1, 4):
                theta =  i * radians_per_elem

                A = np.array([[edge_m, -1.],
                            [np.tan(theta + math.radians(first_scan_theta)), -1]])
                b = np.array([[-edge_b, 0.]]).T

                X = np.linalg.inv(A)@b
            
                if scan[i] <= np.linalg.norm(X):
                    return True
            return False


    def make_tree(self, pose, heading, scan, radians_per_elem):
        self.tree = dict()
        self.path = []
        self.tree[tuple(pose)] = 'root'

        # set goal point
        self.local_goal_point = self.pure_pursuit.get_waypoint(pose)

        iter = 0
        while iter < self.MAX_RRT_SAMPLING_ITERATION:
            
            ## sampling
            #sample_x = random.uniform(min(pose[0],self.local_goal_point[0]), max(pose[0],self.local_goal_point[0]))
            #sample_y = random.uniform(min(pose[1],self.local_goal_point[1]), max(pose[1],self.local_goal_point[1]))
            #sample = np.array([sample_x, sample_y])
            sample = np.array([random.uniform(pose[0] - 4 , pose[0] + 4), random.uniform(pose[1] - 4 , pose[1] + 4)])
            #lidar_index = random.randrange(180,900)
            #sample_dist = random.uniform(0,4)

            #theta = lidar_index * radians_per_elem - np.radians(45)
            #if lidar_index <= 540:
            #    sample = np.array([sample_dist, sample_dist * np.tan(theta)])
            #else:
            #    sample = -np.array([sample_dist, sample_dist * np.tan(theta)])

            #sample = self.coordinate_transform(sample, heading - np.pi/2, lidar_position)


            ## 트리에서 가장 가까운 노드 찾기
            nearest = self.tree[tuple(pose)]
            nearest_dist = float('inf')
            for node in self.tree.keys():
                dist = np.linalg.norm(sample - np.array(node))
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = node
            
            ## nearest 노드에서 sampling된 포인트 방향으로 0.6만큼 떨어진 지점에 노드 생성
            vec = sample - nearest
            unit_vec = vec / np.linalg.norm(vec)
            vec = unit_vec * 0.6
            sample = nearest + vec
           
            ## connection with parent node
            min = float('inf')
            dist = float('inf')
            parent = None
            
            for node in self.tree.keys():
                # RRT_RADIUS 거리 내에 있고 생성한 노드와 자신 사이에 장애물이 없는 노드
                if (np.sum(np.power(np.array(node) - sample, 2)) <= self.RRT_RADIUS**2) and (not self.collision(scan, pose, heading, sample, node, radians_per_elem)):
                    dist = np.linalg.norm(sample - np.array(node))
                    n = node
                    # 이 노드를 부모노드로 놓았을 때 root노드까지 걸리는 비용 계산
                    while self.tree[n] != 'root':
                        dist += np.linalg.norm(np.array(n) - np.array(self.tree[n]))
                        n = self.tree[n]
                    # 비용이 가장 작은 노드를 부모 노드로 설정
                    if dist < min:
                        min = dist
                        parent = node
           
            # update tree
            if parent != None:
                self.tree[tuple(sample)] = parent

            iter +=1

        ## 트리에서 최종 경로 선택
        goal = None
        dist = float('inf')
        # 트리에 추가된 노드 중 local_goal_point와 가장 가까운 노드 찾기 --> goal에 할당
        for node in self.tree.keys():
            d = np.linalg.norm(self.local_goal_point - np.array(node))
            if d < dist:
                dist = d
                goal = node

        ## goal 노드부터 트리를 거꾸로 올라가며 root(ego)까지 경로 추적
        p = []
        n = goal
        while self.tree[n] != 'root':
            p.append(n)
            n = self.tree[n]
        p.append(pose)
        p.reverse()

        ## node - node 사이를 0.2씩 보간
        v = np.array(p[1:]) - np.array(p[:-1])
        for i in range(len(v)):
            segment = np.arange(0,np.linalg.norm(v[i]),0.2)
            for s in segment:
                point = np.array(p[i]) + ( v[i]/np.linalg.norm(v[i]) ) * s 
                self.path.append(point)
        self.path.append(np.array(p[-1]))


    # @njit(fastmath=False, cache=True)
    def find_lookahead_point(self, nearest_point, i, trajectory, lookahead_distance):
        trajectory = trajectory[i:]
        diff = trajectory - nearest_point
        dist = np.linalg.norm(diff, axis=1)
        d = np.where(dist>lookahead_distance)[0]
        if len(d) == 0:
            return trajectory[-1]
        return trajectory[d[0]]


    def _get_current_waypoint(self, lookahead_distance):

        path = np.array(self.path)

        nearest_point, nearest_dist, t, i = self.pure_pursuit.nearest_point_on_trajectory(self.pose, path)

        current_waypoint = np.empty((2,))

        current_waypoint = self.find_lookahead_point(nearest_point, i, path, lookahead_distance)

        return current_waypoint


    def pose_callback(self, pose_msg):
        if len(self.path) == 0:
            return
        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, 
                pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        self.pose_theta = euler.quat2euler(quat, axes='sxyz')[0]
        self.pose = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])

        lookahead_point = self._get_current_waypoint(2.0)

        velocity, angle = self.pure_pursuit.get_actuation(self.pose_theta, lookahead_point, self.pose, 2.0, self.wheelbase)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed          = velocity
        drive_msg.drive.steering_angle = angle
        self.drive_pub.publish(drive_msg)




class PurePursuit:
    def __init__(self, L=1.7, segments=1024, filepath="/sim_ws/src/f1tenth_gym_ros/f1tenth_planner/waypoint/e7_floor5.csv"):
        # TODO: Make self.L a function of the current velocity, so we have more intelligent RRT
        self.L = L
        self.waypoints = np.loadtxt(filepath, delimiter=';', skiprows=3)[:, 1:4]

        print(f"Loaded {len(self.waypoints)} waypoints")


    # @njit(fastmath=False, cache=True)
    def nearest_point_on_trajectory(self, point, trajectory):
        """
        Return the nearest point along the given piecewise linear trajectory.
        Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
        not be an issue so long as trajectories are not insanely long.
            Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)
        point: size 2 numpy array
        trajectory: Nx2 matrix of (x,y) trajectory waypoints
            - these must be unique. If they are not unique, a divide by 0 error will destroy the world
        """
        diffs = trajectory[1:, :2] - trajectory[:-1, :2]
        l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
        # this is equivalent to the elementwise dot product
        # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
        dots = np.empty((trajectory.shape[0] - 1,))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - trajectory[i, :2]), diffs[i, :])
        t = dots / l2s
        t[t < 0.0] = 0.0
        t[t > 1.0] = 1.0
        # t = np.clip(dots / l2s, 0.0, 1.0)
        projections = trajectory[:-1, :2] + (t * diffs.T).T
        # dists = np.linalg.norm(point - projections, axis=1)
        dists = np.empty((projections.shape[0],))
        for i in range(dists.shape[0]):
            temp = point - projections[i]
            dists[i] = np.sqrt(np.sum(temp * temp))
        if len(dists) != 0:
            min_dist_segment = np.argmin(dists)
            return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment
        else:
            return trajectory[0], None, None, 0

    # @njit(fastmath=False, cache=True)
    def first_point_on_trajectory_intersecting_circle(self, point, radius, trajectory, t=0.0, wrap=False):
        """
        starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.
        Assumes that the first segment passes within a single radius of the point
        http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
        """
        start_i = int(t)
        start_t = t % 1.0
        first_t = None
        first_i = None
        first_p = None
        trajectory = np.ascontiguousarray(trajectory)
        for i in range(start_i, trajectory.shape[0] - 1):
            start = trajectory[i, :2]
            end = trajectory[i + 1, :2] + 1e-6
            V = np.ascontiguousarray(end - start)

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            #   print "NO INTERSECTION"
            # else:
            # if discriminant >= 0.0:
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if i == start_i:
                if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                    first_t = t1
                    first_i = i
                    first_p = start + t1 * V
                    break
                if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                    first_t = t2
                    first_i = i
                    first_p = start + t2 * V
                    break
            elif t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        # wrap around to the beginning of the trajectory if no intersection is found1
        if wrap and first_p is None:
            for i in range(-1, start_i):
                start = trajectory[i % trajectory.shape[0], :2]
                end = trajectory[(i + 1) % trajectory.shape[0], :2] + 1e-6
                V = end - start

                a = np.dot(V, V)
                b = 2.0 * np.dot(V, start - point)
                c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
                discriminant = b * b - 4 * a * c

                if discriminant < 0:
                    continue
                discriminant = np.sqrt(discriminant)
                t1 = (-b - discriminant) / (2.0 * a)
                t2 = (-b + discriminant) / (2.0 * a)
                if t1 >= 0.0 and t1 <= 1.0:
                    first_t = t1
                    first_i = i
                    first_p = start + t1 * V
                    break
                elif t2 >= 0.0 and t2 <= 1.0:
                    first_t = t2
                    first_i = i
                    first_p = start + t2 * V
                    break

        return first_p, first_i, first_t


    def get_waypoint(self, pose):
        # get current position of car
        nearest_point, nearest_dist, t, i = self.nearest_point_on_trajectory(pose, self.waypoints)
        if nearest_dist < self.L:
            lookahead_point, i2, t2 = self.first_point_on_trajectory_intersecting_circle(pose, self.L, self.waypoints,
                                                                                                    i + t, wrap=True)
            if i2 == None:
                return None

            current_waypoint = np.empty((2,))
            # x, y
            current_waypoint[0:2] = self.waypoints[i2, :2]
            return current_waypoint
        elif nearest_dist < 20.:
            return np.append(self.waypoints[i, :])
        else:
            return None


    # @njit(fastmath=False, cache=True)
    def get_actuation(self, pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
        """
        Returns actuation
        """
        # Extract the Waypoint information
        waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point - position)
        speed = 2.0
        if np.abs(waypoint_y) < 1e-6:
            return speed, 0.
        # Define the radius of the arc to follow
        radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)


        # Calculate the steering angle based on the curvature of the arc to follow
        steering_angle = np.arctan(wheelbase / radius)
        # return 0. ,0.
        return speed, steering_angle



def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RandomPlanner()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

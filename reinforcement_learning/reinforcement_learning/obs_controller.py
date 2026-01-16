import torch, os
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.optim import Adam
from reinforcement_learning import TD3Controller, ControlValue, Planner
from collections import namedtuple, deque
import numpy as np
from transforms3d import euler

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from interfaces.msg import Goal, Train, Action

def pure_pursuit(current_point, lookahead_point, yaw_angle, L, wheelbase=0.17145 + 0.15875):
    waypoint_vec = (lookahead_point - current_point).T
    waypoint_y = np.dot(np.array([np.sin(-yaw_angle), np.cos(-yaw_angle)]), waypoint_vec)
    radius = 1 / (2.0 * waypoint_y / L**2)
    return np.arctan(wheelbase / radius)

batch_size = 100
memory_size = 10000
NOISE_CLIP = 0.
POLICY_NOISE = 0.

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, state_dim=2, action_dim=2, capacity=memory_size):
        self.memory = deque([], maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def len(self):
        return len(self.memory)
    
    def transition(self, queue):
        datas = Transition(*zip(*self.memory))
        
        return datas

    def to_tensor(self):
        namespace = Transition(*zip(*self.memory))

        states = torch.FloatTensor(namespace.state).cuda()
        actions = torch.FloatTensor(namespace.action).cuda()
        next_states = torch.FloatTensor(namespace.next_state).cuda()
        rewards = torch.FloatTensor(namespace.reward).cuda()

        return states, actions, next_states, rewards

    def sample(self):
        ind = np.random.randint(0, self.len()-1, size=batch_size)

        states = torch.empty((batch_size, self.state_dim)).cuda()
        actions = torch.empty((batch_size, self.action_dim)).cuda()
        next_states = torch.empty((batch_size, self.state_dim)).cuda()
        rewards = torch.empty((batch_size, 1)).cuda()

        states_tensor, actions_tensor, next_states_tensor, rewards_tensor = self.to_tensor()

        for i, j in enumerate(ind): 
            states[i] = states_tensor[j]
            actions[i] = actions_tensor[j]
            next_states[i] = next_states_tensor[j]
            rewards[i] = rewards_tensor[j]

        return states, actions, next_states, rewards

class Planner():

    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 3000000

    def __init__(self, car_width, wheelbase):
        self.car_width = car_width
        self.angle_offset = -np.pi/4
        self.points = []
        self.lookahead_idx = None
        self.obs = []
        self.lp = [0, 0]

        ## hyper parameters ##
        self.wpts_interval = 0.15
        self.track_boundary = self.car_width/3
        self.SAFE_BOX = 3 # index

        self.psi = 2
        self.w_tr_right = 3
        self.w_tr_left = 4

    def load_waypoints(self, path, delim, rowskip):
        self.waypoints = np.loadtxt(path, delimiter=delim, skiprows=rowskip)

    def pure_pursuit(self, current_point, lookahead_point, pose_theta, L, wheelbase=0.3302):
        waypoint_vec = (lookahead_point - current_point).T
        waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), waypoint_vec)
        radius = 1 / (2.0 * waypoint_y / L**2)
        return np.arctan(wheelbase / radius)
    
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
    

    def get_waypoint(self, pose, waypoints, current_idx, L):
        stacked_waypoints = np.vstack((waypoints, waypoints))

        round = (stacked_waypoints[:, 0] - pose[0])**2 + (stacked_waypoints[:, 1] - pose[1])**2
        in_range = np.where(round < (L)**2)[0]
        in_range = in_range[np.where(in_range - current_idx < 30)[0]]

        if not len(in_range) == 0:
            wpt_pos = stacked_waypoints[np.max(in_range)]
        else:
            wpt_pos = stacked_waypoints[current_idx + 7]

        diff = np.linalg.norm(wpt_pos - waypoints, axis=1)
        return np.argmin(diff)

    def get_nearest_idx(self, pose, yaw, waypoints):
        heading_err = abs(np.arctan2(np.sin(yaw - waypoints[:, self.psi]),
                                     np.cos(yaw - waypoints[:, self.psi])))
        diffs = np.linalg.norm(pose - waypoints[:,:2], axis=1)
        in_range = np.where(heading_err < np.pi/2, diffs, float('inf'))

        nearest_idx = np.argmin(in_range)

        return nearest_idx	
    
    def pose_callback(self, pose, pose_theta, lookahead_distance):
        current_idx = self.get_nearest_idx(pose, pose_theta, self.waypoints)
        centerline = self.waypoints[:, :2]
        self.lookahead_idx = self.get_waypoint(pose, centerline, current_idx, lookahead_distance)

        if self.lookahead_idx == None:
            return

        center_point = centerline[self.lookahead_idx, :]
        track_bound_l = self.waypoints[self.lookahead_idx, self.w_tr_left]
        track_bound_r = self.waypoints[self.lookahead_idx, self.w_tr_right]
        norm_vec = self.calc_norm_vec(centerline, self.lookahead_idx)

        self.points = []
        
        if pose_theta < np.pi:
            i=0
            while(i <= track_bound_l - self.track_boundary):
                self.points.append(-i*norm_vec + center_point)
                i += self.wpts_interval
            self.points.reverse()
            i = self.wpts_interval
            while(i <= track_bound_r - self.track_boundary):
                self.points.append( i*norm_vec + center_point)
                i += self.wpts_interval
        elif pose_theta > np.pi:
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
        

    def plan(self, scan_msg, pose, pose_theta, raceline):
        scan_range = len(scan_msg.ranges)
        scan = self.preprocess_lidar(scan_msg.ranges, scan_range)
        
        if len(self.points) == 0 or self.lookahead_idx == None:
            return None, None, None
        else:    
            if pose_theta > np.pi / 2 and pose_theta < 3* np.pi / 2:
                lidar_position = pose - 0. * np.array([1, np.tan(pose_theta)] / np.linalg.norm(np.array([1, np.tan(pose_theta)])))
            else:
                lidar_position = pose + 0. * np.array([1, np.tan(pose_theta)] / np.linalg.norm(np.array([1, np.tan(pose_theta)])))

            points_on_local = self.global_to_local(np.array(self.points), (pose_theta - np.pi/2), lidar_position)

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
            # cost = np.linalg.norm(np.array(self.points) - pose, axis=1)
            cost = np.linalg.norm(np.array(self.points) - self.waypoints[self.lookahead_idx, :2], axis=1)
            # cost = np.linalg.norm(np.array(self.points) - raceline[np.argmin(np.linalg.norm(pose - raceline[:,:2], axis=1)), :2], axis=1)
            # 조향
            # cost = pure_pursuit(pose, self.points, pose_theta, np.linalg.norm(pose-self.points, axis=1), wheelbase=self.wheelbase)
#
            for obs in obstacle:
                for i in range(obs[0]-self.SAFE_BOX, obs[1]+self.SAFE_BOX):
                    if i >= 0 and i < len(self.points):
                        cost[i] = float('inf')

            if np.min(cost) == float('inf'):
                return self.points, self.lp, self.obs
            
            lookahead_point = self.points[np.argmin(cost)]
            self.lp = lookahead_point

            # lookahead_point = self.waypoints[self.lookahead_idx, :2]
            # self.lp = lookahead_point
            
            return self.points, lookahead_point, self.obs

        # return None, self.waypoints[self.lookahead_idx, :2], None

class Env(Node):
    ## Train Parameters ##
    NUM_EPISODES = 10000
    DISCOUNT_FACTOR = 0.9
    
    def __init__(self):
        super().__init__('env')

        self.declare_parameter('use_sim', True)
        self.declare_parameter('wpt_path', 'src/reinforcement_learning/waypoints/Spielberg_map_centerline.csv')
        self.declare_parameter('wpt_delim', ',')
        self.declare_parameter('wpt_rowskip', 0)
        self.declare_parameter('wheelbase', 0.17145 + 0.15875)
        self.declare_parameter('car_width', 0.31)

        self.wheelbase = self.get_parameter('wheelbase').value
        self.sim = self.get_parameter('use_sim').value
        if self.sim:
            pose_sub_topic = '/ego_racecar/odom'
            odom_sub_topic = '/ego_racecar/odom'
        else:
            pose_sub_topic = '/pf/pose/odom'
            odom_sub_topic = '/odom'

        self.pose_sub = self.create_subscription(Odometry, pose_sub_topic, self.pose_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, odom_sub_topic, self.odom_callback, 1)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)
        self.train_sub = self.create_subscription(Train, '/train', self.save, 10)
        self.ld_sub = self.create_subscription(Action, '/lookahead_distance', self.get_lookahead_distance, 1)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.visualize_pub = self.create_publisher(MarkerArray, '/visualize/points', 10)

        self.prev_x = None
        self.prev_y = None
        self.prev_theta = None
        self.prev_state = None
        self.action = [0,0]
        self.fric = 0.0
        self.eps = 0
        self.velocity = 0.0
        self.lookahead_distance = 3.0
        
        self.min_speed = 1.0
        self.max_speed = 6.0
        self.min_steer = -0.5
        self.max_steer = 0.5
        self.max_av = 0.3804 / 3

        self.actor = TD3Controller().cuda()
        self.actor.load_state_dict(torch.load(os.getcwd() + '/controller_actor.pt'))
        # self.actor_target = TD3Controller().cuda()
        # self.actor_target.load_state_dict(self.actor.state_dict())
        # self.critic_1 = ControlValue().cuda()
        # self.critic_2 = ControlValue().cuda()
        # self.critic_target_1 = ControlValue().cuda()
        # self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        # self.critic_target_2 = ControlValue().cuda()
        # self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        # self.actor_optimizer = Adam(self.actor.parameters())
        # self.critic_optimizer = Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=0.001)

        self.replay_memory = ReplayMemory()

        self.planner = Planner(car_width=self.get_parameter('car_width').value,
                               wheelbase=self.get_parameter('wheelbase').value)
        self.planner.load_waypoints(self.get_parameter('wpt_path').value, self.get_parameter('wpt_delim').value, self.get_parameter('wpt_rowskip').value)
        self.planner.wheelbase = self.wheelbase

        self.raceline = np.loadtxt('src/reinforcement_learning/waypoints/test_optimal.csv', delimiter=',')

    def odom_callback(self, odom_msg):
        self.velocity = odom_msg.twist.twist.linear.x

    def get_lookahead_distance(self, ld_msg):
        self.lookahead_distance = ld_msg.lookahead_distance

    def calc_goal_currnt_vec(self, x, y, heading, goal_x, goal_y):
        rotation_matrix = np.array([[np.cos(heading), -np.sin(heading)],
                                    [np.sin(heading), np.cos(heading)]])

        goal = (np.array([goal_x, goal_y]) - np.array([x, y])).T
        goal = goal@rotation_matrix

        vector_norm = np.linalg.norm(goal)
        orientation = np.arctan2(goal[1], goal[0])
        
        return vector_norm, orientation
    
    def calc_diff_prev_desired(self, x, y, heading, prev_x, prev_y, prev_theta, prev_action):
        speed = prev_action[0]
        acted_theta = prev_theta + prev_action[1]
        
        if acted_theta < 0:
            acted_theta += 2*np.pi
        acted_theta = acted_theta % (2*np.pi)

        if acted_theta >= np.pi/2 and acted_theta <= 3*np.pi/2:
            acted_orientation = -np.array([1, np.tan(acted_theta)]) / np.linalg.norm(np.array([1, np.tan(acted_theta)]))
        else:
            acted_orientation = np.array([1, np.tan(acted_theta)]) / np.linalg.norm(np.array([1, np.tan(acted_theta)]))
            
        goal = np.array([prev_x, prev_y]) + speed * acted_orientation
        pos_diff = np.array([x, y]) - np.array([prev_x, prev_y])

        if x == prev_x and y == prev_y:
            return 0

        proj_cos = np.dot(goal, pos_diff) / (np.linalg.norm(goal)*(np.linalg.norm(pos_diff)))
        proj = np.linalg.norm(pos_diff) * proj_cos

        return np.linalg.norm(goal) - proj
    
    """
    def calc_diff_prev_desired(self, current_speed, desired_speed, prev_heading, curr_heading, desired_steer):
        diff_speed = abs(desired_speed - current_speed)
        diff_speed = min(diff_speed, 1)

        diff_steer = abs(prev_heading - curr_heading)
        if diff_steer > np.pi:
            diff_steer = 2*np.pi - diff_steer
        
        return diff_speed, diff_steer
    """
    
    def pose_callback(self, pose_msg):
        ## update position
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y
        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, 
                pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        theta = euler.quat2euler(quat, axes='sxyz')[0]
        if theta < 0:                   # 0 <= theat <= 2pi
            theta = 2*np.pi + theta

        self.pose_x = x
        self.pose_y = y
        self.pose_theta = theta

    def scan_callback(self, scan_msg):
        scans = np.array(scan_msg.ranges)

        self.planner.pose_callback(np.array([self.pose_x, self.pose_y]), self.pose_theta, self.lookahead_distance*1.2)
        self.points, self.lp, self.obs = self.planner.plan(scan_msg, np.array([self.pose_x, self.pose_y]), self.pose_theta, self.raceline)
        lookahead_point = self.lp
        print(lookahead_point)

        # if self.obs:
        #     goal_msg = Goal()
        #     goal_msg.goal_point.x = lookahead_point[0]
        #     goal_msg.goal_point.y = lookahead_point[1]
        # else:
        #     curr_idx = self.planner.get_nearest_idx(np.array([self.pose_x, self.pose_y]), self.pose_theta, self.raceline)
        #     waypoint_idx = self.planner.get_waypoint(np.array([self.pose_x, self.pose_y]), self.raceline, curr_idx, self.lookahead_distance)
        #     goal_msg = Goal()
        #     goal_msg.goal_point.x = self.raceline[waypoint_idx, 0]
        #     goal_msg.goal_point.y = self.raceline[waypoint_idx, 1]
        #     self.lp = self.raceline[waypoint_idx, :2]

        goal_msg = Goal()
        goal_msg.goal_point.x = lookahead_point[0]
        goal_msg.goal_point.y = lookahead_point[1]
        self.forward(goal_msg)

    def get_action(self, action):
        if action.ndim == 1:
            action = action.unsqueeze(0)
        speed = action[:,0]
        steer = action[:,1]
        
        speed = torch.tensor(self.min_speed) + (speed-(-1)) / (1-(-1)) * (self.max_speed - self.min_speed)
        steer = torch.tensor(self.min_steer) + (steer-(-1)) / (1-(-1)) * (self.max_steer - self.min_steer)

        return speed, steer

    def reward_function(self, prev_target, prev_pos, curr_pos, prev_heading, action):
        speed, steering_angle = action[0], action[1]

        ## Vector calculation from the previous vehicle point of view to the target point and current point
        rotate = np.array([[np.cos(prev_heading - np.pi/2), -np.sin(prev_heading - np.pi/2)],
                           [np.sin(prev_heading - np.pi/2), np.cos(prev_heading - np.pi/2)]])
        curr_pos = (curr_pos - prev_pos) @ rotate
        prev_target = (prev_target - prev_pos) @ rotate
        # prev_target = prev_target * speed / 0.04  # 그 속도에서 안정적인 조향 훈련할 수 있음
        prev_target = prev_target * self.max_av

        proj_cos = np.dot(curr_pos, prev_target) / (np.linalg.norm(curr_pos) * np.linalg.norm(prev_target))

        if np.sign(curr_pos[1]) != np.sign(prev_target[1]):
            reward = -np.abs(curr_pos[1])
        elif np.sign(steering_angle) == np.sign(prev_target[0]):  # steering penalty
            reward = -np.abs(steering_angle)
            # reward = np.tanh(reward)
        else:
            # reward = np.linalg.norm(curr_pos - prev_target) / np.linalg.norm(prev_target)
            reward = np.linalg.norm(curr_pos) * proj_cos / np.linalg.norm(prev_target)
            if reward >= 1.:
                reward = 1 - reward
        speed_reward =  (speed-self.min_speed) / (self.max_speed-self.min_speed)
        if speed_reward >= 1.:
            speed_reward = 2 - speed_reward
        return reward #+ speed_reward*0.5

            
    def reset(self):
        self.pose_x = None
        self.prev_x = None
        self.prev_y = None
        self.prev_theta = None
        self.prev_state = None
        self.action = [0,0]

        self.fric = 0.0
        self.collision = False

        return 

    def forward(self, goal_msg):
        # if self.pose_x is None:
        #     return
        pose = np.array([self.pose_x, self.pose_y])
        goal = np.array([goal_msg.goal_point.x, goal_msg.goal_point.y])

        ## get state
        norm, ori = self.calc_goal_currnt_vec(self.pose_x, self.pose_y, self.pose_theta, goal_msg.goal_point.x, goal_msg.goal_point.y)
        # self.get_logger().info(f'norm: {norm}, ori: {ori}')
        # self.fric = self.calc_diff_prev_desired(self.pose_x, self.pose_y, self.pose_theta, self.prev_x, self.prev_y, self.prev_theta, self.action)
        # err_vel, err_str = self.calc_diff_prev_desired(self.velocity, self.action[0], self.pose_theta, self.prev_theta, self.action[1])
        state = [norm, ori]

        ## add to training set
        if self.prev_state != None:
            reward = float(self.reward_function(np.array([self.prev_goal_x, self.prev_goal_y]), np.array([self.prev_x, self.prev_y]), pose, self.prev_theta, self.action))
            print(reward)
            self.replay_memory.push(self.prev_state, self.output, state, [reward])
            # self.get_logger().info(f'Reward: {reward}')

        output = self.actor(torch.tensor(state).to(torch.float32).cuda())

        speed, steer = self.get_action(output)
        speed, steer = float(speed), float(steer)

        print(f'Speed: {speed}, Steering: {steer}')
        
        self.action = [speed, steer]
        self.output = [float(output[0]), float(output[1])]
        # self.action = [float(action[0]), float(action[1])]

        # curr_idx = self.planner.get_nearest_idx(np.array([self.pose_x, self.pose_y]), self.pose_theta, self.raceline)
        # waypoint_idx = self.planner.get_waypoint(np.array([self.pose_x, self.pose_y]), self.raceline, curr_idx, self.lookahead_distance)
        # speed = self.raceline[waypoint_idx, 4] *0.6
        # steer = pure_pursuit(pose, goal, self.pose_theta, self.lookahead_distance)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = steer
        self.drive_pub.publish(drive_msg)

        self.prev_state = state
        self.prev_goal_x = goal_msg.goal_point.x
        self.prev_goal_y = goal_msg.goal_point.y
            
        self.prev_x = self.pose_x
        self.prev_y = self.pose_y
        self.prev_theta = self.pose_theta

        # self.train()

        self.visualize()


    def update_net(self, train_net, target_net):
        for train_param, target_param in zip(train_net.parameters(), target_net.parameters()):
            target_param.data.copy_(train_param.data * 0.995 + target_param * 0.005)
            target_param.requires_grad = False


    def train(self, num_epochs=2):
        if self.replay_memory.len() < batch_size:
            return
        states, actions, next_states, rewards = self.replay_memory.sample()

        for epoch in range(num_epochs):
            noise = torch.normal(torch.zeros(actions.size()), POLICY_NOISE)
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            next_actions = self.actor_target(next_states + noise.cuda())
            # next_speed, next_steer = self.get_action(next_actions)
            # next_actions = torch.stack((next_speed, next_steer), dim=1).cuda()
            next_critic_q1 = self.critic_target_1(torch.cat([next_states, next_actions], dim=1))
            next_critic_q2 = self.critic_target_2(torch.cat([next_states, next_actions], dim=1))
            next_q = torch.min(next_critic_q1, next_critic_q2)
            target_q = rewards + (self.DISCOUNT_FACTOR * next_q).detach()

            critic_q1 = self.critic_1(torch.cat([states, actions], dim=1))
            critic_q2 = self.critic_2(torch.cat([states, actions], dim=1))
            critic_loss = mse_loss(critic_q1, target_q) + mse_loss(critic_q2, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            # print('Critic Loss: ', critic_loss)

        self.update_net(self.critic_1, self.critic_target_1)
        self.update_net(self.critic_2, self.critic_target_2)

        # if critic_loss < 0.1:
        for epoch in range(num_epochs):
            # update actor
            # actor_actions = self.actor(states)
            # actor_speeds, actor_steers = self.get_action(actor_actions)
            # actor_actions = torch.stack((actor_speeds, actor_steers), dim=1).cuda()
            actor_loss = -self.critic_1(torch.cat([states, self.actor(states)], dim=1)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        print(actor_loss)

        self.update_net(self.actor, self.actor_target)

        self.eps += 1

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
        
        # marker = Marker()
        # marker.header.frame_id = "/map"
        # marker.id = 0
        # marker.ns = "target_waypoints"
        # marker.type = 1
        # marker.action = 0
        # marker.pose.position.x = self.lp[0]
        # marker.pose.position.y = self.lp[1]
              
        # marker.color.a = 1.0
        # marker.color.r = 1.0
        # marker.color.g = 1.0
        # marker.color.b = 1.0

        # marker.scale.x = 0.05
        # marker.scale.y = 0.05
        # marker.scale.z = 0.05

        # marker.lifetime.nanosec = int(1e8)

        # marker_arr.markers.append(marker)

        self.visualize_pub.publish(marker_arr)

    def save(self, train_msg):
        if train_msg.save == True:
            # ros2 topic pub --once /train interfaces/msg/Train "{save: True}"
            torch.save(self.critic_1.state_dict(), f'controller_critic.pt')
            torch.save(self.actor.state_dict(), f'controller_actor.pt')
        if train_msg.done:
            # self.reset()
            print("========================done============================")

def main(args=None):
    rclpy.init(args=args)
    node = Env()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
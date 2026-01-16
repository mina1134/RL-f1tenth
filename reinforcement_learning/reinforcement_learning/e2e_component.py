import torch
from torch.nn.functional import mse_loss
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from numba import njit
import os, csv, yaml
from argparse import Namespace
from transforms3d import euler
from scipy.interpolate import splev, splprep
from f1tenth_benchmarks.drl_racing.training_utils import OffPolicyBuffer, DoubleQNet, DoublePolicyNet
from reinforcement_learning import EndToEndPolicy, QNet

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from interfaces.msg import Train
from planner import PurePursuit

# hyper parameters
BATCH_SIZE = 32
GAMMA = 0.99
tau = 0.005
# NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.2
STEP_FREQUENCY = 0.3 #(s)

def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def get_current_idx(pose=np.ndarray, waypoints=np.ndarray):
    dists = np.linalg.norm(waypoints - pose, axis=1)
    nearest_idx = np.argmin(dists)
    return nearest_idx

def get_waypoint_idx(pose, waypoints, L):
    diffs = np.linalg.norm(pose - waypoints, axis=1)
    current_idx = np.argmin(diffs)

    stacked_waypoints = np.vstack((waypoints, waypoints))

    round = (stacked_waypoints[:, 0] - pose[0])**2 + (stacked_waypoints[:, 1] - pose[1])**2
    in_range = np.where(round < (L)**2)[0]
    in_range = in_range[np.where(in_range - current_idx < 30)[0]]

    if not len(in_range) == 0:
        wpt_pos = stacked_waypoints[np.max(in_range)]
    else:
        wpt_pos = stacked_waypoints[current_idx + 5]

    diff_from_wp = np.linalg.norm(wpt_pos - waypoints, axis=1)
    return np.argmin(diff_from_wp)

@njit(cache=True)
def pid(speed, steer, current_speed, current_steer, max_sv=3.2, max_a=9.51, max_v=20.0, min_v=-5.0):
    # steering
    steer_diff = steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        sv = (steer_diff / np.fabs(steer_diff)) * max_sv
    else:
        sv = 0.0

    # accl
    vel_diff = speed - current_speed
    # currently forward
    if current_speed > 0.:
        if (vel_diff > 0):
            # accelerate
            kp = 10.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # braking
            kp = 10.0 * max_a / (-min_v)
            accl = kp * vel_diff
    # currently backwards
    else:
        if (vel_diff > 0):
            # braking
            kp = 2.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # accelerating
            kp = 2.0 * max_a / (-min_v)
            accl = kp * vel_diff

    return accl, sv

class EndToEndTrain(Node):
    INITIAL_POSE = np.array([14.01, 0.40])
    INITIAL_HEADING = 3.14
    
    def __init__(self):
        super().__init__('e2e_component')
        
        self.declare_parameter('actor_params', 'EndToEnd_actor')
        self.declare_parameter('critic_params', 'EndToEnd_critic')
        self.declare_parameter('package_name')
        self.declare_parameter('waypoint_dir')
        self.declare_parameter('waypoint_file')
        self.declare_parameter('delimiter')
        self.declare_parameter('x_s')
        self.declare_parameter('use_sim')

        self.package_name = self.get_parameter('package_name').value
        self.actor_params = self.get_parameter('actor_params').value
        self.critic_params = self.get_parameter('critic_params').value
        self.x_s = self.get_parameter('x_s').value
        self.sim = self.get_parameter('use_sim').value

        param_file = os.getcwd() + '/src/reinforcement_learning/config/EndToEnd.yaml'
        with open(param_file, 'r') as file:
            self.planner_params = Namespace(**yaml.load(file, Loader=yaml.FullLoader))
        self.skip_n = int(1080 / self.planner_params.number_of_beams)
        self.scan_buffer = np.zeros((self.planner_params.n_scans, self.planner_params.number_of_beams))
        self.state_dim = self.planner_params.number_of_beams *2 + 3
        self.action_dim = 2

        # self.actor = torch.load(os.path.join('src', self.package_name, 'log', f'{self.actor_params}.pt')).cuda()
        self.actor = EndToEndPolicy(self.state_dim, self.action_dim).cuda()
        self.actor_target = EndToEndPolicy(self.state_dim, self.action_dim).cuda()
        # self.actor_target = DoublePolicyNet(self.state_dim, self.action_dim).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1 = QNet(self.state_dim, self.action_dim).cuda()
        # self.critic_1 = torch.load(os.path.join('src', self.package_name, 'log', f'{self.critic_params}1.pt')).cuda()
        self.critic_target_1 = QNet(self.state_dim, self.action_dim).cuda()
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_2 = QNet(self.state_dim, self.action_dim).cuda()
        # self.critic_2 = torch.load(os.path.join('src', self.package_name, 'log', f'{self.critic_params}2.pt')).cuda()
        self.critic_target_2 = QNet(self.state_dim, self.action_dim).cuda()
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=1e-3)

        self.replay_buffer = OffPolicyBuffer(self.state_dim, self.action_dim)

        self.pose_topic = '/ego_racecar/odom' if self.sim else '/pf/pose/odom'
        self.odom_topic = '/ego_racecar/odom' if self.sim else '/odom'

        self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 1)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        if self.sim:
            self.train_sub = self.create_subscription(Train, '/train', self.done_callback, 1)

        self.velocity = 0.0
        self.pose = self.INITIAL_POSE
        self.pose_theta = self.INITIAL_HEADING
        self.prev_pose, self.prev_theta = self.INITIAL_POSE, self.INITIAL_HEADING
        self.nn_state, self.nn_act = np.zeros((self.state_dim,)), np.zeros((self.action_dim,))
        self.action = np.zeros((2,))
        
        self.ref_planner = PurePursuit()
        wpt_path = self.get_parameter('waypoint_dir').value
        wpt_file = self.get_parameter('waypoint_file').value
        delimiter = self.get_parameter('delimiter').value

        self.waypoints = np.loadtxt(os.path.join('src', self.package_name, wpt_path, wpt_file), delimiter=delimiter)
        self.ref_path = self.waypoints[:, self.x_s:self.x_s+2]
        p = np.vstack((self.ref_path[0], self.ref_path))
        diffs = np.linalg.norm(p[:-1] - p[1:], axis=1)
        self.cumsum_diffs = np.cumsum(diffs)

        self.el_lengths = np.linalg.norm(np.diff(self.ref_path, axis=0), axis=1)
        self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.tck = splprep([self.ref_path[:, 0], self.ref_path[:, 1]], k=3, s=0, per=True)[0]
        self.cm_ss = np.linspace(0, self.s_path[-1], int(self.s_path[-1] * 100))
        self.cm_path = np.array(splev(self.cm_ss, self.tck, ext=3)).T

    def pose_callback(self, pose_msg):
        self.pose = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, 
                pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        self.pose_theta = euler.quat2euler(quat, axes='sxyz')[0]

    def odom_callback(self, odom_msg):
        self.velocity = odom_msg.twist.twist.linear.x

    def done_callback(self, train_msg):
        if train_msg.done:
            reward = self.reward_function(True, self.prev_pose, self.prev_theta, self.nn_act)
            self.replay_buffer.add(self.nn_state, self.nn_act, self.nn_state, reward, done=False)
            self.train()

            self.velocity = 0.0
            self.pose = self.INITIAL_POSE
            self.pose_theta = self.INITIAL_HEADING
            self.prev_pose, self.prev_theta = self.INITIAL_POSE, self.INITIAL_HEADING
            self.nn_state, self.nn_act = np.zeros((self.state_dim,)), np.zeros((self.action_dim,))
            self.action = np.zeros((2,))
            self.scan_buffer = np.zeros((self.planner_params.n_scans, self.planner_params.number_of_beams))
        if train_msg.save:
            torch.save(self.actor, 'EndToEnd_actor.pt')
            torch.save(self.critic_1, 'EndToEnd_critic1.pt')
            torch.save(self.critic_2, 'EndToEnd_critic2.pt')


    def train(self, iterations=2):
        if self.replay_buffer.size() < BATCH_SIZE:
            return
        for it in range(iterations):
            state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
            self.update_critic(state, action, next_state, reward, done)
        
            if it % POLICY_FREQUENCY == 0:
                self.update_policy(state)
                
                soft_update(self.critic_1, self.critic_target_1, tau)
                soft_update(self.critic_2, self.critic_target_2, tau)
                soft_update(self.actor, self.actor_target, tau)

    def update_critic(self, state, action, next_state, reward, done):
        noise = torch.normal(torch.zeros(action.size()), POLICY_NOISE)
        noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
        next_action = (self.actor_target(next_state) + noise.cuda()).clamp(-1, 1)

        target_Q1 = self.critic_target_1(next_state, next_action)
        target_Q2 = self.critic_target_2(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (done * GAMMA * target_Q).detach()

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_policy(self, state):
        actor_loss = -self.critic_1(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def calculate_progress_m(self, position):
        if len(position) > 2:
            position = position[:2]
        dists = np.linalg.norm(position - self.ref_path, axis=1)
        minimum_dist_idx = np.argmin(dists)
        if minimum_dist_idx == 0:
            if dists[-1] < dists[1]:
                return self.s_path[-1]  # current shortcut
            else:
                start_ind = 0
                end_ind = int(self.s_path[1] * 100)
        elif minimum_dist_idx == len(dists) - 1:
            if dists[-2] < dists[0]:
                start_ind = int(self.s_path[-2] * 100)
                end_ind = int(self.s_path[-1] * 100)
            else:
                return self.s_path[-1]
        else:
            if dists[minimum_dist_idx + 1] > dists[minimum_dist_idx - 1]:
                minimum_dist_idx -= 1
            start_ind = int(self.s_path[minimum_dist_idx] * 100)
            end_ind = int(self.s_path[minimum_dist_idx + 1] * 100)

        cm_path = self.cm_path[start_ind:end_ind]
        dists = np.linalg.norm(position - cm_path, axis=1)
        s_point_m = self.cm_ss[np.argmin(dists) + start_ind]

        return s_point_m


    def calculate_progress_percent(self, position):
        progress_m = self.calculate_progress_m(position)
        progress_percent = progress_m / self.s_path[-1]

        return progress_percent
    

    def get_ref_act(self, pose, pose_theta):
        lookahead_distance = self.planner_params.tal_constant_lookahead + (self.velocity/self.planner_params.max_speed) * (self.planner_params.tal_variable_lookahead)
        lookahead_idx = get_waypoint_idx(pose, self.ref_path, lookahead_distance)
        lookahead_point = self.ref_path[lookahead_idx]

        if self.velocity < 1:
            return np.array([0.0, 2.0])

        true_lookahead_distance = np.linalg.norm(lookahead_point - pose)
        _ , steer = self.ref_planner.get_actuation(pose_theta, lookahead_point, pose, true_lookahead_distance, 0.33)
        steer = np.clip(steer, -self.planner_params.max_steer, self.planner_params.max_steer)
        speed = self.planner_params.constant_speed
        action = np.array([steer, speed])
        
        return action


    def robust_angle_difference_rad(self, x, y):
        """Returns the difference between two angles in RADIANS
        r = x - y"""
        return np.arctan2(np.sin(x-y), np.cos(x-y))

    def calculate_pose(self, s):
        point = np.array(splev(s, self.tck, ext=3)).T
        dx, dy = splev(s, self.tck, der=1, ext=3)
        theta = np.arctan2(dy, dx)
        pose = np.array([point[0], point[1], theta])
        return pose
    
    def reward_function(self, collision, prev_pose, prev_theta, prev_action):
        # if observation['lap_complete']:
        #     return 1  # complete
        if collision:
            return -1
        """
        prev_action = self.transform_action(prev_action)
        ref_act = self.get_ref_act(prev_pose, prev_theta)
        weighted_difference = np.abs(ref_act - prev_action) / self.planner_params.reward_tal_inv_scales
        reward = self.planner_params.reward_tal_constant * (max(1 - np.sum(weighted_difference), 0))
        """

        progress = self.calculate_progress_percent(self.pose)
        track_pose = self.calculate_pose(progress)
        
        cross_track_distance = np.linalg.norm(track_pose[:2] - self.pose)
        distance_penalty = cross_track_distance * self.planner_params.cth_distance_weight 
        heading_error = abs(self.robust_angle_difference_rad(track_pose[2], self.pose_theta))
        speed_heading_reward = (self.velocity / self.planner_params.cth_max_speed) * np.cos(heading_error) * self.planner_params.cth_speed_weight 
        
        reward = max(speed_heading_reward - distance_penalty, 0)

        return reward


    def transform_obs(self, scan):
        desired_speed = self.action[1]
        desired_steer = self.action[0]
        real_speed = self.velocity
        real_steer = self.robust_angle_difference_rad(self.prev_theta, self.pose_theta)

        speed_err = real_speed - desired_speed
        steer_err = real_steer - desired_steer
            
        self.get_logger().info(f'Velocity is {speed_err}. Steer: {steer_err}')
        
        velocity = self.velocity / self.planner_params.max_speed
        scan = np.array(scan)
        scan = np.clip(scan[::self.skip_n] / self.planner_params.range_finder_scale, 0, 1)

        if self.scan_buffer.all() == 0:
            self.get_logger().info(f'CASE A')
            for i in range(self.scan_buffer.shape[0]):
                self.scan_buffer[i, :] = scan
        else:
            self.get_logger().info(f'CASE B')
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan
        dual_scan = np.reshape(self.scan_buffer, (-1))
        nn_obs = np.concatenate((dual_scan, [velocity], [speed_err], [steer_err]))
        # nn_obs = np.concatenate((dual_scan, [velocity]))

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.planner_params.max_steer
        speed = (nn_action[1] + 1) * (self.planner_params.max_speed / 2 - 0.5) + 1
        speed = min(speed, self.planner_params.max_speed)

        action = np.array([steering_angle, speed])

        return action
    
    def scan_callback(self, scan_msg):
        nn_state = self.transform_obs(scan_msg.ranges).reshape(1, -1)  # state = {prev_scan, current_scan, current_velocity, resistance}
        if np.all(self.nn_state != 0):
            reward = self.reward_function(False, self.prev_pose, self.prev_theta, self.action)
            self.replay_buffer.add(self.nn_state, self.nn_act, nn_state, reward, done=False)
            self.train()
        self.nn_act = self.actor(torch.FloatTensor(nn_state).cuda()).cpu().data.numpy().flatten()
        if EXPLORE_NOISE != 0: 
            self.nn_act = (self.nn_act + np.random.normal(0, EXPLORE_NOISE, size=self.action_dim ))
            self.nn_act = self.nn_act.clip(-1, 1)
        self.action = self.transform_action(self.nn_act)


        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(self.action[1])
        drive_msg.drive.steering_angle = float(self.action[0])
        self.drive_pub.publish(drive_msg)

        self.nn_state = nn_state
        self.prev_pose = self.pose
        self.prev_theta = self.pose_theta


def main(args=None):
    rclpy.init(args=args)
    env = EndToEndTrain()
    print("Start")
    rclpy.spin(env)

    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
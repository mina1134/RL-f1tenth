import torch, os, csv
from torch.nn.functional import mse_loss
from torch.optim import Adam
import numpy as np
from scipy.interpolate import splrep, splev
from reinforcement_learning import LDValue, SACLookaheadPlanner
from transforms3d import euler
from collections import namedtuple, deque

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from interfaces.msg import Train, Action

batch_size = 1000
memory_size = 1000000
file_path = os.path.join('src/reinforcement_learning', 'waypoints', 'test_optimal.csv')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, state_dim, action_dim, capacity=memory_size):
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
    
    def reset(self):
        self.__init__()

class Env(Node):
    def __init__(self):
        super().__init__('environment')
        self.declare_parameter('train', True)
        self.declare_parameter('pose_topic')
        self.declare_parameter('package_name')
        self.declare_parameter('raceline_dir')
        self.declare_parameter('raceline_file')
        self.declare_parameter('raceline_delim')
        self.declare_parameter('x', 0)
        self.declare_parameter('y', 1)
        self.declare_parameter('psi', 2)
        self.declare_parameter('kappa', 3)
        self.declare_parameter('velocity', 4)
        self.declare_parameter('pub_action', False)

        self.train = self.get_parameter('train').value

        pose_sub_topic = self.get_parameter('pose_topic').value

        self.pose_sub = self.create_subscription(Odometry, pose_sub_topic, self.pose_callback, 1)
        self.action_sub = self.create_subscription(Train, '/agent/action', self.action_callback, 1)
        self.obs_pub = self.create_publisher(Train, '/update', 1)

        if self.train:
            self.train_sub = self.create_subscription(Train, '/train', self.check_done, 1)

        package_name = self.get_parameter('package_name').value
        raceline_dir = self.get_parameter('raceline_dir').value
        raceline_file = self.get_parameter('raceline_file').value
        raceline_delim = self.get_parameter('raceline_delim').value
        self.lane = np.loadtxt(os.path.join('src', package_name, raceline_dir, raceline_file), delimiter=raceline_delim)

        self.psi = self.get_parameter('psi').value
        self.curv = self.get_parameter('kappa').value
        self.vel = self.get_parameter('velocity').value

        self.collision = False
        self.done = False
        self.track_err = 0.0
        self.prev_track_err = 0.0
        self.curr_idx = None
        self.pose_theta = 0.0

    def pose_callback(self, pose_msg):
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y
        pose = np.array([x, y])

        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, 
                pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        self.pose_theta = euler.quat2euler(quat, axes='sxyz')[0]
        if self.pose_theta < 0:    # 0 < heading < 2pi
            self.pose_theta = 2*np.pi + self.pose_theta

        dists = np.linalg.norm(pose - self.lane[:, :2], axis=1)
        self.curr_idx = np.argmin(dists)
        track_err = np.linalg.norm(pose - self.lane[self.curr_idx, 0:2])

        self.track_err = track_err

    def reward_function(self, yaw, curr_idx, prev_track_err, track_err, L):
        heading_err = abs(np.arctan2(np.sin(yaw - self.lane[curr_idx, self.psi]),
                                     np.cos(yaw - self.lane[curr_idx, self.psi])))
        heading_err = 1 - np.cos(heading_err)

        total_err = 1 - track_err*2.0 - heading_err - L*0.01

        return np.clip(total_err, -1.0, 1.0)


    def check_done(self, train_msg):
        if train_msg.done:
            self.done = True  

    def action_callback(self, action_msg):
        lookahead_distance = action_msg.action.lookahead_distance
        reward = self.reward_function(self.pose_theta, self.curr_idx, self.prev_track_err, self.track_err, lookahead_distance)
        print(reward)

        agent_msg = action_msg
        if self.done:
            agent_msg.done = True
            self.done = False
        agent_msg.reward = reward
        self.obs_pub.publish(agent_msg)

        self.prev_track_err = self.track_err

class Agent(Node):
    INPUTS = 12
    DISCOUNT_FACTOR = 0.99
    KAPPA_STEP = 0.3 # [m]
    WHEELBASE = 0.3302
    MAX_LD = 2.0
    MIN_LD = 0.5
    TARGET_ENTROPY = 1.0

    def __init__(self):
        super().__init__('agent')
        self.declare_parameter('train', True)
        self.declare_parameter('pose_topic')
        self.declare_parameter('package_name')
        self.declare_parameter('raceline_dir')
        self.declare_parameter('raceline_file')
        self.declare_parameter('raceline_delim')
        self.declare_parameter('x', 0)
        self.declare_parameter('y', 1)
        self.declare_parameter('psi', 2)
        self.declare_parameter('kappa', 3)
        self.declare_parameter('velocity', 4)
        self.declare_parameter('pub_action', False)

        self.train = self.get_parameter('train').value

        pose_sub_topic = self.get_parameter('pose_topic').value
        self.pose_sub = self.create_subscription(Odometry, pose_sub_topic, self.pose_callback, 1)
        self.reward_sub = self.create_subscription(Train, '/update', self.step, 1)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        self.output_pub = self.create_publisher(Train, '/agent/action', 1)
        self.waypoint_pub = self.create_publisher(Marker, '/visualize/waypoint', 10)

        self.pub_ld = False
        if self.get_parameter('pub_action').value:
            self.pub_ld = True
            self.ld_pub =self.create_publisher(Action, '/lookahead_distacne', 1)

        if self.train:
            self.train_sub = self.create_subscription(Train, '/train', self.save, 10)

            self.actor = SACLookaheadPlanner(input=self.INPUTS+1).cuda()
            # self.actor.load_state_dict(torch.load(os.getcwd() + '/lookahead_planner_actor_0.6714738011360168.pt'))
            self.actor_target = SACLookaheadPlanner(input=self.INPUTS+1).cuda()
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_optimizer = Adam(self.actor.parameters(), lr=0.001)
            self.log_alpha = torch.zeros(1, requires_grad=True, device='cuda')  # Use log_alpha.exp() for alpha value to keep alpha above zero. 
            self.alpha_optimizer = Adam([self.log_alpha], lr=0.0003)

            self.critic_1 = LDValue(input=self.INPUTS+2).cuda()
            # self.critic_1.load_state_dict(torch.load(os.getcwd() + '/lookahead_planner_critic.pt'))
            self.critic_target_1 = LDValue(input=self.INPUTS+2).cuda()
            self.critic_target_1.load_state_dict(self.critic_1.state_dict())
            self.critic_2 = LDValue(input=self.INPUTS+2).cuda()
            # self.critic_2.load_state_dict(torch.load(os.getcwd() + '/lookahead_planner_critic.pt'))
            self.critic_target_2 = LDValue(input=self.INPUTS+2).cuda()
            self.critic_target_2.load_state_dict(self.critic_2.state_dict())
            self.critic_optimizer = Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=0.001)
        else:
            self.actor = SACLookaheadPlanner(input=self.INPUTS+1).cuda()
            self.actor.load_state_dict(torch.load(os.getcwd() + '/lookahead_planner_actor.pt'))

        self.pose = None
        self.pose_theta = None
        self.curr_idx = None
        self.curr_state = []
        self.done = False
        self.warning = False
        self.prev_L = None
        self.reward_log = []
        self.max_reward = 0.3
        self.vgain = 0.8

        self.replay_memory = ReplayMemory(state_dim=self.INPUTS+1, action_dim=1)

        package_name = self.get_parameter('package_name').value
        raceline_dir = self.get_parameter('raceline_dir').value
        raceline_file = self.get_parameter('raceline_file').value
        raceline_delim = self.get_parameter('raceline_delim').value
        self.lane = np.loadtxt(os.path.join('src', package_name, raceline_dir, raceline_file), delimiter=raceline_delim)

        self.psi = self.get_parameter('psi').value
        self.curv = self.get_parameter('kappa').value
        self.vel = self.get_parameter('velocity').value

        self.min_delta_L = -0.1
        self.max_delta_L = 0.1


    def get_nearest_idx(self, pose, yaw, trajectory):
        heading_err = abs(np.arctan2(np.sin(yaw - trajectory[:, self.psi]),
                                     np.cos(yaw - trajectory[:, self.psi])))
        diffs = np.linalg.norm(pose - trajectory[:,:2], axis=1)
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
        diff = np.linalg.norm(self.lane[ :-1, :2] - self.lane[1: , :2], axis=1)
        stacked_diff = np.hstack((diff, diff))
        curvs = np.empty((self.INPUTS,))
        marker_arr = MarkerArray()
        for i in range(self.INPUTS):
            curv = self.lane[idx, self.curv]
            cnt = stacked_diff[idx]
            while cnt <= self.KAPPA_STEP:
                cnt += stacked_diff[idx]
                idx += 1
                if idx >= len(self.lane):
                    idx = idx - len(self.lane)
                curv += self.lane[idx, self.curv]
            curv = np.rad2deg(self.lane[idx, self.curv])
            curvs[i] = np.abs(curv)
            idx += 1
            if idx >= len(self.lane):
                idx = idx - len(self.lane)
        
        return curvs


    def pose_callback(self, pose_msg):
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y
        self.pose = np.array([x, y])

        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, 
                pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        self.pose_theta = euler.quat2euler(quat, axes='sxyz')[0]
        if self.pose_theta < 0:    # 0 < heading < 2pi
            self.pose_theta = 2*np.pi + self.pose_theta

        self.curr_idx = self.get_nearest_idx(self.pose, self.pose_theta, self.lane)

        if self.train:
            track_err = np.linalg.norm(self.pose - self.lane[self.curr_idx, 0:2])
            self.track_err = track_err
        else:
            self.track_err = 1e-2

        self.curvs = self.get_kappa(self.curr_idx)

        self.curr_state = np.append(self.curvs, self.track_err)

        if self.train:
            _, _, output = self.actor_target(torch.FloatTensor(self.curr_state).cuda())
        else:
            _, _, output = self.actor(torch.FloatTensor(self.curr_state).cuda())
        lookahead_distance = self.MIN_LD + (self.MAX_LD - self.MIN_LD)/2 * (float(output[0]) + 1.0)

        delta_L = 0.0
        if self.prev_L is not None:
            delta_L = lookahead_distance - self.prev_L
            delta_L = min(max(delta_L, self.min_delta_L), self.max_delta_L)
            lookahead_distance = self.prev_L + delta_L
        # lookahead_distance = 0.3 + 0.3*pose_msg.twist.twist.linear.x
        # lookahead_distance = 0.5 + 0.3/(self.lane[self.curr_idx, self.curv] + 0.05)
        # lookahead_distance = 0.05*pose_msg.twist.twist.linear.x**2 - 0.07*self.lane[self.curr_idx, self.curv] - 0.2*np.linalg.norm(self.pose - self.lane[self.curr_idx, 0:2]) + 0.5

        self.get_logger().info(f'cliped lookahead_distance: {lookahead_distance}')
        
        lookahead_idx = self.get_waypoint(self.pose, self.lane[:, 0:2], self.curr_idx, lookahead_distance)
        lookahead_point = self.lane[lookahead_idx, 0:2]

        waypoint = Marker()
        waypoint.header.frame_id = "map"
        waypoint.id = 0
        waypoint.type = 1
        waypoint.action = 0
        waypoint.pose.position.x = lookahead_point[0]
        waypoint.pose.position.y = lookahead_point[1]
        waypoint.pose.position.z = 0.0
        waypoint.color.a = 1.0
        waypoint.color.r = 1.0
        waypoint.color.g = 0.0
        waypoint.color.b = 0.0
        waypoint.scale.x = 0.1
        waypoint.scale.y = 0.1
        waypoint.scale.z = 0.1
        self.waypoint_pub.publish(waypoint)

        steer = self.pure_pursuit(self.pose, self.pose_theta, lookahead_point, self.WHEELBASE)
        speed = self.lane[lookahead_idx, self.vel]

        if self.pub_ld:
            action_msg = Action()
            action_msg.lookahead_distance = lookahead_distance
            self.ld_pub.publish(action_msg)
        else:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed          = speed * self.vgain
            drive_msg.drive.steering_angle = steer
            self.drive_pub.publish(drive_msg)

        if self.train:
            output_msg = Train()   
            output_msg.observation.input = tuple(self.curr_state)
            output_msg.observation.output = float(output[0])
            output_msg.action.lookahead_distance = lookahead_distance
            self.output_pub.publish(output_msg)

        self.prev_L = lookahead_distance

    def step(self, obs_msg):
        state = obs_msg.observation.input
        action = obs_msg.action.lookahead_distance
        done = obs_msg.done
        reward = obs_msg.reward
        self.reward_log.append(reward)

        if self.train:
            if done:
                if min(self.reward_log) > self.max_reward:
                    self.get_logger().info('Save')
                    self.max_reward = min(self.reward_log)
                    torch.save(self.critic_1.state_dict(), f'lookahead_planner_critic.pt')
                    torch.save(self.actor_target.state_dict(), f'lookahead_planner_actor.pt')
                self.model_update()
                print("========================Done================================", min(self.reward_log))
                self.prev_L = None
                self.reward_log = []
                return
            self.replay_memory.push(state, [action], self.curr_state, [reward])

    def update_net(self, train_net, target_net):
        for train_param, target_param in zip(train_net.parameters(), target_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - 0.005) + train_param.data * 0.005)
            target_param.requires_grad = False

    def model_update(self, num_epochs=100):
        print("Update")
        if self.replay_memory.len() < 2:
            return
        states, actions, next_states, rewards = self.replay_memory.sample()

        alpha = self.log_alpha.exp().detach()
        next_actions, next_log_prob,_ = self.actor_target(next_states)
        next_actions = self.MIN_LD + (self.MAX_LD - self.MIN_LD)/2 * (next_actions + 1.0)

        for epoch in range(num_epochs):
            with torch.no_grad():
                next_critic_q1 = self.critic_target_1(torch.cat([next_states, next_actions], dim=1))
                next_critic_q2 = self.critic_target_2(torch.cat([next_states, next_actions], dim=1))
                next_q = torch.min(next_critic_q1, next_critic_q2)
            critic_q1 = self.critic_1(torch.cat([states, actions], dim=1))
            critic_q2 = self.critic_2(torch.cat([states, actions], dim=1))

            with torch.no_grad():
                target_q = rewards + (self.DISCOUNT_FACTOR*next_q)
            critic_loss = mse_loss(critic_q1, target_q) + mse_loss(critic_q2, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        self.update_net(self.critic_1, self.critic_target_1)
        self.update_net(self.critic_2, self.critic_target_2)
        
        for epoch in range(num_epochs):
            print(epoch)
            actions, log_probs, _ = self.actor(states)
            actions = self.MIN_LD + (self.MAX_LD - self.MIN_LD)/2 * (actions + 1.0)
            actor_loss = (alpha*log_probs - self.critic_target_1(torch.cat([states, actions], dim=1))).mean()
            # print(actor_loss)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            alpha_loss = -(self.log_alpha.cuda() * (log_probs - self.TARGET_ENTROPY).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.update_net(self.actor, self.actor_target)

    def pure_pursuit(self, position, pose_theta, lookahead_point, wheelbase):
        lookahead_distance = np.linalg.norm(self.pose - lookahead_point)
        waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point - position)
        if np.abs(waypoint_y) < 1e-6:
            return 0.0
        radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)
        steering_angle = np.arctan(wheelbase / radius)

        return steering_angle
    

    def reset(self, collision):
        self.prev_state = []
        self.replay_memory.reset()
        return 

    def save(self, train_msg):
        if train_msg.save == True:
            # ros2 topic pub --once /train interfaces/msg/Train "{save: True}"
            torch.save(self.critic_1.state_dict(), f'lookahead_planner_critic.pt')
            torch.save(self.actor.state_dict(), f'lookahead_planner_actor.pt')
                
def main(args=None):
    rclpy.init(args=args)
    env = Env()
    agent = Agent()

    executor = MultiThreadedExecutor()

    executor.add_node(env)
    executor.add_node(agent)

    try:
        executor.spin()
    finally:
        executor.remove_node(env)
        executor.remove_node(agent)
        rclpy.shutdown()

if __name__ == '__main__':
    main()
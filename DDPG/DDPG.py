import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch import optim
import torch.nn.functional as F
'''
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim / 2), 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.net(x)
'''
class Actor(nn.Module): # define the network structure for actor and critic
    def __init__(self, s_dim, hidden_dim, a_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1) # initialization of FC1
        self.out = nn.Linear(30, a_dim)
        self.out.weight.data.normal_(0, 0.1) # initilizaiton of OUT
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions = x * 2 # for the game "Pendulum-v0", action range is [-2, 2]
        return actions

class Critic(nn.Module):
    def __init__(self, s_dim, hidden_dim, a_dim):
        super(Critic, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)
    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        actions_value = self.out(F.relu(x+y))
        return actions_value

class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.memory_num = 0
        self.memory_s = []
        self.memory_a = []
        self.memory_r = []
        self.memory_s_ = []
        self.memory_done = []

    def memorystore(self, s, a, r, s_, done):
        self.memory_num += 1
        self.memory_s.append(s)
        self.memory_a.append(a)
        self.memory_r.append(r)
        self.memory_s_.append(s_)
        self.memory_done.append(done)


class OUNoise(object):
    '''Ornstein–Uhlenbeck噪声
    '''
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu # OU噪声的参数
        self.theta        = theta # OU噪声的参数
        self.sigma        = max_sigma # OU噪声的参数
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()

    def reset(self):
        self.obs = np.ones(self.action_dim) * self.mu
        self.step = 0

    def evolve_obs(self):
        x  = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.obs = x + dx
        return self.obs
    def get_action(self, action):

        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.step / self.decay_period) # sigma会逐渐衰减
        self.step += 1
        return np.clip(action + ou_obs, self.low, self.high) # 动作加上噪声后进行剪切



class DDPG:
    def __init__(self, args):
        self.device = torch.device("cuda:0")
        self.actor = Actor(args.state_dim, args.hidden_dim, args.action_dim).to(self.device)
        self.critic = Critic(args.state_dim, args.hidden_dim, args.action_dim).to(self.device)
        self.actor_target = deepcopy(self.actor).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.memory = Memory()

        self.gamma = args.gamma
        self.repeat_times = args.repeat_times
        self.batch_size = args.batch_size
        self.soft_tau = args.soft_update_tau

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return 2 * action.detach().cpu().numpy()[0]



    def update(self):
        buf_len = self.memory.memory_num

        buf_state = torch.FloatTensor(np.array(self.memory.memory_s)).to(self.device)
        buf_state_ = torch.FloatTensor(np.array(self.memory.memory_s_)).to(self.device)
        buf_action = torch.FloatTensor(np.array(self.memory.memory_a)).to(self.device)
        buf_reward = torch.FloatTensor(np.array(self.memory.memory_r)).unsqueeze(1).to(self.device)
        buf_done = torch.FloatTensor(np.float32(self.memory.memory_done)).unsqueeze(1).to(self.device)

        self.memory.clear()

        for j in range(int(buf_len / self.batch_size * self.repeat_times)):
            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            reward = buf_reward[indices]
            state_ = buf_state_[indices]
            done = buf_done[indices]

            policy_loss = self.critic(state, self.actor(state))
            policy_loss = -policy_loss.mean()
            next_action = self.actor_target(state_)
            target_value = self.critic_target(state_, next_action.detach())
            expected_value = reward + (1.0 - done) * self.gamma * target_value
            expected_value = torch.clamp(expected_value, -np.inf, np.inf)

            value = self.critic(state, action)
            value_loss = nn.MSELoss()(value, expected_value.detach())

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()
            # 软更新
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )

    def save(self, path="actor.pt"):
        torch.save(self.actor.state_dict(), path)

    def load(self, path="actor.pt"):
        self.actor.load_state_dict(torch.load(path))


















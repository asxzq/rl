# DQN本质上只用了一个网络
# 为了解决q_target计算不稳定问题，引入另外的target_net，得到NatureDQN
# 此代码实际为 NatureDQN
#  Q-learning，
#  Double Q-learning（解决Q-learning值函数过估计问题），
#  DQN（解决Q-learning大状态空间、动作空间问题），
#  Double DQN（解决DQN值函数过估计问题），
#  PER-DQN（解决经验回放的采样问题）
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np

import torch.nn.functional as F
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # 随机采出小批量转移
        state, action, reward, next_state, done = zip(*batch)  # 解压成状态，动作等
        return state, action, reward, next_state, done
    '''
    def __len__(self):
        return len(self.buffer)
    '''

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """ 初始化q网络，为全连接网络
            input_dim: 输入的特征数即环境的状态数
            output_dim: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    def __init__(self, state_dim, action_dim):
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon_start = 0.90  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 500  # e-greedy策略中epsilon的衰减率
        # 网络相关参数
        self.hidden_dim = 256  # hidden size of net
        self.action_dim = action_dim  # 总的动作个数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.policy_net = MLP(state_dim, action_dim,hidden_dim=self.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim,hidden_dim=self.hidden_dim).to(self.device)
        # 训练相关参数
        self.gamma = 0.95  # 强化学习中的折扣因子
        self.batch_size = 64  # mini-batch SGD中的批量大小
        self.lr = 0.0001  # 学习率
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr) # 优化器
        self.memory_capacity = 100000  # 经验回放的容量
        self.memory = ReplayBuffer(self.memory_capacity)

    def choose_action(self, state):
        self.frame_idx += 1
        epsilon = self.epsilon_end + \
            (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.frame_idx / self.epsilon_decay)
        if random.random() > epsilon:
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            q_values = self.policy_net(state)
            action = torch.argmax(q_values).item()
            #action = q_values.max(1)[1].item() # 选择Q值最大的动作
        else:
            action = random.randrange(self.action_dim)
        return action

    def predict(self,state):
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        q_values = self.policy_net(state)
        action = q_values.max(1)[1].item()
        return action

    def update(self):
        if len(self.memory.buffer) < self.batch_size: # 当memory中不满足一个批量时，不更新策略
            return

        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        # 转为张量
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)

        # 计算当前状态(s_t,a)对应的Q(s_t, a)
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        # 计算下一时刻的状态(s_t_,a)对应的Q值
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1-done_batch)
        # 计算均方根损失
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

        # 优化更新模型
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self):
        torch.save(self.target_net.state_dict(), 'target.pth')
        torch.save(self.policy_net.state_dict(), 'policy.pth')


    def load(self):
        self.target_net.load_state_dict(torch.load('target.pth'))
        self.policy_net.load_state_dict(torch.load('policy.pth'))

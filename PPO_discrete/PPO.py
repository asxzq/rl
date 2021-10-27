import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from torch.autograd import Variable

import torch.optim as optim
from torch.distributions.categorical import Categorical


class Actor(nn.Module):
    def __init__(self,state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)  # 保证总概率为 1
        )

    def forward(self, state):
        dist = self.actor(state)
        # Categorical离散分布
        dist = Categorical(dist)
        return dist


class Critic(nn.Module):
    def __init__(self, state_dim,hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        return value


class PPO:
    def __init__(self, state_dim, hiddem_dim, action_dim, batch_size):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check gpu
        self.actor_learningrate = 0.0002
        self.critic_learningrate = 0.0002
        self.actor = Actor(state_dim,hiddem_dim,action_dim).to(self.device)
        self.critic = Critic(state_dim,hiddem_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learningrate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learningrate)
        self.memory_states = []
        self.memory_actions = []
        self.memory_probs = []
        self.memory_values = []
        self.memory_rewards = []
        self.memory_dones = []
        # 训练参数
        self.eps = 1e-7
        self.n_epochs = 4
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.gae_lambda = 0.95
        self.advantages_norm = False
        self.c1 = 0.5

    def update_lr(self,lr):
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def memory(self, state, action, prob, value, reward, done):
        self.memory_states.append(state)
        self.memory_actions.append(action)
        self.memory_probs.append(prob)
        self.memory_values.append(value)
        self.memory_rewards.append(reward)
        self.memory_dones.append(done)

    def memory_sample(self):
        # 生成等差数列，方便数据索引
        batch_step = np.arange(0, len(self.memory_states), self.batch_size)
        # 生成初始化索引序号
        indices = np.arange(len(self.memory_states), dtype=np.int64)
        # 打乱索引序号
        np.random.shuffle(indices)
        # 把索引序号给分组
        batches = [indices[i:i + self.batch_size] for i in batch_step]
        return np.array(self.memory_states), np.array(self.memory_actions), np.array(self.memory_probs), \
               np.array(self.memory_values), np.array(self.memory_rewards), np.array(self.memory_dones), batches

    def memory_clear(self):
        self.memory_states = []
        self.memory_actions = []
        self.memory_probs = []
        self.memory_values = []
        self.memory_rewards = []
        self.memory_dones = []

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.device)
        # 生成分布
        dist = self.actor(state)
        print(dist)
        # 输出v(s)
        value = self.critic(state)
        # 根据分布采样,生成动作
        action = dist.sample()
        # dist分布下，action的概率，log p(a)
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, probs, value


    def learn(self):
        # 一次训练循环n_epochs次
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory_sample()
            values = vals_arr[:]
            # 计算优势 At
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            # 此处没用迭代，速度降低了
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            # 归一化
            if self.advantages_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
            advantage = torch.tensor(advantage).to(self.device)

            values = torch.tensor(values).to(self.device)
            # 用已经分好的组进行训练
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)
                # 计算新的 log p(a)
                dist = self.actor(states)
                new_probs = dist.log_prob(actions)
                # 计算重要性采样
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                # 计算loss1
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # 获得新的v(s)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                # 此处应该用递推的，不知道此处表达式是否正确
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                # 加和反向传播和分开反向传播几乎没区别，SGD是完全没区别
                total_loss = actor_loss + self.c1 * critic_loss

                self.loss = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                total_loss.backward()

                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory_clear()

    def save(self):
        torch.save(self.actor.state_dict(), 'actor.pt')
        torch.save(self.critic.state_dict(), 'critic.pt')

    def load(self):
        self.actor.load_state_dict(torch.load('actor.pt'))
        self.critic.load_state_dict(torch.load('critic.pt'))


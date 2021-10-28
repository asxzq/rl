import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import torch
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
        )

    def forward(self, state):
        actor_out = self.actor(state)
        action_prob = F.softmax(actor_out, dim=1)
        dict = Categorical(probs=action_prob)
        return dict


class Critic(nn.Module):
    def __init__(self, state_dim,hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        return value



class A2C:
    def __init__(self,state_dim,action_dim):
        self.gamma = 0.95
        self.device = torch.device("cuda")
        self.hidden_dim = 64
        self.lr = 0.00003
        self.actor = Actor(state_dim, self.hidden_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.device)
        # 生成分布
        dist = self.actor(state)
        # 输出v(s)
        value = self.critic(state)
        # 根据分布采样,生成动作
        action = dist.sample()
        # dist分布下，action的概率，log p(a)
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, probs, value

    def learn(self, state, action, prob_log, reward, value, state_, done):
        state_ = torch.tensor([state_], dtype=torch.float).to(self.device)
        value_ = torch.squeeze(self.critic(state_)).item()
        loss_actor = torch.tensor(- 1 * (reward + value - value_) * prob_log)
        loss_critic = torch.tensor((reward + self.gamma * value - value_) ** 2)
        loss_actor.requires_grad_(True)
        loss_critic.requires_grad_(True)
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

    def save(self):
        torch.save(self.actor.state_dict(), 'actor.pt')
        torch.save(self.critic.state_dict(), 'critic.pt')

    def load(self):
        self.actor.load_state_dict(torch.load('actor.pt'))
        self.critic.load_state_dict(torch.load('critic.pt'))

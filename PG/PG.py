import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable

class MLP(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(MLP, self).__init__()
        # 24和36为hidden layer的层数，可根据state_dim, action_dim的情况来改变
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # 此处输出维度为1, 表示向左走的概率

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class PG:
    def __init__(self,state_dim,hiddem_dim,batch_size):
        self.model = MLP(state_dim,hiddem_dim)
        self.memory_action = []
        self.memory_state = []
        self.memory_reward = []
        self.gamma = 0.99
        self.learning_rate = 0.01
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.batch_size = batch_size

    def memory(self, s, reward, a):
        self.memory_state.append(s)
        self.memory_reward.append(reward)
        self.memory_action.append(float(a))

    def memory_clear(self):
        self.memory_state = []
        self.memory_action = []
        self.memory_reward = []

    def choose_action(self, state):
        state = torch.FloatTensor(state)
        state = Variable(state)
        p_left = self.model(state)
        m = Bernoulli(p_left)  # 伯努利分布
        action = m.sample()

        #print(m)
        action = action.data.numpy().astype(int)[0]
        return action


    def choose_action_(self, state):
        state = torch.FloatTensor(state)
        state = Variable(state)
        p_left = self.model(state)
        p=p_left.data.numpy()[0]
        action = 0 if p<=0.5 else 1
        return action

    def learn(self):
        # 计算 对于每个t，i>t ai,ri的总回报
        running_add = 0
        #print(self.memory_reward)
        num = len(self.memory_state)
        for i in reversed(range(num)):
            if self.memory_reward[i]==0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + self.memory_reward[i]
                self.memory_reward[i] = running_add

        # 归一化
        reward_mean = np.mean(self.memory_reward)
        reward_std = np.std(self.memory_reward)

        for i in range(num):
            self.memory_reward[i] = (self.memory_reward[i] - reward_mean) / reward_std

        #print(self.memory_reward)


        # Gradient Desent
        self.optimizer.zero_grad()

        for i in range(num):
            state = self.memory_state[i]
            state = Variable(torch.from_numpy(state).float())
            action = Variable(torch.FloatTensor([self.memory_action[i]]))
            reward = self.memory_reward[i]

            p_left = self.model(state)
            m = Bernoulli(p_left)
            loss = -m.log_prob(action) * reward  # Negtive score function x reward
            #print(loss, m, action, reward)
            loss.backward()

        self.optimizer.step()

        if num >= 200:
            self.learning_rate = 0.0005
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        elif num>=100:
            self.learning_rate = 0.003
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
    def save(self):
        torch.save(self.model.state_dict(), 'pg.pt')

    def load(self):
        self.model.load_state_dict(torch.load('pg.pt'))
